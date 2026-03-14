"""Main agent decision loop called from the /act endpoint.

Orchestrates HTML pruning, Page IR construction, prompt building, LLM calls,
response validation, and action building into a single ``decide()`` function.
"""

from __future__ import annotations

import logging
from urllib.parse import urlsplit

from agent.actions import build_action, normalize_url, preserve_seed, validate_and_fix
from agent.classifier import (
    AddFilmFields,
    EditFilmFields,
    FilterDropdowns,
    ProfileFields,
    SearchFields,
    TaskType,
    classify_task,
    detect_add_film_fields,
    detect_contact_fields,
    detect_edit_film_fields,
    detect_filter_dropdowns,
    detect_login_fields,
    detect_logout_target,
    detect_profile_fields,
    detect_registration_fields,
    detect_search_input,
    extract_add_film_values_from_prompt,
    extract_credentials_from_prompt,
    extract_edit_values_from_prompt,
    extract_film_name_from_prompt,
    extract_filter_criteria_from_prompt,
    extract_profile_values_from_prompt,
    find_button_by_label,
    find_confirm_button,
    find_film_details_link,
    find_remove_button_for_film,
    find_tab_button,
    find_watchlist_button,
    get_add_film_action,
    get_contact_action,
    get_edit_film_action,
    get_login_action,
    get_logout_action,
    get_profile_action,
    get_registration_action,
    get_search_action,
)
from agent.prompts import (
    build_system_prompt,
    build_user_prompt,
    format_history_entry,
)
from agent.state import check_loop, clear_task_state, get_action_signature
from llm.client import LLMClient
from llm.parser import parse_llm_json
from models.actions import NavigateAction, WaitAction
from models.request import ActRequest
from models.response import ActResponse
from parsing.candidates import extract_candidates
from parsing.page_ir import build_page_ir
from parsing.pruning import prune_html

logger = logging.getLogger("agent")

# Module-level singleton for LLM client (lazy init).
_llm_client: LLMClient | None = None


def _get_llm_client() -> LLMClient:
    """Return the module-level LLM client, creating it on first use."""
    global _llm_client  # noqa: PLW0603
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def _build_history_lines(history: list[dict]) -> list[str]:
    """Convert the evaluator's history list into formatted history strings.

    The evaluator provides entries with keys like ``action``, ``url``,
    ``exec_ok``, ``error``. Each is formatted as a readable summary.
    """
    lines: list[str] = []
    for i, entry in enumerate(history):
        action_type = entry.get("action", "unknown")
        # Try to extract a meaningful element description
        element_text = entry.get("element_text", entry.get("text", ""))
        url = entry.get("url", "")
        exec_ok = entry.get("exec_ok", True)

        if exec_ok:
            result = "success"
        else:
            error = entry.get("error", "failed")
            result = f"error: {error}"

        # Check if URL changed from previous entry
        url_changed = None
        if i > 0 and url:
            prev_url = history[i - 1].get("url", "")
            if url != prev_url:
                url_changed = url

        lines.append(
            format_history_entry(
                step=i + 1,
                action_type=action_type,
                element_text=element_text,
                result=result,
                url_changed=url_changed,
            )
        )
    return lines


def decide(request: ActRequest) -> ActResponse:
    """Main decision function called from the /act endpoint.

    Orchestrates:
    1. HTML pruning (single parse via prune_html)
    2. Candidate extraction from pruned soup
    3. Task classification and hard-coded login bypass (pre-LLM)
    4. Page IR construction
    5. Steps remaining computation
    6. History formatting
    7. Loop detection
    8. LLM prompt construction and call with cost tracking
    9. Response validation with fallback (no re-prompt for invalid IDs)
    10. Retry logic for JSON parse errors only

    Returns:
        An ``ActResponse`` with the chosen action(s), or empty actions
        for "done" signal, or a WaitAction fallback on exhausted retries.
    """
    # 1. Prune HTML (single parse -- replaces dual-parse pattern)
    pruned_soup = prune_html(request.snapshot_html)
    title = pruned_soup.title.string if pruned_soup.title and pruned_soup.title.string else ""

    # 2. Extract interactive elements from pruned soup
    candidates = extract_candidates("", soup=pruned_soup)

    # 3. Task classification and hard-coded sequence check (pre-LLM bypass)
    task_type = classify_task(request.prompt)
    logger.info(
        "task_classified",
        extra={
            "task_id": request.task_id,
            "task_type": task_type.value,
            "step_index": request.step_index,
        },
    )
    if task_type == TaskType.LOGIN:
        login_fields = detect_login_fields(candidates)
        if login_fields is not None:
            action_dict = get_login_action(request.step_index, login_fields)
            if action_dict is not None:
                action = build_action(action_dict, candidates, request.url)
                if action is not None:
                    logger.info(
                        "hardcoded login action",
                        extra={
                            "task_id": request.task_id,
                            "step_index": request.step_index,
                            "action_type": type(action).__name__,
                        },
                    )
                    return ActResponse(actions=[action])
            # step_index >= 3 or action_dict is None: fall through to LLM

    if task_type == TaskType.LOGOUT:
        # Priority 1: logout button visible → click it
        logout_target = detect_logout_target(candidates)
        if logout_target is not None:
            action_dict = {"action": "click", "candidate_id": logout_target.button_id}
            action = build_action(action_dict, candidates, request.url)
            if action is not None:
                logger.info(
                    "hardcoded logout action",
                    extra={
                        "task_id": request.task_id,
                        "step_index": request.step_index,
                        "action_type": type(action).__name__,
                    },
                )
                return ActResponse(actions=[action])

        # Priority 2: login form visible → login first (LOGOUT tasks
        # often require "authenticate first, then log out")
        login_fields = detect_login_fields(candidates)
        if login_fields is not None:
            # Determine login sub-step by counting type actions in history
            type_count = sum(
                1 for h in request.history
                if h.get("action", "") == "type"  # exact match, not substring
            )
            # Login sequence: type(0), type(1), click(2).
            # After click-submit, type_count stays at 2 but a click exists.
            # If both conditions met, login is done — fall through to LLM.
            login_done = type_count >= 2 and any(
                h.get("action", "") == "click" for h in request.history
            )
            if not login_done:
                step = min(type_count, 2)
                action_dict = get_login_action(step, login_fields)
                if action_dict is not None:
                    action = build_action(action_dict, candidates, request.url)
                    if action is not None:
                        logger.info(
                            "hardcoded logout(login-first) action",
                            extra={
                                "task_id": request.task_id,
                                "step_index": request.step_index,
                                "action_type": type(action).__name__,
                            },
                        )
                        return ActResponse(actions=[action])

    if task_type == TaskType.REGISTRATION:
        reg_fields = detect_registration_fields(candidates)
        if reg_fields is not None:
            action_dict = get_registration_action(request.step_index, reg_fields)
            if action_dict is not None:
                action = build_action(action_dict, candidates, request.url)
                if action is not None:
                    logger.info(
                        "hardcoded registration action",
                        extra={
                            "task_id": request.task_id,
                            "step_index": request.step_index,
                            "action_type": type(action).__name__,
                        },
                    )
                    return ActResponse(actions=[action])

    if task_type == TaskType.CONTACT:
        contact_fields = detect_contact_fields(candidates)
        if contact_fields is not None:
            action_dict = get_contact_action(request.step_index, contact_fields)
            if action_dict is not None:
                action = build_action(action_dict, candidates, request.url)
                if action is not None:
                    logger.info(
                        "hardcoded contact action",
                        extra={
                            "task_id": request.task_id,
                            "step_index": request.step_index,
                            "action_type": type(action).__name__,
                        },
                    )
                    return ActResponse(actions=[action])

    if task_type == TaskType.ADD_FILM:
        url_path = urlsplit(request.url).path.rstrip("/")
        if url_path != "/add":
            # Navigate directly to /add/ -- saves steps vs clicking through UI
            nav_url = preserve_seed(
                normalize_url("/add/"), request.url
            )
            action = NavigateAction(type="NavigateAction", url=nav_url)
            logger.info(
                "hardcoded add_film navigate",
                extra={"task_id": request.task_id, "step_index": request.step_index},
            )
            return ActResponse(actions=[action])
        # On add form -- detect fields and fill
        add_fields = detect_add_film_fields(candidates)
        if add_fields is not None:
            film_values = extract_add_film_values_from_prompt(request.prompt)
            # step_index offset: step 0 was navigate, form filling starts at relative step 0
            form_step = request.step_index - 1 if request.step_index > 0 else 0
            action_dict = get_add_film_action(form_step, add_fields, film_values)
            if action_dict is not None:
                action = build_action(action_dict, candidates, request.url)
                if action is not None:
                    logger.info(
                        "hardcoded add_film action",
                        extra={
                            "task_id": request.task_id,
                            "step_index": request.step_index,
                            "action_type": type(action).__name__,
                        },
                    )
                    return ActResponse(actions=[action])
        # Fallback: let LLM handle with STRATEGY hint

    if task_type == TaskType.EDIT_FILM:
        url_path = urlsplit(request.url).path.rstrip("/")

        if url_path == "" or url_path == "/":
            # On homepage -- find the target film and click Details
            film_name = extract_film_name_from_prompt(request.prompt)
            if film_name:
                details_id = find_film_details_link(candidates, film_name)
                if details_id is not None:
                    action_dict = {"action": "click", "candidate_id": details_id}
                    action = build_action(action_dict, candidates, request.url)
                    if action is not None:
                        logger.info(
                            "hardcoded edit_film click_details",
                            extra={"task_id": request.task_id, "step_index": request.step_index},
                        )
                        return ActResponse(actions=[action])

        elif "/movie/" in url_path and "/edit" not in url_path:
            # On detail page -- click Edit button
            edit_id = find_button_by_label(candidates, ("edit", "modify", "update"))
            if edit_id is not None:
                action_dict = {"action": "click", "candidate_id": edit_id}
                action = build_action(action_dict, candidates, request.url)
                if action is not None:
                    logger.info(
                        "hardcoded edit_film click_edit",
                        extra={"task_id": request.task_id, "step_index": request.step_index},
                    )
                    return ActResponse(actions=[action])

        elif "/edit" in url_path:
            # On edit form -- detect fields and fill
            edit_fields = detect_edit_film_fields(candidates)
            if edit_fields is not None:
                edit_values = extract_edit_values_from_prompt(request.prompt)
                # Count type actions in history to determine form step
                type_count = sum(1 for h in request.history if h.get("action", "") == "type")
                form_step = type_count  # 0 = first type, 1+ = submit
                action_dict = get_edit_film_action(form_step, edit_fields, edit_values)
                if action_dict is not None:
                    action = build_action(action_dict, candidates, request.url)
                    if action is not None:
                        logger.info(
                            "hardcoded edit_film action",
                            extra={
                                "task_id": request.task_id,
                                "step_index": request.step_index,
                                "action_type": type(action).__name__,
                            },
                        )
                        return ActResponse(actions=[action])

        # Fallback: let LLM handle with STRATEGY hint

    if task_type == TaskType.DELETE_FILM:
        url_path = urlsplit(request.url).path.rstrip("/")

        if url_path == "" or url_path == "/":
            # On homepage -- find the target film and click Details
            film_name = extract_film_name_from_prompt(request.prompt)
            if film_name:
                details_id = find_film_details_link(candidates, film_name)
                if details_id is not None:
                    action_dict = {"action": "click", "candidate_id": details_id}
                    action = build_action(action_dict, candidates, request.url)
                    if action is not None:
                        logger.info(
                            "hardcoded delete_film click_details",
                            extra={"task_id": request.task_id, "step_index": request.step_index},
                        )
                        return ActResponse(actions=[action])

        elif "/movie/" in url_path:
            # On detail page -- try confirm button first (if dialog already showing),
            # then try delete button
            confirm_id = find_confirm_button(candidates)
            if confirm_id is not None:
                action_dict = {"action": "click", "candidate_id": confirm_id}
                action = build_action(action_dict, candidates, request.url)
                if action is not None:
                    logger.info(
                        "hardcoded delete_film confirm",
                        extra={"task_id": request.task_id, "step_index": request.step_index},
                    )
                    return ActResponse(actions=[action])

            delete_id = find_button_by_label(candidates, ("delete", "remove"))
            if delete_id is not None:
                action_dict = {"action": "click", "candidate_id": delete_id}
                action = build_action(action_dict, candidates, request.url)
                if action is not None:
                    logger.info(
                        "hardcoded delete_film click_delete",
                        extra={"task_id": request.task_id, "step_index": request.step_index},
                    )
                    return ActResponse(actions=[action])

        else:
            # On a non-movie page (e.g., confirmation redirect) -- check for confirm button
            confirm_id = find_confirm_button(candidates)
            if confirm_id is not None:
                action_dict = {"action": "click", "candidate_id": confirm_id}
                action = build_action(action_dict, candidates, request.url)
                if action is not None:
                    logger.info(
                        "hardcoded delete_film confirm",
                        extra={"task_id": request.task_id, "step_index": request.step_index},
                    )
                    return ActResponse(actions=[action])

        # Fallback: let LLM handle with STRATEGY hint

    if task_type == TaskType.ADD_TO_WATCHLIST:
        url_path = urlsplit(request.url).path.rstrip("/")

        if "/movie" in url_path:
            # On movie detail page -- click watchlist button
            watchlist_id = find_watchlist_button(candidates)
            if watchlist_id is not None:
                action_dict = {"action": "click", "candidate_id": watchlist_id}
                action = build_action(action_dict, candidates, request.url)
                if action is not None:
                    logger.info(
                        "hardcoded add_to_watchlist click_watchlist",
                        extra={
                            "task_id": request.task_id,
                            "step_index": request.step_index,
                        },
                    )
                    return ActResponse(actions=[action])
        else:
            # On homepage or other page -- find target film and click Details
            film_name = extract_film_name_from_prompt(request.prompt)
            if film_name:
                details_id = find_film_details_link(candidates, film_name)
                if details_id is not None:
                    action_dict = {"action": "click", "candidate_id": details_id}
                    action = build_action(action_dict, candidates, request.url)
                    if action is not None:
                        logger.info(
                            "hardcoded add_to_watchlist click_details",
                            extra={
                                "task_id": request.task_id,
                                "step_index": request.step_index,
                            },
                        )
                        return ActResponse(actions=[action])
        # Fallback: let LLM handle

    if task_type == TaskType.REMOVE_FROM_WATCHLIST:
        url_path = urlsplit(request.url).path.rstrip("/")

        if "/profile" not in url_path:
            # Navigate to profile page
            nav_url = preserve_seed(normalize_url("/profile/"), request.url)
            action = NavigateAction(type="NavigateAction", url=nav_url)
            logger.info(
                "hardcoded remove_from_watchlist navigate_profile",
                extra={
                    "task_id": request.task_id,
                    "step_index": request.step_index,
                },
            )
            return ActResponse(actions=[action])

        # On profile page -- check if we can see the target film's remove button
        film_name = extract_film_name_from_prompt(request.prompt)
        if film_name:
            remove_id = find_remove_button_for_film(candidates, film_name)
            if remove_id is not None:
                action_dict = {"action": "click", "candidate_id": remove_id}
                action = build_action(action_dict, candidates, request.url)
                if action is not None:
                    logger.info(
                        "hardcoded remove_from_watchlist click_remove",
                        extra={
                            "task_id": request.task_id,
                            "step_index": request.step_index,
                        },
                    )
                    return ActResponse(actions=[action])

        # Film not visible -- check if any remove buttons exist (indicating
        # we're already on the watchlist tab but the film isn't there).
        any_remove = any(
            "remove" in (c.label or c.text or "").lower()
            for c in candidates
            if c.tag in ("button", "a", "input")
        )
        if not any_remove:
            # No remove buttons visible -- likely on wrong tab. Click Watchlist tab.
            tab_id = find_tab_button(candidates, ("watchlist", "my list", "saved", "list"))
            if tab_id is not None:
                action_dict = {"action": "click", "candidate_id": tab_id}
                action = build_action(action_dict, candidates, request.url)
                if action is not None:
                    logger.info(
                        "hardcoded remove_from_watchlist click_tab",
                        extra={
                            "task_id": request.task_id,
                            "step_index": request.step_index,
                        },
                    )
                    return ActResponse(actions=[action])
        # Fallback: let LLM handle

    if task_type == TaskType.EDIT_USER:
        url_path = urlsplit(request.url).path.rstrip("/")

        # Priority 1: Login form visible -> login first
        login_fields = detect_login_fields(candidates)
        if login_fields is not None:
            creds = extract_credentials_from_prompt(request.prompt)
            if creds:
                username, password = creds
                type_count = sum(
                    1 for h in request.history
                    if h.get("action", "") == "type"
                )
                login_done = type_count >= 2 and any(
                    h.get("action", "") == "click" for h in request.history
                )
                if not login_done:
                    step = min(type_count, 2)
                    if step == 0:
                        action_dict = {
                            "action": "type",
                            "candidate_id": login_fields.username_id,
                            "text": username,
                        }
                    elif step == 1:
                        action_dict = {
                            "action": "type",
                            "candidate_id": login_fields.password_id,
                            "text": password,
                        }
                    else:
                        action_dict = {
                            "action": "click",
                            "candidate_id": login_fields.submit_id,
                        }
                    action = build_action(action_dict, candidates, request.url)
                    if action is not None:
                        logger.info(
                            "hardcoded edit_user login-first",
                            extra={
                                "task_id": request.task_id,
                                "step_index": request.step_index,
                                "action_type": type(action).__name__,
                            },
                        )
                        return ActResponse(actions=[action])

        # Priority 2: Not on profile page -> navigate to /profile
        elif "/profile" not in url_path:
            nav_url = preserve_seed(normalize_url("/profile/"), request.url)
            action = NavigateAction(type="NavigateAction", url=nav_url)
            logger.info(
                "hardcoded edit_user navigate_profile",
                extra={
                    "task_id": request.task_id,
                    "step_index": request.step_index,
                },
            )
            return ActResponse(actions=[action])

        # Priority 3: On profile page -> detect fields and fill
        else:
            profile_fields = detect_profile_fields(candidates)
            if profile_fields is not None:
                profile_values = extract_profile_values_from_prompt(request.prompt)
                type_count = sum(
                    1 for h in request.history
                    if h.get("action", "") == "type"
                )
                # Subtract login types (2) from total to get profile form step
                profile_type_count = max(0, type_count - 2)
                action_dict = get_profile_action(
                    profile_type_count, profile_fields, profile_values
                )
                if action_dict is not None:
                    action = build_action(action_dict, candidates, request.url)
                    if action is not None:
                        logger.info(
                            "hardcoded edit_user action",
                            extra={
                                "task_id": request.task_id,
                                "step_index": request.step_index,
                                "action_type": type(action).__name__,
                            },
                        )
                        return ActResponse(actions=[action])
        # Fallback: let LLM handle

    if task_type == TaskType.FILM_DETAIL:
        url_path = urlsplit(request.url).path.rstrip("/")

        # Already on a movie detail page -> done
        if "/movie" in url_path:
            return ActResponse(actions=[])

        # On homepage or other page -> find target film and click Details
        film_name = extract_film_name_from_prompt(request.prompt)
        if film_name:
            details_id = find_film_details_link(candidates, film_name)
            if details_id is not None:
                action_dict = {"action": "click", "candidate_id": details_id}
                action = build_action(action_dict, candidates, request.url)
                if action is not None:
                    logger.info(
                        "hardcoded film_detail click_details",
                        extra={"task_id": request.task_id, "step_index": request.step_index},
                    )
                    return ActResponse(actions=[action])
        # Fallback: let LLM handle (criteria-based prompts without specific film name)

    if task_type == TaskType.FILTER_FILM:
        url_path = urlsplit(request.url).path.rstrip("/")

        if "/search" not in url_path:
            # Navigate to search page
            nav_url = preserve_seed(normalize_url("/search"), request.url)
            action = NavigateAction(type="NavigateAction", url=nav_url)
            logger.info(
                "hardcoded filter_film navigate_search",
                extra={"task_id": request.task_id, "step_index": request.step_index},
            )
            return ActResponse(actions=[action])

        # On search page -> extract criteria and select dropdowns
        criteria = extract_filter_criteria_from_prompt(request.prompt)
        if criteria:
            dropdowns = detect_filter_dropdowns(candidates)
            if dropdowns is not None:
                # Count select actions in history to determine which dropdown to fill next
                select_count = sum(
                    1 for h in request.history
                    if h.get("action", h.get("type", "")) in ("select", "SelectDropDownOptionAction")
                )

                # Build ordered steps: genre first (if present), then year
                steps: list[dict] = []
                if "genre" in criteria and dropdowns.genre_id is not None:
                    steps.append({
                        "action": "select",
                        "candidate_id": dropdowns.genre_id,
                        "text": criteria["genre"],
                    })
                if "year" in criteria and dropdowns.year_id is not None:
                    steps.append({
                        "action": "select",
                        "candidate_id": dropdowns.year_id,
                        "text": criteria["year"],
                    })

                if select_count < len(steps):
                    action_dict = steps[select_count]
                    action = build_action(action_dict, candidates, request.url)
                    if action is not None:
                        logger.info(
                            "hardcoded filter_film select",
                            extra={
                                "task_id": request.task_id,
                                "step_index": request.step_index,
                                "action_type": type(action).__name__,
                            },
                        )
                        return ActResponse(actions=[action])
        # Fallback: let LLM handle

    if task_type == TaskType.SEARCH_FILM:
        url_path = urlsplit(request.url).path.rstrip("/")

        if "/search" not in url_path:
            # Step 1: Navigate to search page
            nav_url = preserve_seed(normalize_url("/search"), request.url)
            action = NavigateAction(type="NavigateAction", url=nav_url)
            logger.info(
                "hardcoded search_film navigate_search",
                extra={"task_id": request.task_id, "step_index": request.step_index},
            )
            return ActResponse(actions=[action])

        # Step 2+: On search page -- find text input, type search term, submit
        film_name = extract_film_name_from_prompt(request.prompt)
        if film_name:
            search_fields = detect_search_input(candidates)
            if search_fields is not None:
                # Count type actions in history to determine search sequence step
                type_count = sum(
                    1 for h in request.history
                    if h.get("action", "") == "type"
                )
                action_dict = get_search_action(type_count, search_fields, film_name)
                if action_dict is not None:
                    action = build_action(action_dict, candidates, request.url)
                    if action is not None:
                        logger.info(
                            "hardcoded search_film action",
                            extra={
                                "task_id": request.task_id,
                                "step_index": request.step_index,
                                "action_type": type(action).__name__,
                            },
                        )
                        return ActResponse(actions=[action])
        # Fallback: let LLM handle

    # 4. Build compact Page IR
    page_ir = build_page_ir(pruned_soup, request.url, title, candidates)

    # 5. Compute steps remaining
    steps_remaining = max(1, 12 - request.step_index)

    # 6. Build history lines
    history_lines = _build_history_lines(request.history)

    # 7. Loop detection -- use last action sig from history
    loop_hint: str | None = None
    if request.history:
        last_entry = request.history[-1]
        last_sig = get_action_signature(last_entry)
        loop_hint = check_loop(request.task_id, request.url, last_sig)

    # 8. Build LLM messages
    system_msg = build_system_prompt(task_type)
    user_msg = build_user_prompt(
        task_prompt=request.prompt,
        page_ir=page_ir,
        history_lines=history_lines,
        steps_remaining=steps_remaining,
        loop_hint=loop_hint,
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # 9. Get LLM client
    client = _get_llm_client()

    # Retry state -- only retry on JSON parse failure
    max_retries = 1  # At most 1 retry (2 LLM calls total)

    for attempt in range(max_retries + 1):
        try:
            # 9. Call LLM
            resp = client.chat_completions(
                task_id=request.task_id,
                messages=messages,
            )

            # 10. Log cost from usage object
            usage = resp.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            estimated_cost = (prompt_tokens * 0.05 + completion_tokens * 0.40) / 1_000_000
            logger.info(
                "llm_call",
                extra={
                    "task_id": request.task_id,
                    "step_index": request.step_index,
                    "attempt": attempt + 1,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "estimated_cost_usd": round(estimated_cost, 6),
                },
            )

            # 11. Parse response
            content = resp["choices"][0]["message"]["content"]
            decision = parse_llm_json(content)

            # 12. Build action (validate_and_fix is called inside build_action)
            action = build_action(
                decision, candidates, request.url,
                step_index=request.step_index,
            )

            # 13. Handle "done" signal
            if action is None:
                logger.info(
                    "agent decided: done",
                    extra={"task_id": request.task_id, "action_type": "done"},
                )
                clear_task_state(request.task_id)
                return ActResponse(actions=[])

            # 14. Valid action obtained (build_action returns ScrollAction
            # fallback for invalid decisions, never None except for "done")
            action_type = type(action).__name__
            logger.info(
                "agent decided",
                extra={
                    "task_id": request.task_id,
                    "action_type": action_type,
                },
            )
            return ActResponse(actions=[action])

        except ValueError:
            # 15. Invalid JSON -- retry with stronger instruction
            if attempt < max_retries:
                retry_instruction = (
                    "Your previous response was not valid JSON. "
                    "You MUST respond with a JSON object only. "
                    "No markdown, no commentary, no code fences."
                )
                messages.append({"role": "user", "content": retry_instruction})
                logger.info(
                    "invalid JSON from LLM, retrying",
                    extra={"task_id": request.task_id},
                )
                continue
            # Fall through to fallback below
            break

    # 16. All retries exhausted -- return WaitAction fallback
    logger.warning(
        "all retries exhausted, returning fallback",
        extra={"task_id": request.task_id},
    )
    return ActResponse(actions=[WaitAction(type="WaitAction", time_seconds=1.0)])
