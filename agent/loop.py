"""Main agent decision loop called from the /act endpoint.

Orchestrates HTML pruning, Page IR construction, prompt building, LLM calls,
response validation, and action building into a single ``decide()`` function.
"""

from __future__ import annotations

import logging

from agent.actions import build_action, validate_and_fix
from agent.classifier import (
    TaskType,
    classify_task,
    detect_contact_fields,
    detect_login_fields,
    detect_logout_target,
    detect_registration_fields,
    get_contact_action,
    get_login_action,
    get_logout_action,
    get_registration_action,
)
from agent.prompts import (
    build_system_prompt,
    build_user_prompt,
    format_history_entry,
)
from agent.state import check_loop, clear_task_state, get_action_signature
from llm.client import LLMClient
from llm.parser import parse_llm_json
from models.actions import WaitAction
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
    system_msg = build_system_prompt()
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
            estimated_cost = (prompt_tokens * 1.75 + completion_tokens * 14.00) / 1_000_000
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
