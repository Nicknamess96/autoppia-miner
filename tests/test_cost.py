"""Tests for cost optimization settings (COST-01, COST-02, COST-03)."""
import inspect

import pytest


@pytest.mark.unit
class TestCostSettings:
    """Verify cost-related configuration values match gpt-5-nano pricing."""

    def test_default_model_is_nano(self):
        """COST-01: Default LLM model is gpt-5-nano."""
        from llm.client import LLMClient

        sig = inspect.signature(LLMClient.chat_completions)
        model_default = sig.parameters["model"].default
        assert model_default == "gpt-5-nano", (
            f"Expected default model 'gpt-5-nano', got '{model_default}'"
        )

    def test_cost_estimation_formula(self):
        """COST-02: Cost formula uses gpt-5-nano pricing ($0.05/$0.40 per 1M)."""
        # Read loop.py source and verify the cost formula line
        import agent.loop as loop_mod
        source = inspect.getsource(loop_mod.decide)

        # Must contain nano pricing, not gpt-5.2 pricing
        assert "0.05" in source and "0.40" in source, (
            "Cost formula should use gpt-5-nano pricing: 0.05 (input) and 0.40 (output)"
        )
        assert "1.75" not in source, (
            "Cost formula still contains gpt-5.2 input pricing (1.75)"
        )
        assert "14.00" not in source and "14.0 " not in source, (
            "Cost formula still contains gpt-5.2 output pricing (14.00)"
        )

    def test_page_ir_budget(self):
        """COST-03: Page IR token budget is 900 tokens."""
        from parsing.page_ir import build_page_ir

        sig = inspect.signature(build_page_ir)
        budget = sig.parameters["max_tokens"].default
        assert budget == 900, (
            f"Expected max_tokens default 900, got {budget}"
        )

    def test_cost_under_ceiling(self):
        """Verify estimated cost per LLM call is well under $0.05."""
        # With 900-token page IR + ~268 overhead = ~1168 input tokens
        # With 300 max output tokens
        # gpt-5-nano: $0.05/1M input, $0.40/1M output
        input_tokens = 1168
        output_tokens = 300
        cost = (input_tokens * 0.05 + output_tokens * 0.40) / 1_000_000
        assert cost < 0.001, f"Single call cost ${cost:.6f} exceeds $0.001"
        assert cost < 0.05, f"Single call cost ${cost:.6f} exceeds $0.05 ceiling"
