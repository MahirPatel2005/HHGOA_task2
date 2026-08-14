from app.guardrails import GuardrailEngine


def test_answer_guardrail_accepts_common_bracket_citation_style():
    decision = GuardrailEngine().check_answer(
        "The capital is Panaji【S1】.",
        ["Panaji is the capital city of Goa."],
    )

    assert decision.allowed is True
