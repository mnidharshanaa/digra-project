"""
Prompt templates used only for building the correct/incorrect response
pools (Appendix B-3). Debate-round prompts (Module 2) live separately in
src/agents/prompts.py since they have a different shape (they include
other agents' prior responses, not few-shot examples).

Kept as pure string-building functions — no model calls here — so they're
trivially unit-testable and the wording can be iterated on without
touching orchestration code.
"""

from __future__ import annotations

FINAL_ANSWER_INSTRUCTION = (
    "End your response with a line in the exact form 'Final answer: <your answer>'."
)


def build_neutral_prompt(question: str) -> str:
    """Round-1-style prompt with no seeding — used for the initial 50-sample
    draw in Appendix B-3 step 1, and for genuine 'Standard Debate' round-1
    generation (Module 2)."""
    return (
        f"Answer the following question concisely.\n\n"
        f"Question: {question}\n\n"
        f"{FINAL_ANSWER_INSTRUCTION}"
    )


def build_guided_correct_prompt(
    question: str,
    gold_answer: str,
    correct_examples: list,
) -> str:
    """
    Appendix B-3 step 3: the model is given the gold answer directly plus
    n1-shot examples drawn from its own already-correct round-1 samples,
    and asked to produce another correct response in a similar style.
    """
    if not correct_examples:
        examples_block = "(no prior correct examples available)"
    else:
        examples_block = "\n\n".join(
            f"Example correct response {i + 1}:\n{ex}"
            for i, ex in enumerate(correct_examples)
        )

    return (
        f"Question: {question}\n"
        f"The correct answer is: {gold_answer}\n\n"
        f"Here are example correct responses to this question:\n\n{examples_block}\n\n"
        f"Write another response to the question above that reaches the correct "
        f"answer ({gold_answer}), in a similar style to the examples above.\n\n"
        f"{FINAL_ANSWER_INSTRUCTION}"
    )
