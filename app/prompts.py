"""
LangChain ChatPromptTemplate definitions for the Hallucination Tester pipeline.

All three pipeline stages — question generation, answer generation, and
fact-checking — are defined here so that they can be imported and composed
cleanly in hallucination_checker.py.
"""

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Stage 1 — Question Generation
# ---------------------------------------------------------------------------

QUESTION_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a meticulous researcher who designs rigorous factual "
                "questions. Your task is to generate exactly {num_questions} "
                "distinct, specific, and verifiable factual questions about the "
                "given topic. Each question must:\n"
                "- Be answerable with publicly known facts\n"
                "- Test a single, concrete piece of information\n"
                "- Avoid vague or opinion-based phrasing\n\n"
                "Respond with a JSON object in this exact format (no markdown fences):\n"
                '{{"questions": ["question 1", "question 2", ...]}}'
            ),
        ),
        (
            "human",
            "Generate {num_questions} factual questions about the following topic:\n\n"
            "Topic: {topic}",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Stage 2 — Answer Generation
# ---------------------------------------------------------------------------

ANSWER_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a knowledgeable assistant. Answer the following factual "
                "question as accurately and concisely as possible. "
                "Provide only the direct answer — no preamble, no hedging phrases "
                "like 'As an AI…', and no markdown formatting. "
                "If you are genuinely uncertain about a specific detail, state that "
                "clearly rather than guessing."
            ),
        ),
        (
            "human",
            "Question: {question}",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Stage 3 — Fact-Checking / Hallucination Scoring
# ---------------------------------------------------------------------------

FACT_CHECK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an expert fact-checker with access to broad world knowledge "
                "up to your training cutoff. Your job is to evaluate whether a given "
                "answer to a factual question is accurate.\n\n"
                "Verdict definitions:\n"
                "  ACCURATE      — The answer is factually correct and well-supported.\n"
                "  HALLUCINATED  — The answer contains one or more clear factual errors "
                "or fabrications.\n"
                "  UNCERTAIN     — You cannot confidently verify or refute the answer "
                "given your knowledge limitations.\n\n"
                "Respond with a JSON object in this exact format (no markdown fences):\n"
                '{{\n'
                '  "verdict": "ACCURATE | HALLUCINATED | UNCERTAIN",\n'
                '  "confidence": <float 0.0-1.0>,\n'
                '  "reasoning": "<one or two sentences explaining your verdict>"\n'
                "}}"
            ),
        ),
        (
            "human",
            "Question: {question}\n\nAnswer to evaluate: {answer}",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Reference-based Prompts (Stage 1 & Stage 3 fallback)
# ---------------------------------------------------------------------------

QUESTION_GENERATION_REF_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a meticulous researcher who designs rigorous factual "
                "questions. Your task is to generate exactly {num_questions} "
                "distinct, specific, and verifiable factual questions about the "
                "given topic, based ONLY on the provided reference text. Each question must:\n"
                "- Be answerable directly using the provided reference text\n"
                "- Test a concrete piece of information from the reference\n"
                "- Avoid vague or opinion-based phrasing\n\n"
                "Respond with a JSON object in this exact format (no markdown fences):\n"
                '{{"questions": ["question 1", "question 2", ...]}}'
            ),
        ),
        (
            "human",
            "Generate {num_questions} factual questions about the following topic based on the reference text:\n\n"
            "Topic: {topic}\n\n"
            "Reference Text:\n{reference}",
        ),
    ]
)

FACT_CHECK_REF_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an expert fact-checker. Your job is to evaluate whether a given "
                "answer to a factual question is accurate, based ONLY on the provided reference text.\n\n"
                "Verdict definitions:\n"
                "  ACCURATE      — The answer is factually correct and fully supported by the reference text.\n"
                "  HALLUCINATED  — The answer contradicts the reference text, or contains claims not supported by the reference text.\n"
                "  UNCERTAIN     — The reference text does not contain enough information to verify or refute the answer.\n\n"
                "Respond with a JSON object in this exact format (no markdown fences):\n"
                '{{\n'
                '  "verdict": "ACCURATE | HALLUCINATED | UNCERTAIN",\n'
                '  "confidence": <float 0.0-1.0>,\n'
                '  "reasoning": "<one or two sentences explaining your verdict based on the reference text>"\n'
                "}}"
            ),
        ),
        (
            "human",
            "Question: {question}\n\nAnswer to evaluate: {answer}\n\nReference Text:\n{reference}",
        ),
    ]
)
