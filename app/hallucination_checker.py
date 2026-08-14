"""
Core hallucination-checking pipeline built with LangChain + Groq.

Pipeline stages
---------------
1. Question Generation  — LLM generates N factual questions about a topic.
2. Answer Generation    — LLM answers each question (run concurrently).
3. Fact-Checking        — A second LLM call scores each answer for accuracy.

Retry logic (tenacity) is applied to every LLM call with exponential back-off.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import List

from langchain_openai import ChatOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.models import (
    HallucinationResponse,
    QuestionAnswer,
    VerificationResult,
    Verdict,
)
from app.prompts import (
    ANSWER_GENERATION_PROMPT,
    FACT_CHECK_PROMPT,
    QUESTION_GENERATION_PROMPT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> str:
    """
    Strip markdown code fences if present, then return the raw JSON string.

    Args:
        text: Raw LLM response text that may contain markdown fences.

    Returns:
        A plain JSON string suitable for ``json.loads``.
    """
    match = _JSON_BLOCK_RE.search(text)
    return match.group(1).strip() if match else text.strip()


async def _invoke_with_retry(chain, inputs: dict, *, max_attempts: int = 3) -> str:
    """
    Invoke a LangChain runnable with exponential-backoff retry.

    Args:
        chain:        Any LangChain ``Runnable`` (prompt | llm).
        inputs:       Dictionary of template variables.
        max_attempts: Maximum number of attempts before raising.

    Returns:
        The string content of the LLM response.

    Raises:
        Exception: Re-raises the last exception after exhausting retries.
    """
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    ):
        with attempt:
            response = await chain.ainvoke(inputs)
            # LangChain returns an AIMessage; extract string content.
            return response.content if hasattr(response, "content") else str(response)


# ---------------------------------------------------------------------------
# Pipeline Class
# ---------------------------------------------------------------------------


class HallucinationChecker:
    """
    Orchestrates the three-stage hallucination-checking pipeline.

    Args:
        model_name: OpenAI model identifier (e.g. ``gpt-4o-mini``).
        api_key:    OpenAI API key. If *None* the SDK reads ``OPENAI_API_KEY``
                    from the environment automatically.
    """

    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        self.model_name = model_name
        kwargs: dict = {"model": model_name, "max_tokens": 2048, "temperature": 0}
        # TODO: maybe add a way to customize the temperature if people want more creative questions?
        if api_key:
            kwargs["api_key"] = api_key

        self._llm = ChatOpenAI(**kwargs)

        # Pre-build LCEL chains (prompt | llm)
        # We tie the prompt and llm together here. Nice and simple.
        self._question_chain = QUESTION_GENERATION_PROMPT | self._llm
        self._answer_chain = ANSWER_GENERATION_PROMPT | self._llm
        self._fact_check_chain = FACT_CHECK_PROMPT | self._llm

    # ------------------------------------------------------------------
    # Stage 1: Question Generation
    # ------------------------------------------------------------------

    async def _generate_questions(self, topic: str, num_questions: int) -> List[str]:
        """
        Ask LLM to generate ``num_questions`` factual questions about ``topic``.

        Args:
            topic:         The subject to generate questions about.
            num_questions: Desired number of questions.

        Returns:
            List of question strings.

        Raises:
            ValueError: If the LLM response cannot be parsed as expected JSON.
        """
        logger.info("Stage 1 — generating %d questions for topic: %r", num_questions, topic)

        raw = await _invoke_with_retry(
            self._question_chain,
            {"topic": topic, "num_questions": num_questions},
        )
        data = json.loads(_extract_json(raw))
        questions: List[str] = data.get("questions", [])

        if not questions:
            raise ValueError("LLM returned an empty question list.")

        # Trim or extend gracefully to exactly num_questions.
        questions = questions[:num_questions]
        logger.info("Stage 1 — received %d questions.", len(questions))
        return questions

    # ------------------------------------------------------------------
    # Stage 2: Answer Generation (concurrent)
    # ------------------------------------------------------------------

    async def _answer_question(self, question: str) -> QuestionAnswer:
        """
        Ask LLM to answer a single factual question.

        Args:
            question: The factual question to answer.

        Returns:
            A :class:`QuestionAnswer` pairing the question with the answer.
        """
        logger.debug("Stage 2 — answering: %r", question)
        answer = await _invoke_with_retry(
            self._answer_chain, {"question": question}
        )
        return QuestionAnswer(question=question, answer=answer.strip())

    async def _generate_answers(self, questions: List[str]) -> List[QuestionAnswer]:
        """
        Answer all questions concurrently using ``asyncio.gather``.

        Args:
            questions: List of factual questions.

        Returns:
            List of :class:`QuestionAnswer` objects in the same order.
        """
        logger.info("Stage 2 — answering %d questions concurrently.", len(questions))
        tasks = [self._answer_question(q) for q in questions]
        results: List[QuestionAnswer] = await asyncio.gather(*tasks)
        logger.info("Stage 2 — all answers collected.")
        return results

    # ------------------------------------------------------------------
    # Stage 3: Fact-Checking / Hallucination Scoring
    # ------------------------------------------------------------------

    async def _fact_check(self, qa: QuestionAnswer) -> VerificationResult:
        """
        Fact-check a single question-answer pair.
        """
        logger.debug("Stage 3 — fact-checking: %r", qa.question)

        raw = await _invoke_with_retry(
            self._fact_check_chain,
            {"question": qa.question, "answer": qa.answer},
        )
        data = json.loads(_extract_json(raw))

        # Sometimes the model replies with lowercase or weird casings. Let's normalize it.
        verdict_str = data.get("verdict", "UNCERTAIN").strip().upper()
        try:
            verdict = Verdict(verdict_str)
        except ValueError:
            # Fallback to uncertain if the LLM hallucinated a verdict
            verdict = Verdict.UNCERTAIN

        return VerificationResult(
            question=qa.question,
            answer=qa.answer,
            verdict=verdict,
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", "No reasoning provided."),
        )

    async def _fact_check_all(
        self, qa_pairs: List[QuestionAnswer]
    ) -> List[VerificationResult]:
        """
        Fact-check all QA pairs concurrently.

        Args:
            qa_pairs: List of question-answer pairs to verify.

        Returns:
            List of :class:`VerificationResult` objects.
        """
        logger.info("Stage 3 — fact-checking %d answers concurrently.", len(qa_pairs))
        tasks = [self._fact_check(qa) for qa in qa_pairs]
        results: List[VerificationResult] = await asyncio.gather(*tasks)
        logger.info("Stage 3 — fact-checking complete.")
        return results

    # ------------------------------------------------------------------
    # Public Entry-Point
    # ------------------------------------------------------------------

    async def run(self, topic: str, num_questions: int) -> HallucinationResponse:
        """
        Execute the full three-stage hallucination-checking pipeline.

        Args:
            topic:         The subject to probe.
            num_questions: How many questions to generate and test.

        Returns:
            A fully populated :class:`HallucinationResponse`.
        """
        logger.info(
            "Pipeline start — topic=%r  num_questions=%d  model=%s",
            topic,
            num_questions,
            self.model_name,
        )

        # Stage 1
        questions = await self._generate_questions(topic, num_questions)

        # Stage 2
        qa_pairs = await self._generate_answers(questions)

        # Stage 3
        verification_results = await self._fact_check_all(qa_pairs)

        # Aggregate metrics
        total = len(verification_results)
        hallucinated_count = sum(
            1 for r in verification_results if r.verdict == Verdict.HALLUCINATED
        )
        hallucination_rate = hallucinated_count / total if total else 0.0

        # Build human-readable summary
        accurate_count = sum(
            1 for r in verification_results if r.verdict == Verdict.ACCURATE
        )
        uncertain_count = sum(
            1 for r in verification_results if r.verdict == Verdict.UNCERTAIN
        )
        summary = (
            f"Tested {total} questions about '{topic}'. "
            f"Results: {accurate_count} accurate, "
            f"{hallucinated_count} hallucinated, "
            f"{uncertain_count} uncertain. "
            f"Hallucination rate: {hallucination_rate:.0%}."
        )

        logger.info("Pipeline complete — hallucination_rate=%.2f", hallucination_rate)

        return HallucinationResponse(
            topic=topic,
            questions_tested=total,
            hallucination_rate=round(hallucination_rate, 4),
            results=verification_results,
            summary=summary,
        )
