"""
Core hallucination-checking pipeline built with LangChain + OpenAI.

Pipeline stages
---------------
1. Question Generation  — LLM generates N factual questions about a topic.
2. Answer Generation    — LLM answers each question (run concurrently).
3. Fact-Checking        — A second LLM call scores each answer for accuracy.

Retry logic (tenacity) is applied to every LLM call with exponential back-off,
filtering out permanent errors (like auth and bad requests).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from langchain_openai import ChatOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.models import (
    HallucinationResponse,
    QuestionAnswer,
    VerificationResult,
    Verdict,
    QuestionGenerationResult,
    VerificationAssessment,
    EvaluationSummary,
)
from app.prompts import (
    ANSWER_GENERATION_PROMPT,
    FACT_CHECK_PROMPT,
    QUESTION_GENERATION_PROMPT,
    QUESTION_GENERATION_REF_PROMPT,
    FACT_CHECK_REF_PROMPT,
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


def _is_transient_error(exception: Exception) -> bool:
    """
    Determine if an exception is transient (should be retried) or permanent (fail fast).

    Transient errors:
    - Rate limits (429)
    - Server errors (5xx)
    - Connection/Network timeouts (httpx.TimeoutException, httpx.NetworkError)

    Permanent errors:
    - Invalid request (400)
    - Authentication failure (401)
    - Permission/Access denied (403)
    - Resource not found (404)
    """
    import httpx
    if isinstance(exception, httpx.HTTPError):
        if isinstance(exception, (httpx.TimeoutException, httpx.NetworkError)):
            return True
        if hasattr(exception, "response") and exception.response is not None:
            status = exception.response.status_code
            if status in (429, 500, 502, 503, 504):
                return True
            return False

    # Check OpenAI SDK specific errors
    try:
        from openai import AuthenticationError, BadRequestError, PermissionDeniedError, NotFoundError, OpenAIError
        if isinstance(exception, (AuthenticationError, BadRequestError, PermissionDeniedError, NotFoundError)):
            return False
        if isinstance(exception, OpenAIError):
            if hasattr(exception, "status_code"):
                status = getattr(exception, "status_code")
                if status in (400, 401, 403, 404):
                    return False
            return True
    except ImportError:
        pass

    return True


def _log_retry(retry_state):
    """Log retry attempts for observability."""
    logger.warning(
        "Retry event: Attempt %d failed. Retrying... Exception: %s",
        retry_state.attempt_number,
        retry_state.outcome.exception(),
    )


async def _invoke_with_retry(chain, inputs: dict, *, max_attempts: int | None = None):
    """
    Invoke a LangChain runnable with exponential-backoff retry.
    Supports both standard chains returning AIMessage and structured output runnables.
    """
    if max_attempts is None:
        max_attempts = int(os.getenv("MAX_RETRIES", "3"))

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_transient_error),
        after=_log_retry,
        reraise=True,
    ):
        with attempt:
            response = await chain.ainvoke(inputs)
            if asyncio.iscoroutine(response) or hasattr(response, "__await__"):
                response = await response
            if hasattr(response, "content"):
                return response.content
            return response


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

    def __init__(self, model_name: str, api_key: str | None = None, max_concurrency: int | None = None) -> None:
        self.model_name = model_name
        self.timeout = float(os.getenv("LLM_TIMEOUT", "30.0"))

        kwargs: dict = {
            "model": model_name,
            "max_tokens": 2048,
            "temperature": 0,
            "timeout": self.timeout,
        }
        if api_key:
            kwargs["api_key"] = api_key

        self._llm = ChatOpenAI(**kwargs)

        # Set up semaphore for concurrency limiting
        if max_concurrency is None:
            max_concurrency = int(os.getenv("MAX_CONCURRENCY", "3"))
        self._semaphore = asyncio.Semaphore(max_concurrency)

        # Pre-build standard LCEL chains (prompt | llm)
        self._question_chain = QUESTION_GENERATION_PROMPT | self._llm
        self._answer_chain = ANSWER_GENERATION_PROMPT | self._llm
        self._fact_check_chain = FACT_CHECK_PROMPT | self._llm

        # Pre-build reference-based LCEL chains
        self._question_ref_chain = QUESTION_GENERATION_REF_PROMPT | self._llm
        self._fact_check_ref_chain = FACT_CHECK_REF_PROMPT | self._llm

        # Try to build structured output versions if supported
        try:
            self._structured_question_llm = self._llm.with_structured_output(QuestionGenerationResult)
            self._structured_question_chain = QUESTION_GENERATION_PROMPT | self._structured_question_llm
            self._structured_question_ref_chain = QUESTION_GENERATION_REF_PROMPT | self._structured_question_llm
        except Exception as e:
            logger.warning("Structured output not supported for question generation: %s", e)
            self._structured_question_chain = None
            self._structured_question_ref_chain = None

        try:
            self._structured_fact_check_llm = self._llm.with_structured_output(VerificationAssessment)
            self._structured_fact_check_chain = FACT_CHECK_PROMPT | self._structured_fact_check_llm
            self._structured_fact_check_ref_chain = FACT_CHECK_REF_PROMPT | self._structured_fact_check_llm
        except Exception as e:
            logger.warning("Structured output not supported for fact checking: %s", e)
            self._structured_fact_check_chain = None
            self._structured_fact_check_ref_chain = None

    # ------------------------------------------------------------------
    # Stage 1: Question Generation
    # ------------------------------------------------------------------

    async def _generate_questions(self, topic: str, num_questions: int, reference: Optional[str] = None) -> List[str]:
        """
        Ask LLM to generate ``num_questions`` factual questions about ``topic``.

        Args:
            topic:         The subject to generate questions about.
            num_questions: Desired number of questions.
            reference:     Optional context text to generate questions from.

        Returns:
            List of question strings.

        Raises:
            ValueError: If the LLM response cannot be parsed.
        """
        logger.info(
            "Stage 1 — generating %d questions for topic: %r (reference mode: %s)",
            num_questions,
            topic,
            "Yes" if reference else "No",
        )

        inputs = {"topic": topic, "num_questions": num_questions}
        if reference:
            inputs["reference"] = reference
            structured_chain = self._structured_question_ref_chain
            fallback_chain = self._question_ref_chain
        else:
            structured_chain = self._structured_question_chain
            fallback_chain = self._question_chain

        questions: List[str] = []

        # Try structured output first
        if structured_chain is not None:
            try:
                res = await _invoke_with_retry(structured_chain, inputs)
                if isinstance(res, QuestionGenerationResult):
                    questions = res.questions
                elif isinstance(res, dict):
                    questions = res.get("questions", [])
            except Exception as e:
                logger.warning("Structured question generation failed, falling back to manual parsing: %s", e)

        # Fallback to manual parsing
        if not questions:
            raw = await _invoke_with_retry(fallback_chain, inputs)
            try:
                data = json.loads(_extract_json(raw))
                questions = data.get("questions", [])
            except Exception as e:
                raise ValueError(f"Failed to parse question generation output: {e}") from e

        if not questions:
            raise ValueError("LLM returned an empty question list.")

        # Trim or extend gracefully to exactly num_questions.
        questions = questions[:num_questions]
        logger.info("Stage 1 — received %d questions.", len(questions))
        return questions

    # ------------------------------------------------------------------
    # Stage 2: Answer Generation (concurrent with Semaphore & Timeout)
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
        try:
            async with self._semaphore:
                answer = await _invoke_with_retry(
                    self._answer_chain, {"question": question}
                )
                return QuestionAnswer(question=question, answer=answer.strip())
        except Exception as exc:
            logger.error("Stage 2 — failed to answer question %r: %s", question, exc)
            # Partial failure handling
            return QuestionAnswer(question=question, answer=f"[Error: Answer generation failed: {exc}]")

    async def _generate_answers(self, questions: List[str]) -> List[QuestionAnswer]:
        """
        Answer all questions concurrently using ``asyncio.gather``.

        Args:
            questions: List of factual questions.

        Returns:
            List of :class:`QuestionAnswer` objects in the same order.
        """
        logger.info("Stage 2 — answering %d questions concurrently with max_concurrency limit.", len(questions))
        tasks = [self._answer_question(q) for q in questions]
        results: List[QuestionAnswer] = await asyncio.gather(*tasks)
        logger.info("Stage 2 — all answers collected.")
        return results

    # ------------------------------------------------------------------
    # Stage 3: Fact-Checking / Hallucination Scoring (concurrent with Semaphore)
    # ------------------------------------------------------------------

    async def _fact_check(self, qa: QuestionAnswer, reference: Optional[str] = None) -> VerificationResult:
        """
        Fact-check a single question-answer pair.
        """
        logger.debug("Stage 3 — fact-checking: %r", qa.question)

        # Skip evaluation if answer generation failed in Stage 2
        if qa.answer.startswith("[Error:"):
            return VerificationResult(
                question=qa.question,
                answer=qa.answer,
                verdict=Verdict.UNCERTAIN,
                confidence=0.0,
                reasoning="Factual checking skipped because answer generation failed.",
                llm_judge_verdict=Verdict.UNCERTAIN,
            )

        inputs = {"question": qa.question, "answer": qa.answer}
        if reference:
            inputs["reference"] = reference
            structured_chain = self._structured_fact_check_ref_chain
            fallback_chain = self._fact_check_ref_chain
        else:
            structured_chain = self._structured_fact_check_chain
            fallback_chain = self._fact_check_chain

        assessment = None

        try:
            async with self._semaphore:
                # Try structured output first
                if structured_chain is not None:
                    try:
                        res = await _invoke_with_retry(structured_chain, inputs)
                        if isinstance(res, VerificationAssessment):
                            assessment = res
                        elif isinstance(res, dict):
                            assessment = VerificationAssessment(
                                verdict=res.get("verdict", "UNCERTAIN"),
                                confidence=float(res.get("confidence", 0.5)),
                                reasoning=res.get("reasoning", "No reasoning provided."),
                            )
                    except Exception as e:
                        logger.warning("Structured fact check failed, falling back to manual parsing: %s", e)

                # Fallback to manual parsing
                if assessment is None:
                    raw = await _invoke_with_retry(fallback_chain, inputs)
                    data = json.loads(_extract_json(raw))

                    verdict_str = str(data.get("verdict", "UNCERTAIN")).strip().upper()
                    try:
                        verdict = Verdict(verdict_str)
                    except ValueError:
                        verdict = Verdict.UNCERTAIN

                    raw_confidence = data.get("confidence", 0.5)
                    try:
                        confidence = float(raw_confidence)
                    except (ValueError, TypeError):
                        confidence = 0.5

                    # Clip confidence before instantiating VerificationAssessment to prevent validation errors
                    confidence = max(0.0, min(1.0, confidence))

                    assessment = VerificationAssessment(
                        verdict=verdict,
                        confidence=confidence,
                        reasoning=str(data.get("reasoning", "No reasoning provided.")),
                    )
        except Exception as exc:
            logger.error("Stage 3 — fact-checking failed for question %r: %s", qa.question, exc)
            return VerificationResult(
                question=qa.question,
                answer=qa.answer,
                verdict=Verdict.UNCERTAIN,
                confidence=0.0,
                reasoning=f"Fact-checking failed due to LLM error: {exc}",
                llm_judge_verdict=Verdict.UNCERTAIN,
            )

        # Normalize confidence score
        norm_confidence = max(0.0, min(1.0, assessment.confidence))

        # Normalize verdict and fall back to UNCERTAIN if invalid
        verdict_str = assessment.verdict.name if hasattr(assessment.verdict, "name") else str(assessment.verdict)
        verdict_str = verdict_str.strip().upper()
        try:
            final_verdict = Verdict(verdict_str)
        except ValueError:
            final_verdict = Verdict.UNCERTAIN

        return VerificationResult(
            question=qa.question,
            answer=qa.answer,
            verdict=final_verdict,
            confidence=norm_confidence,
            reasoning=assessment.reasoning,
            llm_judge_verdict=assessment.verdict,
        )

    async def _fact_check_all(
        self, qa_pairs: List[QuestionAnswer], reference: Optional[str] = None
    ) -> List[VerificationResult]:
        """
        Fact-check all QA pairs concurrently.

        Args:
            qa_pairs:  List of question-answer pairs to verify.
            reference: Optional reference context.

        Returns:
            List of :class:`VerificationResult` objects.
        """
        logger.info("Stage 3 — fact-checking %d answers concurrently.", len(qa_pairs))
        tasks = [self._fact_check(qa, reference) for qa in qa_pairs]
        results: List[VerificationResult] = await asyncio.gather(*tasks)
        logger.info("Stage 3 — fact-checking complete.")
        return results

# ---------------------------------------------------------------------------
# Public Entry-Point
# ---------------------------------------------------------------------------

    async def run(self, topic: str, num_questions: int, reference: Optional[str] = None) -> HallucinationResponse:
        """
        Execute the full three-stage hallucination-checking pipeline.

        Args:
            topic:         The subject to probe.
            num_questions: How many questions to generate and test.
            reference:     Optional context document to evaluate against.

        Returns:
            A fully populated :class:`HallucinationResponse`.
        """
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        evaluation_mode = "REFERENCE_BASED" if reference else "LLM_JUDGE_ONLY"

        logger.info(
            "Pipeline start — analysis_id=%s  topic=%r  num_questions=%d  model=%s  mode=%s",
            analysis_id,
            topic,
            num_questions,
            self.model_name,
            evaluation_mode,
        )

        t_start = time.perf_counter()

        # Stage 1: Question Generation
        t0 = time.perf_counter()
        try:
            questions = await self._generate_questions(topic, num_questions, reference)
            q_gen_latency = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:
            logger.error("Pipeline failed in Stage 1 (Question Generation): %s", exc)
            raise

        # Stage 2: Answer Generation
        t1 = time.perf_counter()
        qa_pairs = await self._generate_answers(questions)
        a_gen_latency = (time.perf_counter() - t1) * 1000.0

        # Stage 3: Fact-Checking
        t2 = time.perf_counter()
        verification_results = await self._fact_check_all(qa_pairs, reference)
        fact_check_latency = (time.perf_counter() - t2) * 1000.0

        total_latency = (time.perf_counter() - t_start) * 1000.0

        # Aggregate metrics
        total = len(verification_results)
        accurate_count = sum(1 for r in verification_results if r.verdict == Verdict.ACCURATE)
        hallucinated_count = sum(1 for r in verification_results if r.verdict == Verdict.HALLUCINATED)
        uncertain_count = sum(1 for r in verification_results if r.verdict == Verdict.UNCERTAIN)

        # Deterministic hallucination rate calculation
        hallucination_rate = hallucinated_count / total if total > 0 else 0.0

        # Build human-readable summary
        summary = (
            f"Tested {total} questions about '{topic}'. "
            f"Results: {accurate_count} accurate, "
            f"{hallucinated_count} hallucinated, "
            f"{uncertain_count} uncertain. "
            f"Hallucination rate: {hallucination_rate:.0%}."
        )

        eval_summary = EvaluationSummary(
            accurate_count=accurate_count,
            hallucinated_count=hallucinated_count,
            uncertain_count=uncertain_count,
            hallucination_rate=round(hallucination_rate, 4),
            summary_text=summary,
        )

        logger.info(
            "Pipeline complete — analysis_id=%s  questions_tested=%d  model_used=%s  "
            "evaluation_mode=%s  hallucination_rate=%.4f  total_duration_ms=%.2f  status=success",
            analysis_id,
            total,
            self.model_name,
            evaluation_mode,
            hallucination_rate,
            total_latency,
        )

        return HallucinationResponse(
            topic=topic,
            questions_tested=total,
            hallucination_rate=round(hallucination_rate, 4),
            results=verification_results,
            summary=summary,
            analysis_id=analysis_id,
            timestamp=timestamp,
            model_used=self.model_name,
            evaluation_mode=evaluation_mode,
            question_generation_latency_ms=round(q_gen_latency, 2),
            answer_generation_latency_ms=round(a_gen_latency, 2),
            fact_check_latency_ms=round(fact_check_latency, 2),
            total_latency_ms=round(total_latency, 2),
            evaluation_summary=eval_summary,
        )
