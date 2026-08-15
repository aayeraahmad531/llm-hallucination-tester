import os
import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import asyncio

# Set environment variables before importing app to configure mock environment
os.environ["OPENAI_API_KEY"] = "mock-openai-api-key"
os.environ["MAX_QUESTIONS"] = "5"
os.environ["MAX_CONCURRENCY"] = "2"
os.environ["MAX_RETRIES"] = "2"
os.environ["LLM_TIMEOUT"] = "5.0"

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.main import app
from app.models import Verdict, QuestionGenerationResult, VerificationAssessment
from app.hallucination_checker import HallucinationChecker

class TestHallucinationTester(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        """Test GET /health returns liveness metadata."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "LLM Hallucination Tester")

    def test_root_endpoint(self):
        """Test GET / serves the HTML index page."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("LLM Hallucination Tester", response.text)

    def test_request_validation(self):
        """Test payload validation for POST /check-hallucination."""
        # Empty topic
        response = self.client.post("/check-hallucination", json={"topic": "   ", "num_questions": 3})
        self.assertEqual(response.status_code, 422)

        # Topic too short
        response = self.client.post("/check-hallucination", json={"topic": "ab", "num_questions": 3})
        self.assertEqual(response.status_code, 422)

        # Topic too long (exceeds 500 characters)
        response = self.client.post("/check-hallucination", json={"topic": "a" * 501, "num_questions": 3})
        self.assertEqual(response.status_code, 422)

        # Negative questions
        response = self.client.post("/check-hallucination", json={"topic": "valid topic", "num_questions": 0})
        self.assertEqual(response.status_code, 422)

        # Excessive questions (configured limit is 5)
        response = self.client.post("/check-hallucination", json={"topic": "valid topic", "num_questions": 6})
        self.assertEqual(response.status_code, 400)
        self.assertIn("exceeds the server maximum", response.json()["detail"])


class TestHallucinationPipelineAsync(unittest.IsolatedAsyncioTestCase):
    @patch("app.hallucination_checker.ChatOpenAI")
    async def test_successful_run_llm_only(self, mock_chat_openai):
        """Test a complete successful run of the pipeline in LLM-only mode."""
        # Setup mocks
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        # Mock structured output for Stage 1 (Question Gen)
        mock_question_structured = AsyncMock()
        mock_question_structured.return_value = QuestionGenerationResult(
            questions=["Question 1", "Question 2"]
        )

        # Mock structured output for Stage 3 (Fact Check)
        mock_fact_check_structured = AsyncMock()
        mock_fact_check_structured.side_effect = [
            VerificationAssessment(verdict=Verdict.ACCURATE, confidence=0.9, reasoning="Newton proved it."),
            VerificationAssessment(verdict=Verdict.HALLUCINATED, confidence=0.8, reasoning="Einstein disagreed."),
        ]

        # Mock normal text generation for Stage 2 (Answer Gen)
        mock_instance.side_effect = [
            AIMessage(content="Gravity pulls things down."),
            AIMessage(content="Gravity is electromagnetism."),
        ]

        def custom_with_structured_output(schema, **kwargs):
            if schema == QuestionGenerationResult:
                return mock_question_structured
            if schema == VerificationAssessment:
                return mock_fact_check_structured
            return MagicMock()

        mock_instance.with_structured_output.side_effect = custom_with_structured_output

        # Run pipeline
        checker = HallucinationChecker(model_name="gpt-4o-mini", api_key="test-key")
        result = await checker.run(topic="Gravity", num_questions=2)

        # Assertions
        self.assertEqual(result.topic, "Gravity")
        self.assertEqual(result.questions_tested, 2)
        self.assertEqual(result.evaluation_mode, "LLM_JUDGE_ONLY")
        self.assertEqual(result.hallucination_rate, 0.5)  # 1 out of 2 is hallucinated
        self.assertEqual(len(result.results), 2)

        # Result 1: ACCURATE
        self.assertEqual(result.results[0].question, "Question 1")
        self.assertEqual(result.results[0].answer, "Gravity pulls things down.")
        self.assertEqual(result.results[0].verdict, Verdict.ACCURATE)
        self.assertEqual(result.results[0].confidence, 0.9)
        self.assertEqual(result.results[0].llm_judge_verdict, Verdict.ACCURATE)

        # Result 2: HALLUCINATED
        self.assertEqual(result.results[1].question, "Question 2")
        self.assertEqual(result.results[1].verdict, Verdict.HALLUCINATED)
        self.assertEqual(result.results[1].confidence, 0.8)

        # Latencies check
        self.assertGreater(result.total_latency_ms, 0)
        self.assertGreater(result.question_generation_latency_ms, 0)
        self.assertGreater(result.answer_generation_latency_ms, 0)
        self.assertGreater(result.fact_check_latency_ms, 0)

        # Check structured summary
        self.assertEqual(result.evaluation_summary.accurate_count, 1)
        self.assertEqual(result.evaluation_summary.hallucinated_count, 1)
        self.assertEqual(result.evaluation_summary.uncertain_count, 0)
        self.assertEqual(result.evaluation_summary.hallucination_rate, 0.5)

    @patch("app.hallucination_checker.ChatOpenAI")
    async def test_successful_run_reference_based(self, mock_chat_openai):
        """Test a complete successful run of the pipeline in Reference mode."""
        # Setup mocks
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        # Mock structured output for Stage 1 (Question Gen)
        mock_question_structured = AsyncMock()
        mock_question_structured.return_value = QuestionGenerationResult(
            questions=["Ref Question 1"]
        )

        # Mock structured output for Stage 3 (Fact Check)
        mock_fact_check_structured = AsyncMock()
        mock_fact_check_structured.return_value = VerificationAssessment(
            verdict=Verdict.UNCERTAIN, confidence=0.4, reasoning="Not enough context."
        )

        # Mock normal text generation for Stage 2 (Answer Gen)
        mock_instance.return_value = AIMessage(content="Newton wrote Principia.")

        def custom_with_structured_output(schema, **kwargs):
            if schema == QuestionGenerationResult:
                return mock_question_structured
            if schema == VerificationAssessment:
                return mock_fact_check_structured
            return MagicMock()

        mock_instance.with_structured_output.side_effect = custom_with_structured_output

        # Run pipeline
        checker = HallucinationChecker(model_name="gpt-4o-mini", api_key="test-key")
        result = await checker.run(topic="Newton", num_questions=1, reference="Newton published Principia in 1687.")

        # Assertions
        self.assertEqual(result.topic, "Newton")
        self.assertEqual(result.questions_tested, 1)
        self.assertEqual(result.evaluation_mode, "REFERENCE_BASED")
        self.assertEqual(result.hallucination_rate, 0.0)
        self.assertEqual(result.results[0].verdict, Verdict.UNCERTAIN)
        self.assertEqual(result.results[0].confidence, 0.4)

    @patch("app.hallucination_checker.ChatOpenAI")
    async def test_fallback_parsing(self, mock_chat_openai):
        """Test manual fallback JSON parsing when structured output raises an exception."""
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        # Structured output raises an error (e.g. library mismatch)
        mock_instance.with_structured_output.side_effect = Exception("Structured mode disabled")

        # Mock standard invoke calls returning raw JSON string blocks with markdown fences
        mock_instance.side_effect = [
            AIMessage(content='```json\n{"questions": ["Q1"]}\n```'),  # Stage 1
            AIMessage(content="Newton discovery."),                     # Stage 2
            AIMessage(content='```json\n{"verdict": "ACCURATE", "confidence": 0.95, "reasoning": "Correct."}\n```'), # Stage 3
        ]

        checker = HallucinationChecker(model_name="gpt-4o-mini", api_key="test-key")
        result = await checker.run(topic="Newton", num_questions=1)

        self.assertEqual(result.questions_tested, 1)
        self.assertEqual(result.results[0].verdict, Verdict.ACCURATE)
        self.assertEqual(result.results[0].confidence, 0.95)

    @patch("app.hallucination_checker.ChatOpenAI")
    async def test_partial_failure_handling(self, mock_chat_openai):
        """Test that if one question fails in Stage 2 or 3, it evaluates to UNCERTAIN rather than crashing."""
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        # Mock structured output for Stage 1 (Question Gen) - 2 questions
        mock_question_structured = AsyncMock()
        mock_question_structured.return_value = QuestionGenerationResult(
            questions=["Question 1", "Question 2"]
        )

        # Mock Stage 3 structured fact check: Question 1 succeeds, Question 2 fact check fails (raises error)
        mock_fact_check_structured = AsyncMock()
        mock_fact_check_structured.side_effect = [
            VerificationAssessment(verdict=Verdict.ACCURATE, confidence=0.9, reasoning="Newton proved it."),
            Exception("Simulated upstream timeout on fact check"),
        ]

        # Mock Stage 2 answers: Question 1 succeeds, Question 2 answers fails (raises error)
        mock_instance.side_effect = [
            AIMessage(content="Gravity is real."),
            Exception("Simulated upstream error in Stage 2"),
        ]

        def custom_with_structured_output(schema, **kwargs):
            if schema == QuestionGenerationResult:
                return mock_question_structured
            if schema == VerificationAssessment:
                return mock_fact_check_structured
            return MagicMock()

        mock_instance.with_structured_output.side_effect = custom_with_structured_output

        checker = HallucinationChecker(model_name="gpt-4o-mini", api_key="test-key")
        result = await checker.run(topic="Gravity", num_questions=2)

        self.assertEqual(result.questions_tested, 2)
        
        # Result 1 should succeed
        self.assertEqual(result.results[0].verdict, Verdict.ACCURATE)
        
        # Result 2 should be marked UNCERTAIN because answer generation failed in Stage 2
        self.assertEqual(result.results[1].verdict, Verdict.UNCERTAIN)
        self.assertIn("Answer generation failed", result.results[1].answer)

    @patch("app.hallucination_checker.ChatOpenAI")
    async def test_verdict_normalization_and_fallback(self, mock_chat_openai):
        """Test that invalid verdicts or out of range confidence are safely normalized."""
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        # Disable structured output to test raw response parsing
        mock_instance.with_structured_output.side_effect = Exception("Fallback mode")

        # Case 1: Verdict in lowercase, confidence out of range (> 1.0)
        # Case 2: Verdict is completely invalid (should fallback to UNCERTAIN), confidence < 0
        mock_instance.side_effect = [
            # Case 1
            AIMessage(content='{"questions": ["Q1"]}'),
            AIMessage(content="Answer 1"),
            AIMessage(content='{"verdict": "accurate", "confidence": 1.5, "reasoning": "Correct."}'),
            # Case 2
            AIMessage(content='{"questions": ["Q1"]}'),
            AIMessage(content="Answer 1"),
            AIMessage(content='{"verdict": "SUPER_CORRECT", "confidence": -0.5, "reasoning": "Correct."}'),
        ]

        checker = HallucinationChecker(model_name="gpt-4o-mini", api_key="test-key")
        
        # Run Case 1
        result = await checker.run(topic="Test", num_questions=1)
        self.assertEqual(result.results[0].verdict, Verdict.ACCURATE)
        self.assertEqual(result.results[0].confidence, 1.0) # Clipped to 1.0

        # Run Case 2
        result = await checker.run(topic="Test", num_questions=1)
        self.assertEqual(result.results[0].verdict, Verdict.UNCERTAIN)
        self.assertEqual(result.results[0].confidence, 0.0) # Clipped to 0.0

    @patch("app.hallucination_checker.ChatOpenAI")
    async def test_concurrency_limiting(self, mock_chat_openai):
        """Test concurrency limit (using semaphore) inside the pipeline execution."""
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        # Setup structured mocks
        mock_question_structured = AsyncMock()
        mock_question_structured.return_value = QuestionGenerationResult(
            questions=["Q1", "Q2", "Q3"]
        )
        mock_fact_check_structured = AsyncMock()
        mock_fact_check_structured.return_value = VerificationAssessment(
            verdict=Verdict.ACCURATE, confidence=0.9, reasoning="Ok"
        )

        # We will track active concurrent tasks for Answer Gen
        active_tasks = 0
        max_concurrent_seen = 0

        async def slow_answer(*args, **kwargs):
            nonlocal active_tasks, max_concurrent_seen
            active_tasks += 1
            max_concurrent_seen = max(max_concurrent_seen, active_tasks)
            await asyncio.sleep(0.1)
            active_tasks -= 1
            return AIMessage(content="Answer")

        mock_instance.side_effect = slow_answer

        def custom_with_structured_output(schema, **kwargs):
            if schema == QuestionGenerationResult:
                return mock_question_structured
            if schema == VerificationAssessment:
                return mock_fact_check_structured
            return MagicMock()

        mock_instance.with_structured_output.side_effect = custom_with_structured_output

        # Concurrency limit is set to 2 in env
        checker = HallucinationChecker(model_name="gpt-4o-mini", api_key="test-key", max_concurrency=2)
        result = await checker.run(topic="Test", num_questions=3)

        self.assertEqual(result.questions_tested, 3)
        self.assertLessEqual(max_concurrent_seen, 2) # Concurrency must not exceed 2
        self.assertGreater(max_concurrent_seen, 0)
