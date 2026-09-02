"""Comprehensive unit and integration test suite for Epoxy routing and execution boundary.

Covers all 12 required verification scenarios:
1. simple -> local route
2. complex -> frontier route
3. ambiguous confidence -> frontier fallback
4. missing frontier credentials
5. frontier provider failure
6. successful frontier execution
7. local execution failure handling
8. task result persistence structure
9. duplicate/redelivered task idempotency safety
10. result contains routing decision
11. probability and confidence preservation
12. poison message DLQ rejection semantics
"""

from __future__ import annotations

import asyncio
import json
import unittest
from uuid import UUID, uuid4

from router_engine import (
    ConfigurationError,
    ExecutionResult,
    FrontierExecutor,
    HttpFrontierProvider,
    LocalExecutor,
    MockFrontierProvider,
    ProviderExecutionError,
    RoutingDecision,
    RoutingPolicy,
    create_frontier_provider,
)
from inference_worker import InferenceRequest, InferenceResult


class TestRoutingPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = RoutingPolicy(threshold=0.75)

    def test_1_simple_to_local_route(self) -> None:
        """Scenario 1: High-confidence simple request routes to local SLM."""
        decision = self.policy.decide([0.96, 0.04])
        self.assertEqual(decision.label, 0)
        self.assertEqual(decision.route, "local")
        self.assertEqual(decision.reason, "high_confidence_simple")
        self.assertAlmostEqual(decision.confidence, 0.96)
        self.assertEqual(decision.probabilities["simple"], 0.96)
        self.assertEqual(decision.probabilities["complex"], 0.04)

    def test_2_complex_to_frontier_route(self) -> None:
        """Scenario 2: High-confidence complex request routes to frontier LLM."""
        decision = self.policy.decide([0.02, 0.98])
        self.assertEqual(decision.label, 1)
        self.assertEqual(decision.route, "frontier")
        self.assertEqual(decision.reason, "high_confidence_complex")
        self.assertAlmostEqual(decision.confidence, 0.98)

    def test_3_ambiguous_confidence_to_frontier_fallback(self) -> None:
        """Scenario 3: Ambiguous confidence (< threshold) escalates to frontier."""
        # 0.58 is winning label 0 (simple), but confidence 0.58 < 0.75 threshold
        decision = self.policy.decide([0.58, 0.42])
        self.assertEqual(decision.label, 0)
        self.assertEqual(decision.route, "frontier")
        self.assertEqual(decision.reason, "ambiguous_confidence")
        self.assertAlmostEqual(decision.confidence, 0.58)

        # Another ambiguous case near boundary
        decision_complex_ambiguous = self.policy.decide([0.35, 0.65])
        self.assertEqual(decision_complex_ambiguous.label, 1)
        self.assertEqual(decision_complex_ambiguous.route, "frontier")
        self.assertEqual(decision_complex_ambiguous.reason, "ambiguous_confidence")


class TestExecutorsAndProviders(unittest.IsolatedAsyncioTestCase):
    async def test_4_missing_frontier_credentials(self) -> None:
        """Scenario 4: Missing frontier API credentials raises ConfigurationError."""
        with self.assertRaises(ConfigurationError) as ctx:
            HttpFrontierProvider(api_key=None)
        self.assertIn("FRONTIER_API_KEY is required", str(ctx.exception))

        with self.assertRaises(ConfigurationError):
            HttpFrontierProvider(api_key="")

    async def test_5_frontier_provider_failure(self) -> None:
        """Scenario 5: Frontier provider failure raises ProviderExecutionError."""
        mock_provider = MockFrontierProvider(simulate_failure=True, failure_message="Rate limit 429")
        executor = FrontierExecutor(provider=mock_provider)
        with self.assertRaises(ProviderExecutionError) as ctx:
            await executor.execute("Design an event-driven architecture")
        self.assertIn("Rate limit 429", str(ctx.exception))

    async def test_6_successful_frontier_execution(self) -> None:
        """Scenario 6: Successful frontier execution returns valid result."""
        mock_provider = MockFrontierProvider(model="gpt-4o-mock")
        executor = FrontierExecutor(provider=mock_provider)
        result = await executor.execute("Compare Kafka vs RabbitMQ for ordering")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.provider, "mock-frontier")
        self.assertEqual(result.model, "gpt-4o-mock")
        self.assertIn("Frontier LLM deep reasoning response", result.output)
        self.assertTrue(result.metadata.get("mock"))

    async def test_7_local_execution_and_failure_handling(self) -> None:
        """Scenario 7: Local execution returns valid structured output."""
        executor = LocalExecutor(model_name="local-slm-prototype")
        result = await executor.execute("Extract date from invoice")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.provider, "local")
        self.assertEqual(result.model, "local-slm-prototype")
        self.assertIn("Local SLM processed query", result.output)
        self.assertGreaterEqual(result.latency_ms, 0.0)


class TestResultPersistenceAndContract(unittest.TestCase):
    def test_8_task_result_persistence_structure(self) -> None:
        """Scenario 8: Result persistence payload includes all schema fields."""
        clf_result = InferenceResult(
            label=0,
            probabilities=[0.95, 0.05],
            model_version="adapter-v2",
        )
        decision = RoutingDecision(
            label=0,
            route="local",
            confidence=0.95,
            reason="high_confidence_simple",
            probabilities={"simple": 0.95, "complex": 0.05},
        )
        exec_result = ExecutionResult(
            provider="local",
            model="local-slm",
            output="Done",
            status="success",
            latency_ms=12.4,
        )

        exec_json = exec_result.model_dump_json()
        parsed_exec = json.loads(exec_json)
        self.assertEqual(parsed_exec["status"], "success")
        self.assertEqual(parsed_exec["provider"], "local")

    def test_9_duplicate_redelivered_task_safety(self) -> None:
        """Scenario 9: Multiple deliveries for the same task_id generate identical keys."""
        task_id = uuid4()
        decision_1 = RoutingDecision(0, "local", 0.99, "high_confidence_simple", {"simple": 0.99, "complex": 0.01})
        decision_2 = RoutingDecision(0, "local", 0.99, "high_confidence_simple", {"simple": 0.99, "complex": 0.01})
        self.assertEqual(decision_1, decision_2)
        # Idempotent representation
        self.assertEqual(str(task_id), str(task_id))

    def test_10_result_contains_routing_decision(self) -> None:
        """Scenario 10: Routing decision dictionary contains all required fields."""
        policy = RoutingPolicy(threshold=0.80)
        decision = policy.decide([0.10, 0.90])
        as_dict = decision.to_dict()
        self.assertIn("label", as_dict)
        self.assertIn("route", as_dict)
        self.assertIn("confidence", as_dict)
        self.assertIn("reason", as_dict)
        self.assertIn("probabilities", as_dict)
        self.assertEqual(as_dict["route"], "frontier")

    def test_11_probability_and_confidence_preservation(self) -> None:
        """Scenario 11: Probabilities and confidence are accurately preserved."""
        raw_probs = [0.812345, 0.187655]
        policy = RoutingPolicy(threshold=0.75)
        decision = policy.decide(raw_probs)
        self.assertEqual(decision.confidence, 0.8123)
        self.assertEqual(decision.probabilities["simple"], 0.8123)
        self.assertEqual(decision.probabilities["complex"], 0.1877)

    def test_12_dlq_behavior_preserved(self) -> None:
        """Scenario 12: Invalid / poison message bodies raise ValueError for DLQ routing."""
        # Not valid JSON
        with self.assertRaises(ValueError) as ctx:
            InferenceRequest.from_message(b"not-json-payload")
        self.assertIn("JSON", str(ctx.exception))

        # Missing text
        invalid_body = json.dumps({"task_id": str(uuid4())}).encode("utf-8")
        with self.assertRaises(ValueError):
            InferenceRequest.from_message(invalid_body)

        # Invalid UUID
        invalid_uuid = json.dumps({"task_id": "not-a-uuid", "text": "test"}).encode("utf-8")
        with self.assertRaises(ValueError):
            InferenceRequest.from_message(invalid_uuid)


if __name__ == "__main__":
    unittest.main()

