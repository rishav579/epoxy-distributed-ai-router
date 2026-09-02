"""Routing engine and execution abstractions for Epoxy.

Decouples complexity classification from execution:
- Evaluates classifier probabilities against a configurable confidence threshold
- Enforces an ambiguity escalation policy (uncertain requests escalate to frontier)
- Dispatches to LocalExecutor (SLM path) or FrontierExecutor (frontier provider path)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field


RouteType = Literal["local", "frontier"]
RoutingReason = Literal[
    "high_confidence_simple",
    "high_confidence_complex",
    "ambiguous_confidence",
]


class ConfigurationError(Exception):
    """Raised when provider configuration or credentials are missing."""


class ProviderExecutionError(Exception):
    """Raised when an external or mock provider execution fails."""


@dataclass(frozen=True)
class RoutingDecision:
    label: int
    route: RouteType
    confidence: float
    reason: RoutingReason
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "route": self.route,
            "confidence": self.confidence,
            "reason": self.reason,
            "probabilities": self.probabilities,
        }


class RoutingPolicy:
    """Confidence-aware routing policy with safety-first ambiguity escalation."""

    def __init__(self, threshold: float = 0.75) -> None:
        if not 0.5 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.5 and 1.0")
        self.threshold = threshold

    def decide(self, probabilities: list[float]) -> RoutingDecision:
        if len(probabilities) < 2:
            raise ValueError("Expected at least 2 class probabilities (simple, complex)")

        p_simple = float(probabilities[0])
        p_complex = float(probabilities[1])
        prob_dict = {
            "simple": round(p_simple, 4),
            "complex": round(p_complex, 4),
        }

        label = 0 if p_simple >= p_complex else 1
        confidence = round(max(p_simple, p_complex), 4)

        if confidence < self.threshold:
            # Ambiguous confidence -> safety fallback to frontier
            return RoutingDecision(
                label=label,
                route="frontier",
                confidence=confidence,
                reason="ambiguous_confidence",
                probabilities=prob_dict,
            )

        if label == 0:
            return RoutingDecision(
                label=label,
                route="local",
                confidence=confidence,
                reason="high_confidence_simple",
                probabilities=prob_dict,
            )

        return RoutingDecision(
            label=label,
            route="frontier",
            confidence=confidence,
            reason="high_confidence_complex",
            probabilities=prob_dict,
        )


class ExecutionResult(BaseModel):
    provider: str
    model: str
    output: str
    status: Literal["success", "failed"]
    latency_ms: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LocalExecutor:
    """Execution boundary for the local SLM path.
    
    Serves as the pluggable adapter where a real generative local model
    (e.g., Llama/Mistral/Gemma quantized) connects.
    """

    def __init__(self, model_name: str = "local-slm-prototype") -> None:
        self.model_name = model_name

    async def execute(self, text: str) -> ExecutionResult:
        start_time = time.perf_counter()
        preview = text[:80] + ("..." if len(text) > 80 else "")
        output = f"[Local SLM processed query]: {preview}"
        latency_ms = max(round((time.perf_counter() - start_time) * 1000, 2), 0.01)
        return ExecutionResult(
            provider="local",
            model=self.model_name,
            output=output,
            status="success",
            latency_ms=latency_ms,
            metadata={"tier": "slm", "char_length": len(text)},
        )


class FrontierProvider(ABC):
    """Abstract interface for frontier LLM execution."""

    @abstractmethod
    async def generate(self, text: str) -> ExecutionResult:
        """Execute inference against the frontier provider."""


class MockFrontierProvider(FrontierProvider):
    """Deterministic mock provider for offline testing and CI environments."""

    def __init__(
        self,
        model: str = "gpt-4o-mock",
        simulate_failure: bool = False,
        failure_message: str = "Simulated frontier provider error",
    ) -> None:
        self.model = model
        self.simulate_failure = simulate_failure
        self.failure_message = failure_message

    async def generate(self, text: str) -> ExecutionResult:
        start_time = time.perf_counter()
        if self.simulate_failure:
            raise ProviderExecutionError(self.failure_message)

        preview = text[:80] + ("..." if len(text) > 80 else "")
        output = f"[Frontier LLM deep reasoning response]: Completed high-complexity analysis for '{preview}'."
        latency_ms = max(round((time.perf_counter() - start_time) * 1000, 2), 0.01)
        return ExecutionResult(
            provider="mock-frontier",
            model=self.model,
            output=output,
            status="success",
            latency_ms=latency_ms,
            metadata={
                "tier": "frontier",
                "mock": True,
                "estimated_tokens": len(text.split()) + 40,
            },
        )


class HttpFrontierProvider(FrontierProvider):
    """Real HTTP adapter for OpenAI-compatible or frontier LLM endpoints."""

    def __init__(
        self,
        api_key: str | None,
        api_base: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 30.0,
    ) -> None:
        if not api_key:
            raise ConfigurationError("FRONTIER_API_KEY is required for HttpFrontierProvider")
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def generate(self, text: str) -> ExecutionResult:
        import httpx

        start_time = time.perf_counter()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1024,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                return ExecutionResult(
                    provider="frontier-http",
                    model=self.model,
                    output=content,
                    status="success",
                    latency_ms=latency_ms,
                    metadata={"usage": usage},
                )
        except Exception as error:
            raise ProviderExecutionError(f"HTTP frontier call failed: {error}") from error


class FrontierExecutor:
    """Dispatches requests to the configured frontier provider."""

    def __init__(self, provider: FrontierProvider) -> None:
        self.provider = provider

    async def execute(self, text: str) -> ExecutionResult:
        return await self.provider.generate(text)


def create_frontier_provider(
    provider_type: str,
    api_key: str | None = None,
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
) -> FrontierProvider:
    """Factory creating the appropriate frontier provider adapter."""
    normalized = provider_type.strip().lower()
    if normalized == "mock":
        return MockFrontierProvider(model=model)
    if normalized in ("http", "openai", "real"):
        return HttpFrontierProvider(api_key=api_key, api_base=api_base, model=model)
    raise ValueError(f"Unknown frontier provider type: {provider_type} (expected 'mock' or 'http')")

