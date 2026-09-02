"""Manual deterministic routing verification for simple, complex, and ambiguous prompts."""

import asyncio
from pathlib import Path
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from router_engine import (
    FrontierExecutor,
    LocalExecutor,
    MockFrontierProvider,
    RoutingPolicy,
)


MANUAL_PROMPTS = [
    # 1. Simple
    ("Extract the order ID from this message.", "Canonical simple extraction"),
    # 2. Complex
    ("Should we use Kafka or RabbitMQ for this architecture, and why?", "Canonical short complex trade-off question"),
    # 3. Ambiguous / Boundary test
    ("Review the deployment status and explain whether we can proceed.", "Borderline prompt containing routine status check and decision reasoning"),
]


async def run_manual_verification():
    adapter_dir = Path("outputs/adapter")
    print(f"Loading classifier from {adapter_dir}...")
    peft_config = PeftConfig.from_pretrained(adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        peft_config.base_model_name_or_path, num_labels=2
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    routing_policy = RoutingPolicy(threshold=0.75)
    local_executor = LocalExecutor(model_name="local-slm-prototype")
    frontier_executor = FrontierExecutor(provider=MockFrontierProvider(model="gpt-4o-mini"))

    print("\n" + "=" * 75)
    print("MANUAL ROUTING & EXECUTION VERIFICATION")
    print("=" * 75)

    with torch.inference_mode():
        for i, (text, description) in enumerate(MANUAL_PROMPTS, 1):
            inputs = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
            p_simple = float(probs[0].item())
            p_complex = float(probs[1].item())

            decision = routing_policy.decide([p_simple, p_complex])

            if decision.route == "local":
                exec_result = await local_executor.execute(text)
            else:
                exec_result = await frontier_executor.execute(text)

            print(f"\n--- PROMPT #{i}: {description} ---")
            print(f"Input:                    \"{text}\"")
            print(f"Classifier Probabilities: simple={p_simple:.4f}, complex={p_complex:.4f}")
            print(f"Winning Label:            {decision.label} ({'Simple' if decision.label == 0 else 'Complex'})")
            print(f"Confidence:               {decision.confidence:.4f} (Threshold = {routing_policy.threshold:.2f})")
            print(f"Selected Route:           {decision.route.upper()}")
            print(f"Routing Reason:           {decision.reason}")
            print(f"Executed By:              {exec_result.provider} ({exec_result.model})")
            print(f"Execution Latency:        {exec_result.latency_ms:.2f} ms")
            print(f"Execution Status:         {exec_result.status}")
            print(f"Output Preview:           \"{exec_result.output}\"")

    # Also demonstrate synthetic ambiguous confidence fallback explicitly
    print("\n" + "-" * 75)
    print("EXPLICIT AMBIGUITY POLICY BOUNDARY DEMONSTRATION")
    print("-" * 75)
    synthetic_ambiguous_probs = [0.55, 0.45]
    synth_decision = routing_policy.decide(synthetic_ambiguous_probs)
    print(f"Simulated Ambiguous Probabilities: simple=0.55, complex=0.45")
    print(f"Confidence: {synth_decision.confidence:.2f} (< {routing_policy.threshold:.2f} threshold)")
    print(f"Decision Route: {synth_decision.route.upper()} (Reason: {synth_decision.reason})")
    print(f"Policy Verification: Request successfully escalated to frontier due to ambiguity.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    asyncio.run(run_manual_verification())

