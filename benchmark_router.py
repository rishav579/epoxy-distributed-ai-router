"""Benchmark the local LoRA router for classification quality, latency, and cost.

The generated data is intentionally deterministic and human-readable.  Replace it
with ``--dataset path/to/prompts.csv`` when a representative labelled set exists.
Required CSV columns: ``prompt`` and ``label`` (``simple`` or ``complex``).
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from peft import PeftConfig, PeftModel
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase


LABEL_TO_ID = {"simple": 0, "complex": 1}
DEFAULT_BATCH_SIZES = (1, 8, 32)


@dataclass(frozen=True)
class Example:
    prompt: str
    label: str


@dataclass(frozen=True)
class LoadedRouter:
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    device: torch.device


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_synthetic_dataset() -> list[Example]:
    """Create exactly 250 simple and 250 complex, varied routing prompts."""
    subjects = (
        "a password reset", "an invoice", "a delivery tracking number", "meeting notes",
        "a calendar invite", "a subscription", "a product return", "a support ticket",
        "a user profile", "a service status page",
    )
    simple_templates = (
        "Summarize {subject} in one sentence.",
        "Extract the main date from {subject}.",
        "Classify the sentiment of {subject} as positive or negative.",
        "Rewrite {subject} as a short, polite acknowledgement.",
        "List the three key facts in {subject}.",
        "Translate this short message about {subject} into Spanish.",
        "Answer yes or no: does {subject} need attention?",
        "Turn {subject} into a concise email subject line.",
        "Identify the named person or company in {subject}.",
        "Correct grammar in this note about {subject}.",
        "Assign one category to {subject}: billing, account, or delivery.",
        "Give a five-word title for {subject}.",
        "Extract the requested action from {subject}.",
        "Detect whether {subject} contains an order identifier.",
        "Return a two-bullet summary of {subject}.",
        "Format {subject} as a short checklist.",
        "Find the urgency level in {subject}.",
        "Convert the date mentioned in {subject} to ISO format.",
        "Generate one follow-up question about {subject}.",
        "State the language used in {subject}.",
        "Redact a phone number from {subject}.",
        "Extract all email addresses from {subject}.",
        "Provide a one-line response to {subject}.",
        "Determine if {subject} is a question.",
        "Label {subject} as informational or actionable.",
    )
    complex_templates = (
        "Compare three vendor proposals for {subject}, rank trade-offs, and recommend a strategy with risks.",
        "Design a migration plan for {subject} across regions, including rollback criteria and a dependency graph.",
        "Investigate why {subject} caused a production incident; form hypotheses and propose instrumented experiments.",
        "Draft a compliance analysis for {subject} across GDPR, SOC 2, and retention requirements with caveats.",
        "Build a quarterly forecast from {subject}, state assumptions, and run best/base/worst-case scenarios.",
        "Propose an architecture for {subject} that balances consistency, availability, security, and operating cost.",
        "Write a detailed incident postmortem for {subject}, including timeline, root causes, and corrective actions.",
        "Analyze conflicting stakeholder requirements around {subject} and propose a phased decision framework.",
        "Create a test strategy for {subject} covering unit, integration, load, chaos, and acceptance testing.",
        "Evaluate legal, operational, and reputational risks of changing {subject}; recommend mitigations.",
        "Develop a multi-step debugging plan for intermittent failures involving {subject}, databases, and queues.",
        "Explain how {subject} should be redesigned for tenfold traffic while meeting a strict latency SLO.",
        "Create an implementation roadmap for {subject} with milestones, owners, estimates, and dependencies.",
        "Reconcile inconsistent data sources for {subject} and specify validation rules and an audit trail.",
        "Assess whether to build or buy for {subject}, using a weighted decision matrix and sensitivity analysis.",
        "Produce a security threat model for {subject} with attack trees and prioritized controls.",
        "Design an A/B experiment for {subject}, including sample-size assumptions and success metrics.",
        "Plan a disaster recovery exercise for {subject}, including RTO/RPO targets and failure simulations.",
        "Develop an internationalization strategy for {subject}, accounting for languages, taxes, and local regulations.",
        "Review a proposed policy for {subject}, identify edge cases, and write an escalation procedure.",
        "Construct a capacity model for {subject} under seasonal demand and recommend an autoscaling policy.",
        "Map end-to-end data lineage for {subject}, identify privacy risks, and propose governance controls.",
        "Create a negotiation brief about {subject}, with alternatives, concessions, and stakeholder incentives.",
        "Analyze a five-year total cost of ownership for {subject}, including uncertainty and opportunity cost.",
        "Propose a resilient event-driven workflow for {subject}, with idempotency, retries, DLQs, and observability.",
    )
    simple = [Example(template.format(subject=subject), "simple") for template in simple_templates for subject in subjects]
    complex_ = [Example(template.format(subject=subject), "complex") for template in complex_templates for subject in subjects]
    return simple + complex_


def load_or_create_dataset(path: Path) -> list[Example]:
    if path.is_file():
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        examples = [Example(row["prompt"].strip(), row["label"].strip().lower()) for row in rows]
    else:
        examples = build_synthetic_dataset()
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("prompt", "label"))
            writer.writeheader()
            writer.writerows({"prompt": item.prompt, "label": item.label} for item in examples)
        print(f"Created synthetic dataset: {path}")
    if len(examples) != 500:
        raise ValueError(f"Expected exactly 500 evaluation prompts; found {len(examples)} in {path}")
    if any(not item.prompt or item.label not in LABEL_TO_ID for item in examples):
        raise ValueError("Every row needs a non-empty prompt and a label of 'simple' or 'complex'.")
    return examples


def load_router(adapter_dir: Path, num_labels: int) -> LoadedRouter:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"No LoRA adapter configuration found: {config_path}")
    config = PeftConfig.from_pretrained(adapter_dir)
    if not config.base_model_name_or_path:
        raise RuntimeError("The adapter does not specify its base model.")
    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)
    model.eval()
    return LoadedRouter(model, tokenizer, device)


def predict_batches(router: LoadedRouter, prompts: Iterable[str], batch_size: int, max_length: int) -> tuple[list[int], list[float]]:
    prompt_list = list(prompts)
    predictions: list[int] = []
    latencies_ms: list[float] = []
    with torch.inference_mode():
        for index in range(0, len(prompt_list), batch_size):
            batch = prompt_list[index:index + batch_size]
            inputs = router.tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            inputs = {name: value.to(router.device) for name, value in inputs.items()}
            if router.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter_ns()
            outputs = router.model(**inputs)
            if router.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            predictions.extend(torch.argmax(outputs.logits, dim=-1).cpu().tolist())
            latencies_ms.append(elapsed_ms)
    return predictions, latencies_ms


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile from no values.")
    position = (len(ordered) - 1) * percentile_value / 100
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def benchmark_latency(router: LoadedRouter, prompts: list[str], max_length: int) -> dict[int, dict[str, float]]:
    results: dict[int, dict[str, float]] = {}
    for batch_size in DEFAULT_BATCH_SIZES:
        # Warm up model kernels and tokenizer-independent execution; do not record these runs.
        predict_batches(router, prompts[:batch_size], batch_size, max_length)
        _, samples = predict_batches(router, prompts, batch_size, max_length)
        results[batch_size] = {
            "p50": percentile(samples, 50),
            "p95": percentile(samples, 95),
            "p99": percentile(samples, 99),
            "mean": statistics.fmean(samples),
            "requests": float(len(samples)),
        }
    return results


def estimate_cost(router: LoadedRouter, examples: list[Example], predictions: list[int], max_length: int, input_rate: float, output_rate: float, output_tokens: int) -> dict[str, float]:
    encoded = router.tokenizer([item.prompt for item in examples], truncation=True, max_length=max_length)
    input_tokens = sum(len(token_ids) for token_ids in encoded["input_ids"])
    all_gpt4o = (input_tokens * input_rate + len(examples) * output_tokens * output_rate) / 1_000_000
    frontier_requests = sum(prediction == LABEL_TO_ID["complex"] for prediction in predictions)
    dynamic = (input_tokens * input_rate + frontier_requests * output_tokens * output_rate) / 1_000_000
    return {
        "input_tokens": float(input_tokens),
        "frontier_requests": float(frontier_requests),
        "all_gpt4o": all_gpt4o,
        "dynamic": dynamic,
        "savings": all_gpt4o - dynamic,
        "savings_percent": 100 * (all_gpt4o - dynamic) / all_gpt4o if all_gpt4o else 0.0,
    }


def render_report(precision: float, recall: float, f1: float, matrix: list[list[int]], latency: dict[int, dict[str, float]], cost: dict[str, float], device: torch.device) -> str:
    latency_rows = "\n".join(
        f"| {size} | {values['requests']:.0f} | {values['p50']:.2f} | {values['p95']:.2f} | {values['p99']:.2f} | {values['mean']:.2f} |"
        for size, values in latency.items()
    )
    return f"""# Router benchmark report

## Run summary

- Evaluation prompts: 500 (250 simple, 250 complex)
- Device: `{device.type}`
- Positive class: `complex`

## Classification quality

| Precision | Recall | F1-score |
| ---: | ---: | ---: |
| {precision:.4f} | {recall:.4f} | {f1:.4f} |

Confusion matrix (rows: actual `simple`, `complex`; columns: predicted `simple`, `complex`):

| Actual \\ Predicted | Simple | Complex |
| --- | ---: | ---: |
| Simple | {matrix[0][0]} | {matrix[0][1]} |
| Complex | {matrix[1][0]} | {matrix[1][1]} |

## Inference latency

Latency is model forward-pass time per batch after one warm-up batch; values are milliseconds.

| Batch size | Measured batches | P50 | P95 | P99 | Mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
{latency_rows}

## Cost projection

Projection uses the configured GPT-4o input/output token rates, actual tokenized prompt lengths,
and {int(cost['frontier_requests'])} classifier-selected frontier requests. Local SLM inference is
treated as $0 marginal API cost.

| Scenario | Projected cost |
| --- | ---: |
| 100% GPT-4o | ${cost['all_gpt4o']:.6f} |
| Dynamic router | ${cost['dynamic']:.6f} |
| Savings | ${cost['savings']:.6f} ({cost['savings_percent']:.2f}%) |
"""


def log_wandb(metrics: dict[str, float], report_path: Path) -> None:
    if not os.environ.get("WANDB_API_KEY"):
        return
    try:
        import wandb
    except ImportError as error:
        print(f"WANDB_API_KEY is set, but wandb is unavailable; skipping W&B logging ({error}).")
        return
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "router-benchmark"), job_type="evaluation")
    try:
        wandb.log(metrics)
        wandb.save(str(report_path))
    finally:
        run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a local LoRA inference router.")
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/adapter"))
    parser.add_argument("--dataset", type=Path, default=Path("benchmark_dataset.csv"))
    parser.add_argument("--report", type=Path, default=Path("benchmark_report.md"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--gpt4o-input-per-million", type=float, default=2.50)
    parser.add_argument("--gpt4o-output-per-million", type=float, default=10.00)
    parser.add_argument("--estimated-output-tokens", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_length < 1 or args.num_labels != 2 or args.estimated_output_tokens < 0:
        raise ValueError("--max-length must be positive, --num-labels must be 2, and output tokens cannot be negative.")
    if args.gpt4o_input_per_million < 0 or args.gpt4o_output_per_million < 0:
        raise ValueError("Token prices cannot be negative.")
    examples = load_or_create_dataset(args.dataset)
    router = load_router(args.adapter_dir, args.num_labels)
    prompts = [item.prompt for item in examples]
    labels = [LABEL_TO_ID[item.label] for item in examples]
    predictions, _ = predict_batches(router, prompts, 32, args.max_length)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", pos_label=1, zero_division=0)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1]).tolist()
    latency = benchmark_latency(router, prompts, args.max_length)
    cost = estimate_cost(router, examples, predictions, args.max_length, args.gpt4o_input_per_million, args.gpt4o_output_per_million, args.estimated_output_tokens)
    report = render_report(precision, recall, f1, matrix, latency, cost, router.device)
    args.report.write_text(report, encoding="utf-8")
    print(report)
    print(f"Markdown report written to: {args.report}")
    wandb_metrics = {
        "classification/precision_complex": precision, "classification/recall_complex": recall,
        "classification/f1_complex": f1, "cost/all_gpt4o_usd": cost["all_gpt4o"],
        "cost/dynamic_usd": cost["dynamic"], "cost/savings_usd": cost["savings"],
        "cost/savings_percent": cost["savings_percent"],
    }
    for batch_size, values in latency.items():
        for metric, value in values.items():
            wandb_metrics[f"latency_ms/batch_{batch_size}_{metric}"] = value
    log_wandb(wandb_metrics, args.report)


if __name__ == "__main__":
    main()
