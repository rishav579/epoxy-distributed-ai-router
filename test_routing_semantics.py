"""Regression tests for routing dataset integrity, label semantics, and benchmark alignment."""

from __future__ import annotations

import csv
from pathlib import Path

from benchmark_router import LABEL_TO_ID, build_synthetic_dataset


def test_label_mapping_semantics() -> None:
    """Verify that routing semantics are strictly 0=simple, 1=complex."""
    assert LABEL_TO_ID == {"simple": 0, "complex": 1}, (
        f"LABEL_TO_ID semantics drifted: expected {{'simple': 0, 'complex': 1}}, got {LABEL_TO_ID}"
    )


def test_dataset_columns_and_labels() -> None:
    """Verify dataset columns, integer labels {0, 1}, and expected class balance."""
    for filename, expected_total, expected_simple, expected_complex in [
        ("train.csv", 400, 200, 200),
        ("val.csv", 100, 50, 50),
    ]:
        path = Path(filename)
        assert path.is_file(), f"Missing required dataset: {filename}"

        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["text", "label"], (
                f"{filename} must have exactly ['text', 'label'] header, got {reader.fieldnames}"
            )
            rows = list(reader)

        assert len(rows) == expected_total, f"{filename} row count mismatch: expected {expected_total}, got {len(rows)}"

        labels = []
        for i, row in enumerate(rows):
            assert row["text"] and row["text"].strip(), f"Empty text in {filename} row {i}"
            try:
                label = int(row["label"])
            except ValueError:
                raise AssertionError(f"Non-integer label '{row['label']}' in {filename} row {i}")
            assert label in (0, 1), f"Invalid label {label} in {filename} row {i}; must be 0 or 1"
            labels.append(label)

        simple_count = sum(lbl == 0 for lbl in labels)
        complex_count = sum(lbl == 1 for lbl in labels)
        assert simple_count == expected_simple, (
            f"{filename} simple class balance mismatch: expected {expected_simple}, got {simple_count}"
        )
        assert complex_count == expected_complex, (
            f"{filename} complex class balance mismatch: expected {expected_complex}, got {complex_count}"
        )


def test_benchmark_dataset_alignment_and_isolation() -> None:
    """Verify benchmark dataset semantics and ensure zero leakage from train/val sets."""
    benchmark_examples = build_synthetic_dataset()
    assert len(benchmark_examples) == 500, f"Expected 500 benchmark examples, got {len(benchmark_examples)}"

    benchmark_prompts = set()
    for ex in benchmark_examples:
        assert ex.prompt and ex.prompt.strip(), "Benchmark example has empty prompt"
        assert ex.label in LABEL_TO_ID, f"Unknown benchmark label '{ex.label}', not in {LABEL_TO_ID}"
        benchmark_prompts.add(ex.prompt)

    # Check zero overlap with train/val datasets
    for filename in ("train.csv", "val.csv"):
        with Path(filename).open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert row["text"] not in benchmark_prompts, (
                    f"Data leakage detected! Prompt from {filename} exists in benchmark set: {row['text']}"
                )


if __name__ == "__main__":
    test_label_mapping_semantics()
    test_dataset_columns_and_labels()
    test_benchmark_dataset_alignment_and_isolation()
    print("All routing semantics and dataset regression tests passed successfully!")
