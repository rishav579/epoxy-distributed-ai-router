"""Consume inference requests and write results to PostgreSQL.

The local registry simulation loads the newest adapter directory beneath
LOCAL_MODEL_REGISTRY_DIR (default: outputs/adapter). A registry entry is a
Hugging Face PEFT adapter directory containing adapter_config.json.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import aio_pika
import asyncpg
import torch
from aio_pika.abc import AbstractIncomingMessage, AbstractQueue, AbstractRobustConnection
from peft import PeftConfig, PeftModel
from prometheus_client import Counter, start_http_server
from torch import Tensor, nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase


LOGGER = logging.getLogger("inference_worker")
COMPLETED_MESSAGES = Counter("inference_worker_completed_messages_total", "Successfully acknowledged messages")
FAILED_MESSAGES = Counter("inference_worker_failed_messages_total", "Messages rejected to the dead letter queue")


@dataclass(frozen=True)
class Settings:
    database_url: str
    amqp_url: str
    queue_name: str
    dead_letter_exchange: str
    dead_letter_queue: str
    registry_path: Path
    max_length: int
    num_labels: int

    @classmethod
    def from_environment(cls) -> Settings:
        database_url = os.environ.get("DATABASE_URL")
        amqp_url = os.environ.get("AMQP_URL")
        if not database_url or not amqp_url:
            raise RuntimeError("DATABASE_URL and AMQP_URL must be set")
        return cls(
            database_url=database_url,
            amqp_url=amqp_url,
            queue_name=os.environ.get("INFERENCE_QUEUE", "inference.requests"),
            dead_letter_exchange=os.environ.get("INFERENCE_DLX", "inference.dlx"),
            dead_letter_queue=os.environ.get("INFERENCE_DLQ", "inference.requests.dlq"),
            registry_path=Path(os.environ.get("LOCAL_MODEL_REGISTRY_DIR", "outputs/adapter")),
            max_length=int(os.environ.get("MAX_LENGTH", "256")),
            num_labels=int(os.environ.get("NUM_LABELS", "2")),
        )


@dataclass(frozen=True)
class InferenceRequest:
    task_id: UUID
    text: str

    @classmethod
    def from_message(cls, body: bytes) -> InferenceRequest:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Message body is not valid UTF-8 JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("Message body must be a JSON object")
        task_id, text = payload.get("task_id"), payload.get("text")
        if not isinstance(task_id, str) or not isinstance(text, str) or not text.strip():
            raise ValueError("Message must contain a UUID task_id and non-empty text")
        try:
            return cls(task_id=UUID(task_id), text=text)
        except ValueError as error:
            raise ValueError("Message contains an invalid task_id") from error


@dataclass(frozen=True)
class InferenceResult:
    label: int
    probabilities: list[float]
    model_version: str


@dataclass(frozen=True)
class LoadedModel:
    model: nn.Module
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    version: str
    max_length: int

    def predict(self, text: str) -> InferenceResult:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs: dict[str, Tensor] = {name: value.to(self.device) for name, value in encoded.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1).squeeze(0)
        return InferenceResult(
            label=int(torch.argmax(probabilities).item()),
            probabilities=[float(value) for value in probabilities.cpu().tolist()],
            model_version=self.version,
        )


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_latest_adapter(registry_path: Path) -> Path:
    if (registry_path / "adapter_config.json").is_file():
        return registry_path
    candidates = [
        path for path in registry_path.iterdir()
        if path.is_dir() and (path / "adapter_config.json").is_file()
    ] if registry_path.is_dir() else []
    if not candidates:
        raise FileNotFoundError(
            f"No PEFT adapter found in local model registry: {registry_path.resolve()}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_model(settings: Settings) -> LoadedModel:
    adapter_path = resolve_latest_adapter(settings.registry_path)
    peft_config = PeftConfig.from_pretrained(adapter_path)
    if not peft_config.base_model_name_or_path:
        raise RuntimeError(f"Adapter has no base model reference: {adapter_path}")
    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        peft_config.base_model_name_or_path,
        num_labels=settings.num_labels,
        ignore_mismatched_sizes=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
    model.eval()
    LOGGER.info("Loaded model version=%s device=%s", adapter_path.name, device.type)
    return LoadedModel(model=model, tokenizer=tokenizer, device=device, version=adapter_path.name, max_length=settings.max_length)


async def initialize_database(database: asyncpg.Pool) -> None:
    await database.execute(
        """
        CREATE TABLE IF NOT EXISTS inference_tasks (
            task_id UUID PRIMARY KEY,
            status TEXT NOT NULL CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS inference_results (
            task_id UUID PRIMARY KEY REFERENCES inference_tasks(task_id),
            label INTEGER NOT NULL,
            probabilities JSONB NOT NULL,
            model_version TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


async def initialize_queue(connection: AbstractRobustConnection, settings: Settings) -> AbstractQueue:
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)
    dead_letter_exchange = await channel.declare_exchange(
        settings.dead_letter_exchange, aio_pika.ExchangeType.DIRECT, durable=True
    )
    dead_letter_queue = await channel.declare_queue(settings.dead_letter_queue, durable=True)
    await dead_letter_queue.bind(dead_letter_exchange, routing_key=settings.queue_name)
    return await channel.declare_queue(
        settings.queue_name,
        durable=True,
        arguments={
            "x-dead-letter-exchange": settings.dead_letter_exchange,
            "x-dead-letter-routing-key": settings.queue_name,
        },
    )


async def mark_processing(database: asyncpg.Pool, task_id: UUID) -> None:
    updated = await database.execute(
        """
        UPDATE inference_tasks SET status = 'processing', error = NULL, updated_at = NOW()
        WHERE task_id = $1
        """,
        task_id,
    )
    if updated != "UPDATE 1":
        raise ValueError(f"Unknown task_id: {task_id}")


async def save_result(database: asyncpg.Pool, task_id: UUID, result: InferenceResult) -> None:
    async with database.acquire() as connection, connection.transaction():
        await connection.execute(
            """
            INSERT INTO inference_results (task_id, label, probabilities, model_version)
            VALUES ($1, $2, $3::jsonb, $4)
            ON CONFLICT (task_id) DO UPDATE SET
                label = EXCLUDED.label,
                probabilities = EXCLUDED.probabilities,
                model_version = EXCLUDED.model_version,
                created_at = NOW()
            """,
            task_id,
            result.label,
            json.dumps(result.probabilities),
            result.model_version,
        )
        updated = await connection.execute(
            """
            UPDATE inference_tasks SET status = 'completed', error = NULL, updated_at = NOW()
            WHERE task_id = $1
            """,
            task_id,
        )
        if updated != "UPDATE 1":
            raise ValueError(f"Unknown task_id: {task_id}")


def release_oom_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


async def process_message(
    message: AbstractIncomingMessage,
    database: asyncpg.Pool,
    loaded_model: LoadedModel,
) -> None:
    try:
        request = InferenceRequest.from_message(message.body)
        await mark_processing(database, request.task_id)
        result = loaded_model.predict(request.text)
        await save_result(database, request.task_id, result)
    except Exception:
        LOGGER.exception("Inference failed; sending delivery tag=%s to DLQ", message.delivery_tag)
        release_oom_memory(loaded_model.device)
        await message.nack(requeue=False)
        FAILED_MESSAGES.inc()
        return
    await message.ack()
    COMPLETED_MESSAGES.inc()
    LOGGER.info("Completed task_id=%s label=%s", request.task_id, result.label)


def install_shutdown_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()

    def request_stop() -> None:
        LOGGER.info("Shutdown requested; worker will finish the current message")
        stop_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, request_stop)
        except NotImplementedError:
            signal.signal(signum, lambda _signal, _frame: loop.call_soon_threadsafe(request_stop))


async def run_worker() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = Settings.from_environment()
    loaded_model = load_model(settings)
    database = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=5)
    connection = await aio_pika.connect_robust(settings.amqp_url)
    stop_event = asyncio.Event()
    install_shutdown_handlers(stop_event)
    try:
        await initialize_database(database)
        queue = await initialize_queue(connection, settings)
        start_http_server(8001)
        LOGGER.info("Worker consuming queue=%s", settings.queue_name)
        while not stop_event.is_set():
            try:
                message = await queue.get(no_ack=False, fail=False, timeout=1.0)
            except TimeoutError:
                continue
            if message is not None:
                await process_message(message, database, loaded_model)
    finally:
        await connection.close()
        await database.close()
        LOGGER.info("Worker stopped")


if __name__ == "__main__":
    asyncio.run(run_worker())
