"""Async inference gateway. Workers consume `INFERENCE_QUEUE` and update task status.

Run with: uvicorn inference_gateway:app --host 0.0.0.0 --port 8000
Required environment: DATABASE_URL and AMQP_URL.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal, cast
from uuid import UUID, uuid4

import aio_pika
import asyncpg
from aio_pika import DeliveryMode, Message
from aio_pika.abc import AbstractRobustConnection
from aio_pika.exceptions import AMQPException
from aio_pika.pool import Pool
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest


TaskStatus = Literal["queued", "processing", "completed", "failed"]
RabbitPool = Pool[AbstractRobustConnection]


@dataclass(frozen=True)
class Settings:
    database_url: str
    amqp_url: str
    queue_name: str
    dead_letter_exchange: str
    dead_letter_queue: str

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
        )


@dataclass(frozen=True)
class Resources:
    database: asyncpg.Pool
    rabbit: RabbitPool
    settings: Settings


class PredictRequest(BaseModel):
    text: Annotated[str, Field(min_length=1, max_length=20_000)]


class PredictResponse(BaseModel):
    task_id: UUID
    status: Literal["queued"]


class QueueMessage(BaseModel):
    task_id: UUID
    text: str


class TaskStatusResponse(BaseModel):
    task_id: UUID
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    error: str | None


async def open_rabbit_connection(settings: Settings) -> AbstractRobustConnection:
    return await aio_pika.connect_robust(settings.amqp_url)


async def initialize_database(database: asyncpg.Pool) -> None:
    await database.execute(
        """
        CREATE TABLE IF NOT EXISTS inference_tasks (
            task_id UUID PRIMARY KEY,
            status TEXT NOT NULL CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            error TEXT
        )
        """
    )


async def initialize_queue(rabbit: RabbitPool, settings: Settings) -> None:
    async with rabbit.acquire() as connection:
        channel = await connection.channel()
        try:
            dead_letter_exchange = await channel.declare_exchange(
                settings.dead_letter_exchange, aio_pika.ExchangeType.DIRECT, durable=True
            )
            dead_letter_queue = await channel.declare_queue(settings.dead_letter_queue, durable=True)
            await dead_letter_queue.bind(dead_letter_exchange, routing_key=settings.queue_name)
            await channel.declare_queue(
                settings.queue_name,
                durable=True,
                arguments={
                    "x-dead-letter-exchange": settings.dead_letter_exchange,
                    "x-dead-letter-routing-key": settings.queue_name,
                },
            )
        finally:
            await channel.close()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings.from_environment()
    database = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=10)
    rabbit = Pool(lambda: open_rabbit_connection(settings), max_size=10)
    try:
        await initialize_database(database)
        await initialize_queue(rabbit, settings)
        app.state.resources = Resources(database=database, rabbit=rabbit, settings=settings)
        yield
    finally:
        await rabbit.close()
        await database.close()


app = FastAPI(title="Inference Gateway", lifespan=lifespan)


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def get_resources(request: Request) -> Resources:
    return cast(Resources, request.app.state.resources)


ResourcesDependency = Annotated[Resources, Depends(get_resources)]


async def create_task(database: asyncpg.Pool, task_id: UUID) -> None:
    await database.execute(
        "INSERT INTO inference_tasks (task_id, status) VALUES ($1, 'queued')", task_id
    )


async def mark_publish_failure(database: asyncpg.Pool, task_id: UUID, error: Exception) -> None:
    await database.execute(
        """
        UPDATE inference_tasks
        SET status = 'failed', error = $2, updated_at = NOW()
        WHERE task_id = $1
        """,
        task_id,
        str(error)[:1_000],
    )


async def publish_task(resources: Resources, task_id: UUID, text: str) -> None:
    # The worker receives only the ID and text; task status remains the DB's source of truth.
    message = Message(
        body=QueueMessage(task_id=task_id, text=text).model_dump_json().encode("utf-8"),
        content_type="application/json",
        delivery_mode=DeliveryMode.PERSISTENT,
    )
    async with resources.rabbit.acquire() as connection:
        channel = await connection.channel()
        try:
            await channel.default_exchange.publish(message, routing_key=resources.settings.queue_name)
        finally:
            await channel.close()


@app.post("/predict", response_model=PredictResponse, status_code=status.HTTP_202_ACCEPTED)
async def predict(payload: PredictRequest, resources: ResourcesDependency) -> PredictResponse:
    task_id = uuid4()
    await create_task(resources.database, task_id)
    try:
        await publish_task(resources, task_id, payload.text)
    except (AMQPException, OSError) as error:
        await mark_publish_failure(resources.database, task_id, error)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Queue unavailable") from error
    return PredictResponse(task_id=task_id, status="queued")


@app.get("/status/{task_id}", response_model=TaskStatusResponse)
async def task_status(task_id: UUID, resources: ResourcesDependency) -> TaskStatusResponse:
    record = await resources.database.fetchrow(
        """
        SELECT task_id, status, created_at, updated_at, error
        FROM inference_tasks WHERE task_id = $1
        """,
        task_id,
    )
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    return TaskStatusResponse(
        task_id=cast(UUID, record["task_id"]),
        status=cast(TaskStatus, record["status"]),
        created_at=cast(datetime, record["created_at"]),
        updated_at=cast(datetime, record["updated_at"]),
        error=cast(str | None, record["error"]),
    )
