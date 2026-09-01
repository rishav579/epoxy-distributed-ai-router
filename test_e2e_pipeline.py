"""Concurrent end-to-end validation for the inference gateway and worker."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID

import httpx


VALID_REQUESTS = 50
POISON_REQUESTS = 5
TERMINAL_STATUSES = frozenset({"completed", "failed"})


@dataclass(frozen=True)
class Settings:
    gateway_url: str
    rabbitmq_api_url: str
    rabbitmq_user: str
    rabbitmq_password: str
    timeout_seconds: float

    @classmethod
    def from_environment(cls) -> Settings:
        return cls(
            gateway_url=os.environ.get("GATEWAY_URL", "http://127.0.0.1:8000").rstrip("/"),
            rabbitmq_api_url=os.environ.get("RABBITMQ_API_URL", "http://127.0.0.1:15672").rstrip("/"),
            rabbitmq_user=os.environ.get("RABBITMQ_USER", "inference"),
            rabbitmq_password=os.environ.get("RABBITMQ_PASSWORD", "inference_local_password"),
            timeout_seconds=float(os.environ.get("E2E_TIMEOUT_SECONDS", "120")),
        )


def object_json(response: httpx.Response) -> Mapping[str, Any]:
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise AssertionError(f"Expected object JSON from {response.request.url}")
    return cast(Mapping[str, Any], payload)


async def submit_prediction(client: httpx.AsyncClient, gateway_url: str, index: int) -> UUID:
    response = await client.post(f"{gateway_url}/predict", json={"text": f"e2e inference request {index}"})
    if response.status_code != httpx.codes.ACCEPTED:
        raise AssertionError(f"POST /predict returned {response.status_code}: {response.text}")
    payload = object_json(response)
    if payload.get("status") != "queued" or not isinstance(payload.get("task_id"), str):
        raise AssertionError(f"Invalid prediction response: {payload}")
    return UUID(payload["task_id"])


async def queue_message_count(client: httpx.AsyncClient, settings: Settings, queue_name: str) -> int:
    response = await client.get(
        f"{settings.rabbitmq_api_url}/api/queues/%2F/{queue_name}",
        auth=httpx.BasicAuth(settings.rabbitmq_user, settings.rabbitmq_password),
    )
    payload = object_json(response)
    count = payload.get("messages")
    if not isinstance(count, int):
        raise AssertionError(f"RabbitMQ did not return a message count: {payload}")
    return count


async def publish_poison_message(client: httpx.AsyncClient, settings: Settings, index: int) -> None:
    response = await client.post(
        f"{settings.rabbitmq_api_url}/api/exchanges/%2F/amq.default/publish",
        auth=httpx.BasicAuth(settings.rabbitmq_user, settings.rabbitmq_password),
        json={
            "properties": {"delivery_mode": 2, "content_type": "application/json"},
            "routing_key": "inference.requests",
            "payload": f"not-json-{index}",
            "payload_encoding": "string",
        },
    )
    payload = object_json(response)
    if payload.get("routed") is not True:
        raise AssertionError(f"Poison message was not routed: {payload}")


async def wait_for_terminal_statuses(
    client: httpx.AsyncClient,
    settings: Settings,
    task_ids: set[UUID],
) -> dict[UUID, str]:
    unresolved = set(task_ids)
    results: dict[UUID, str] = {}
    deadline = time.monotonic() + settings.timeout_seconds
    while unresolved and time.monotonic() < deadline:
        responses = await asyncio.gather(
            *(client.get(f"{settings.gateway_url}/status/{task_id}") for task_id in unresolved)
        )
        for task_id, response in zip(tuple(unresolved), responses, strict=True):
            payload = object_json(response)
            task_status = payload.get("status")
            if task_status in TERMINAL_STATUSES:
                results[task_id] = cast(str, task_status)
        unresolved.difference_update(results)
        if unresolved:
            await asyncio.sleep(0.5)
    if unresolved:
        raise TimeoutError(f"Tasks did not resolve: {sorted(map(str, unresolved))}")
    return results


async def wait_for_dlq(client: httpx.AsyncClient, settings: Settings, minimum_messages: int) -> int:
    deadline = time.monotonic() + settings.timeout_seconds
    while time.monotonic() < deadline:
        messages = await queue_message_count(client, settings, "inference.requests.dlq")
        if messages >= minimum_messages:
            return messages
        await asyncio.sleep(0.5)
    raise TimeoutError(f"DLQ did not reach {minimum_messages} messages")


async def main() -> None:
    settings = Settings.from_environment()
    timeout = httpx.Timeout(settings.timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=100)) as client:
        dlq_before = await queue_message_count(client, settings, "inference.requests.dlq")
        task_ids = set(await asyncio.gather(
            *(submit_prediction(client, settings.gateway_url, index) for index in range(VALID_REQUESTS))
        ))
        if len(task_ids) != VALID_REQUESTS:
            raise AssertionError("Gateway returned duplicate task IDs")
        await asyncio.gather(
            *(publish_poison_message(client, settings, index) for index in range(POISON_REQUESTS))
        )
        statuses = await wait_for_terminal_statuses(client, settings, task_ids)
        dlq_after = await wait_for_dlq(client, settings, dlq_before + POISON_REQUESTS)
    print(f"resolved={len(statuses)} completed={sum(value == 'completed' for value in statuses.values())} failed={sum(value == 'failed' for value in statuses.values())} dlq_messages={dlq_after}")


if __name__ == "__main__":
    asyncio.run(main())
