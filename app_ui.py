"""Minimal Streamlit frontend for the asynchronous inference gateway."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import requests
import streamlit as st


DEFAULT_GATEWAY_URL = "http://127.0.0.1:8000"
POLL_INTERVAL_SECONDS = 0.5
POLL_TIMEOUT_SECONDS = 120.0
TERMINAL_STATUSES = {"completed", "failed"}


def request_json(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    response = requests.request(method, url, **kwargs)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Gateway returned non-object JSON from {url}")
    return payload


def extract_output(payload: dict[str, Any]) -> Any | None:
    """Support current/future gateway response shapes without assuming one schema."""
    for key in ("result", "output", "prediction", "inference_result"):
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def poll_task(gateway_url: str, task_id: str, status_box: Any) -> dict[str, Any]:
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    status_url = f"{gateway_url}/status/{task_id}"
    last_status = "queued"
    while time.monotonic() < deadline:
        payload = request_json("GET", status_url, timeout=10)
        status = str(payload.get("status", "unknown"))
        last_status = status
        status_box.write(f"Task `{task_id}` — **{status}**")
        if status in TERMINAL_STATUSES:
            return payload
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"Task {task_id} did not reach a terminal state within {POLL_TIMEOUT_SECONDS:.0f} seconds "
        f"(last status: {last_status})."
    )


def render_result(payload: dict[str, Any]) -> None:
    status = payload.get("status", "unknown")
    if status == "failed":
        st.error(payload.get("error") or "The worker failed this inference task.")
        return

    st.success(f"Inference completed — status: {status}")
    output = extract_output(payload)
    if output is not None:
        st.subheader("Model output")
        if isinstance(output, (dict, list)):
            st.json(output)
        else:
            st.write(output)
    else:
        st.info(
            "The task completed, but this gateway response contains no result payload. "
            "Expose the persisted inference result from `/status/{task_id}` to render model output here."
        )
    with st.expander("Task details"):
        st.json(payload)


def main() -> None:
    st.set_page_config(page_title="Semantic Inference Router", page_icon="🧠", layout="centered")
    st.title("Distributed Semantic Inference Router")
    st.caption("Submit a request to the local FastAPI gateway and monitor the background worker.")

    with st.sidebar:
        st.header("Connection")
        gateway_url = st.text_input("Gateway URL", value=DEFAULT_GATEWAY_URL).strip().rstrip("/")
        st.caption("The gateway must be running before submitting an inference request.")

    prompt = st.text_area(
        "Input text",
        height=220,
        placeholder="Paste a medical note or other text to classify…",
        help="The request is queued for asynchronous LoRA model inference.",
    )
    run_inference = st.button("Run Inference", type="primary", use_container_width=True)

    if not run_inference:
        return
    if not gateway_url:
        st.error("Enter a gateway URL before running inference.")
        return
    if not prompt.strip():
        st.warning("Enter some text before running inference.")
        return

    try:
        with st.status("Submitting inference request…", expanded=True) as operation:
            response = request_json(
                "POST",
                f"{gateway_url}/predict",
                json={"text": prompt},
                timeout=15,
            )
            task_id = response.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError("Gateway response did not contain a valid task_id.")
            st.write(f"Queued task `{task_id}`")
            result = poll_task(gateway_url, task_id, st.empty())
            operation.update(label="Inference finished", state="complete", expanded=False)
        render_result(result)
    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not connect to {gateway_url}. Start the local gateway and worker, then try again."
        )
    except requests.exceptions.Timeout:
        st.error("The gateway request timed out. Check gateway/worker health and try again.")
    except requests.exceptions.HTTPError as error:
        detail = error.response.text[:500] if error.response is not None else str(error)
        st.error(f"Gateway returned an HTTP error: {detail}")
    except TimeoutError as error:
        st.warning(str(error))
    except (ValueError, requests.exceptions.RequestException) as error:
        st.error(f"Inference request failed: {error}")

    st.caption(f"Checked at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")


if __name__ == "__main__":
    main()
