# Distributed Semantic Inference Router

## Purpose and system boundary

This service accepts prediction requests without making the client wait for model
execution. The FastAPI gateway durably records a task and publishes a small AMQP
message. A separately scalable worker loads a versioned Hugging Face PEFT/LoRA
adapter, performs PyTorch inference, and commits the result to PostgreSQL. RabbitMQ
is the work buffer and dead-letter transport; PostgreSQL is the source of truth for
task state and results.

The local deployment uses PostgreSQL 16 and RabbitMQ 3.13. Kubernetes manifests
deploy the gateway and worker separately. The model registry is mounted read-only by
workers, so model rollout is an artifact/versioning concern rather than a container
image rebuild concern.

## Request ingestion

```text
                                     +------------------+
                                     |   PostgreSQL     |
                                     | inference_tasks  |
Client --HTTP POST /predict--------> | (status=queued)  |
  ^                                  +--------^---------+
  |                                           |
  | 202 {task_id,status=queued}               | status lookup
  |                                           |
  +------------------ FastAPI gateway <-------+
                         |
                         | durable publish: task_id + text
                         v
                 +-------------------+
                 | RabbitMQ exchange |
                 | / queue           |
                 | inference.requests|
                 +-------------------+
```

The gateway creates the task row before publishing. `POST /predict` returns `202`
only after the publish succeeds. `GET /status/{task_id}` reads PostgreSQL, keeping
the queue private and making status queries independent of worker placement.

## Consumer, routing, and execution boundary

```text
 RabbitMQ durable queue
 inference.requests
          |
          | manual delivery (prefetch=1)
          v
 +-------------------+       +--------------------+
 | inference_worker  | ----> | PyTorch + PEFT     |
 | parse message     |       | LoRA classifier    |
 | mark processing   |       | predict(text)      |
 +---------+---------+       +---------+----------+
           |                           | (label, probs)
           |                           v
           |                 +--------------------+
           |                 | RoutingPolicy      |
           |                 | (confidence check) |
           |                 +----+----------+----+
           |                      |          |
           |       route: local   |          | route: frontier
           |                      v          v
           |               +----------+  +-------------+
           |               | Local    |  | Frontier    |
           |               | Executor |  | Executor    |
           |               +----+-----+  +-----+-------+
           |                    |              |
           |                    +-------+------+
           |                            |
           | success                    v (exec_result)
           v                 +--------------------+
 +-------------------+       | Exception          |
 | PostgreSQL txn    |       | (model/exec/DB)    |
 | result + completed|       +---------+----------+
 +---------+---------+                 |
           |                           v
           | commit, then ACK      message.nack (requeue=False)
           +-------------------->  RabbitMQ DLX
                                  -> inference.requests.dlq
```

The worker separates complexity classification from response execution:
1. **Classification:** Evaluates input text with DistilBERT LoRA to compute `[p_simple, p_complex]`.
2. **Routing Decision:**
   - If `confidence >= threshold` (default `0.75`) and `label == 0`, routes to `local` with reason `high_confidence_simple`.
   - If `confidence >= threshold` and `label == 1`, routes to `frontier` with reason `high_confidence_complex`.
   - If `confidence < threshold`, escalates to `frontier` with reason `ambiguous_confidence`.
3. **Execution Boundary:**
   - `LocalExecutor`: Simulates or dispatches to a local SLM response model.
   - `FrontierExecutor`: Dispatches to `MockFrontierProvider` (for deterministic offline testing) or `HttpFrontierProvider` (for OpenAI-compatible endpoints with `FRONTIER_API_KEY`).
4. **Failure & Durability:** The worker records all outputs (`route`, `confidence`, `probabilities`, `execution_result`, `status`) to PostgreSQL inside an atomic transaction. A message is positively acknowledged (`message.ack()`) ONLY after the database transaction commits. Any unhandled exception causes task failure recording and negative acknowledgment (`message.nack(requeue=False)`) to the dead-letter exchange.

## Kubernetes autoscaling loop

```text
 +------------------+   scrape   +-------------+   expose external   +-----+
 | RabbitMQ         | ----------> | Prometheus  | ------------------> | HPA |
 | queue messages   |             | metric     |  Prometheus Adapter |     |
 | ready            |             |             |                     +--+--+
 +------------------+             +-------------+                        |
        ^                                                               |
        | publish / consume                                             | desired replicas
        |                                                               v
 +------+----------------+       scale Deployment       +----------------------+
 | inference.requests   | <----------------------------- | inference-worker    |
 | queue depth          |                                | worker pods         |
 +----------------------+                                +----------------------+
```

`worker-hpa.yaml` uses the external metric
`rabbitmq_queue_messages_ready`, selects queue `inference.requests`, and targets
an average of 10 ready messages per worker. It scales from 1 to 20 replicas, scales
up immediately, and uses a slower 300-second scale-down stabilization window to
avoid oscillation. Each worker has `prefetch_count=1`, so a pod receives at most one
in-flight message and queue depth remains a useful measure of unmet work.

Prometheus scraping of the worker's `/metrics` endpoint and the Prometheus Adapter's
external-metric API are cluster prerequisites; they are intentionally separate from
the application Deployments.

## Engineering decisions: why

### Why `aio-pika` pooling and manual ACKs instead of Celery/Redis tasks?

The workload already has RabbitMQ as its durable broker, so introducing Celery and
Redis would add another broker/client lifecycle, serialization conventions, and
operational failure domain without improving delivery semantics. `aio-pika` exposes
the AMQP primitives this service needs directly:

- The gateway's `aio_pika.pool.Pool` bounds and reuses robust AMQP connections while
  allowing channels to be short-lived and isolated per publish.
- Durable queues, persistent messages, a dead-letter exchange, and explicit routing
  keys are visible in the application contract and inspectable with RabbitMQ tools.
- Manual `ack()` after the PostgreSQL commit gives an exact commit/ack boundary.
  `nack(requeue=False)` sends poison messages and terminal processing failures to the
  DLQ rather than silently dropping them.
- `prefetch_count=1` supplies broker-level backpressure for CPU/GPU-bound inference.

Celery is a sound choice for a broad task platform, but its worker protocol and
result-backend conventions would obscure these deliberately small, auditable
semantics. Redis is not needed for either the queue or the durable task state here.

### Why decouple the API gateway from model inference?

Torch model loading is memory-heavy, can take seconds, and has variable execution
time. Running it inside FastAPI would couple request concurrency to model memory and
would let a slow/blocked inference consume HTTP worker capacity. It would also make
rolling model versions and GPU/CPU placement harder.

The gateway therefore stays stateless apart from the database transaction and can
scale for HTTP throughput. Workers can be independently sized for accelerator
memory, scaled from queue demand, restarted after model faults, and rolled out with
a new adapter. RabbitMQ absorbs bursts and provides explicit delivery durability.
The API's quick `202` response also gives callers a stable task-id/status contract.

### Why queue length rather than CPU or memory for worker HPA?

CPU and memory describe resource consumption, not demand. A worker can be CPU-bound
while keeping up, or have high resident model memory while the queue is empty. The
business SLO is queue waiting time; `rabbitmq_queue_messages_ready` measures work
that has not yet been assigned to a consumer. With prefetch one, it is a close
signal for backlog per pod and therefore maps directly to the number of workers
needed to drain the queue.

CPU/memory remain useful resource requests, limits, and saturation alerts. They are
poor primary scaling signals for this asynchronous, model-loaded workload. The HPA
target of 10 ready messages per worker, rapid scale-up, and delayed scale-down trade
reaction speed for stability.

### How are database outages handled while messages are in flight?

Database writes are deliberately before the ACK boundary:

1. The gateway must create the task row before it returns `202`. If that insert
   fails, no message is published and the client receives an error. If publishing
   fails after insertion, the gateway attempts to mark the task `failed` and returns
   `503`; the row remains an explicit operational signal if the compensating update
   itself cannot reach PostgreSQL.
2. The worker marks a task `processing`, runs inference, and writes the result plus
   `completed` status transactionally. If PostgreSQL is unavailable at either write,
   the handler catches the exception and issues `nack(requeue=False)`, preserving the
   message in the DLQ instead of acknowledging it.
3. If the worker process or connection disappears before an ACK, RabbitMQ detects the
   closed consumer and redelivers the unacknowledged message. This protects work
   during transient worker failures. If the database was reachable for
   `completed` but the ACK was lost, the result upsert and task state update are
   idempotent, so a redelivery does not create a second result.

This is at-least-once delivery, not exactly-once execution. The task UUID primary
key, result upsert, and post-commit ACK make duplicate delivery safe. A sustained
database outage intentionally accumulates operational evidence in the DLQ rather
than hot-looping retries; operators can restore the database and replay DLQ messages
according to their incident policy.

## Operational and security posture

Rather than relying on generic "CIS-compliant" labels, the platform documents and implements specific, auditable Kubernetes security controls:
- **Workload security contexts:** Container workloads run as non-root (`runAsNonRoot: true`, `runAsUser: 10001`), prohibit privilege escalation (`allowPrivilegeEscalation: false`), enforce a read-only root filesystem (`readOnlyRootFilesystem: true`), drop all Linux capabilities (`capabilities: drop: ["ALL"]`), and enforce the `RuntimeDefault` seccomp profile.
- **Credential & token isolation:** API credential automounting is disabled (`automountServiceAccountToken: false`), and application secrets are injected via Kubernetes `Secret` resources (`inference-secrets`). Worker model weights are mounted read-only (`readOnly: true`).
- **Cluster network architecture:** EKS worker nodes are placed in private subnets with `cluster_endpoint_private_access = true`. Public endpoint access (`cluster_endpoint_public_access = true`) is enabled because external GitHub-hosted Actions runners require API access to deploy manifests to the control plane, while worker compute nodes remain non-routable from the public internet.
- **Probes and lifecycle:** PostgreSQL and RabbitMQ have container health checks locally. Kubernetes uses TCP readiness/liveness/startup probes and a 90-second worker termination grace period.
- **Observability:** Gateway and worker expose Prometheus metrics. Worker counters include completed and DLQ-failed messages; queue depth is the autoscaling metric.
- **Concurrency discipline:** The current worker uses one message at a time intentionally (`prefetch_count=1`). Increase concurrency only after measuring model memory, database pool capacity, and GPU contention; otherwise queue-based autoscaling loses its predictable work-per-pod behavior.

## Trade-offs and limits

The design favors explicitness and recoverability over a generalized workflow
engine. There is no automatic retry budget/backoff for database failures; the DLQ is
the failure boundary and needs replay tooling/alerting in production. PostgreSQL is
also a dependency of both ingestion and completion, so its availability determines
end-to-end durability even though RabbitMQ can continue buffering messages.
Finally, external-metric scaling depends on correct Prometheus labels and adapter
configuration; without that control-plane path, the worker Deployment remains at
its configured replica count rather than silently scaling on a less relevant signal.
