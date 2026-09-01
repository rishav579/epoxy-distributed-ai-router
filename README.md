# Epoxy — Distributed Semantic Inference Router

Epoxy is an asynchronous inference platform designed to reduce LLM operating cost
without sacrificing a path to frontier-model quality. A lightweight local model
handles straightforward requests; only ambiguous or complex requests need to be
forwarded to a frontier LLM. The result is a cost-aware routing boundary that can be
measured, scaled, and operated like a production service.

> **Current implementation:** the repository provides the local PyTorch/LoRA
> classifier, durable inference pipeline, benchmark tooling, and deployment
> foundations. Frontier-LLM forwarding is the next policy/integration layer.

## Why Epoxy?

Most production traffic is repetitive, bounded, and inexpensive to classify. Sending
every request to a premium model creates unnecessary latency and token spend. Epoxy
separates the routing decision from execution:

- **Simple** requests can run on a local small language model (SLM).
- **Complex** requests can be escalated to a frontier LLM when the policy requires
  deeper reasoning or broader context.
- Every request receives a durable task ID, observable status, and recoverable
  failure path.

## Architecture

```text
Client
  │ POST /predict
  ▼
FastAPI Gateway ──► PostgreSQL (task state)
  │
  └───────────────► RabbitMQ (durable work queue + DLQ)
                           │
                           ▼
                    PyTorch/LoRA Worker
                           │
                           └──► PostgreSQL (result + status)
```

The gateway acknowledges an HTTP request after task creation and durable queue
publication. Workers consume with manual ACKs and `prefetch_count=1`, run local
inference, commit the result transactionally, and ACK only after the database commit.
Failures are negatively acknowledged to `inference.requests.dlq`; an unacknowledged
message can be redelivered if a worker exits unexpectedly.

## Tech stack

- **API:** FastAPI, Uvicorn, `aio-pika`, `asyncpg`
- **Messaging:** RabbitMQ with durable queues and dead-letter exchange
- **Persistence:** PostgreSQL
- **ML:** PyTorch, Hugging Face Transformers, PEFT/LoRA
- **Packaging:** Docker multi-stage images, Docker Compose
- **Cloud:** AWS EKS, RDS PostgreSQL, private S3 model artifacts, IRSA
- **Infrastructure:** Terraform with private VPC subnets and queue-depth HPA
- **Quality:** Python E2E load/failure test, benchmark reports, optional Weights & Biases

## Local setup (Windows)

### Prerequisites

- Windows 10/11 with PowerShell
- Docker Desktop with Linux containers and Compose v2
- Python 3.11+ (the script can create `.venv` using the `py` launcher)

### Run the complete local pipeline

From the project directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_local_e2e.ps1
```

The script validates Docker, starts PostgreSQL and RabbitMQ, installs all dependency
files, verifies `outputs/adapter`, and runs a fast one-epoch mock LoRA training job
when the adapter is absent. It then starts the gateway and worker, polls their real
readiness endpoints, runs `test_e2e_pipeline.py`, and prints database/queue status.

For a local preview after the test:

```powershell
.\run_local_e2e.ps1 -KeepLocalProcesses
```

To stop the Compose services when finished:

```powershell
.\run_local_e2e.ps1 -TearDownCompose
```

### Local endpoints

| Component | URL |
| --- | --- |
| Gateway Swagger UI | http://127.0.0.1:8000/docs |
| Gateway metrics | http://127.0.0.1:8000/metrics |
| Worker metrics | http://127.0.0.1:8001/metrics |
| RabbitMQ management | http://127.0.0.1:15672 |

Submit a task:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/predict `
  -Method Post -ContentType 'application/json' `
  -Body '{"text":"Classify this support request"}'
```

Use the returned `task_id` with `GET /status/{task_id}`.

## Benchmarking

After dependencies and a local adapter are available:

```powershell
python benchmark_router.py
```

This creates or loads 500 labelled prompts, reports precision/recall/F1 and a
confusion matrix, measures P50/P95/P99 latency at batch sizes 1/8/32, projects
dynamic-routing savings, and writes `benchmark_report.md`. Set `WANDB_API_KEY` to
log metrics to Weights & Biases.

## Cloud deployment

Terraform modules are under `deploy/terraform` and define a private-by-default AWS
foundation: multi-AZ VPC, private RDS, private S3 model artifacts, private-endpoint
EKS, and an IRSA role scoped to the worker service account. Kubernetes manifests are
under `deploy/kubernetes`; the worker HPA scales on
`rabbitmq_queue_messages_ready` rather than CPU alone.

The GitHub Actions release workflow at
`.github/workflows/production-deploy.yml` runs E2E tests, builds immutable gateway
and worker images, pushes them to ECR, and deploys the manifests to EKS using GitHub
OIDC—no long-lived AWS access keys.

## Repository guide

```text
inference_gateway.py             FastAPI request/status API
inference_worker.py              RabbitMQ consumer and LoRA inference
train_lora_classifier.py         Local adapter training
test_e2e_pipeline.py             Concurrent success/DLQ validation
benchmark_router.py              Quality, latency, and cost benchmark
run_local_e2e.ps1                 Windows end-to-end orchestration
deploy/kubernetes/                EKS Deployments and queue-depth HPA
deploy/terraform/                 AWS infrastructure modules
```

## Reliability principles

- At-least-once delivery with explicit ACK/NACK boundaries
- Transactional result persistence and task UUID idempotency
- Durable RabbitMQ queues plus a dead-letter queue for poison messages
- Independent gateway and worker scaling
- Private data-plane resources and short-lived workload credentials

## License

Add the project license before publishing a production distribution.
