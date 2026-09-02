# Epoxy — Distributed Semantic Inference Router

Epoxy is an asynchronous inference platform designed to reduce LLM operating cost
without sacrificing a path to frontier-model quality. A lightweight local model
handles straightforward requests; only ambiguous or complex requests need to be
forwarded to a frontier LLM. The result is a cost-aware routing boundary that can be
measured, scaled, and operated like a production service.

> **Current implementation:** Epoxy implements a two-tier inference routing architecture:
> 1. **Complexity Classifier:** Fine-tuned DistilBERT LoRA sequence classifier categorizing requests into simple (0) vs complex (1).
> 2. **Confidence & Ambiguity Routing Policy:** High-confidence simple requests route locally; high-confidence complex requests route to the frontier tier; ambiguous requests (confidence below configured threshold, default `0.75`) escalate to the frontier tier for safety.
> 3. **Execution Boundaries:**
>    - **Local path:** `LocalExecutor` abstraction providing structured responses for the local SLM tier.
>    - **Frontier path:** `FrontierExecutor` supporting pluggable adapters (`MockFrontierProvider` for offline testing and deterministic CI, and `HttpFrontierProvider` for OpenAI/generic HTTP endpoints with `FRONTIER_API_KEY`).
> 4. **Durable Persistence:** PostgreSQL records task status, classifier probabilities, routing decisions, confidence, and execution output before AMQP ACK.

## Why Epoxy?

Most production traffic is repetitive, bounded, and inexpensive to classify. Sending
every request to a premium model creates unnecessary latency and token spend. Epoxy
separates the routing decision from execution:

- **Simple** requests run on a local small language model (SLM) path.
- **Complex** requests escalate to a frontier LLM when deep reasoning or synthesis is needed.
- **Ambiguous** requests escalate to the frontier LLM by policy to prevent degraded answers.
- Every request receives a durable task ID, observable status, full probability distribution, and recoverable failure path.

## Architecture

```text
Client
  │ POST /predict
  ▼
FastAPI Gateway ──► PostgreSQL (task state: queued)
  │
  └───────────────► RabbitMQ (durable queue + DLQ)
                           │
                           ▼
                    Inference Worker
                           │
                 [Complexity Classifier]
                     (label, probs)
                           │
                 [Routing Policy Engine]
              (threshold, confidence check)
               ┌───────────┴───────────┐
               ▼                       ▼
          route: local            route: frontier
         (LocalExecutor)      (FrontierExecutor)
               │                       │
               └───────────┬───────────┘
                           ▼
                 PostgreSQL Transaction
             (results: route, confidence,
              execution_result, status)
                           │
                  RabbitMQ Manual ACK
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

Terraform modules are under `deploy/terraform` and define the AWS infrastructure foundation:
- **Multi-AZ VPC:** Public and private subnets across 3 AZs with dedicated NAT gateways.
- **Private Data Plane:** RDS PostgreSQL 16 Multi-AZ instance on private subnets, accessible only by EKS worker nodes.
- **Private S3 Model Store:** Encrypted bucket (AES256, HTTPS enforced, public access blocked) for model weights and LoRA adapters.
- **EKS Cluster:** Kubernetes 1.31 cluster with worker nodes in private subnets, IRSA enabled, and both private and public API endpoint access enabled (`cluster_endpoint_private_access = true`, `cluster_endpoint_public_access = true`). Public endpoint access is required for external GitHub-hosted Actions runners to authenticate and deploy manifests to the control plane, while compute nodes remain isolated on private subnets.
- **GitHub Actions OIDC:** OpenID Connect provider and IAM role with trust policy strictly scoped to `repo:rishav579/epoxy-distributed-ai-router` (targeting `refs/heads/main` and `environment:production`), eliminating static AWS keys.

### Kubernetes deployment dependencies

The Kubernetes manifests under `deploy/kubernetes` decouple the stateless workload from environment-specific backing infrastructure via standard Kubernetes abstractions:
- **`inference-secrets` (Secret):** Must be provisioned in the cluster namespace to supply `database-url` (RDS connection string) and `amqp-url` (RabbitMQ broker connection string).
- **`inference-model-registry` (PersistentVolumeClaim):** Mounted at `/models` by worker pods to access model weights synchronized from S3 or shared storage.
- **Prometheus Adapter / KEDA:** Required by `deploy/kubernetes/worker-hpa.yaml` to feed the `rabbitmq_queue_messages_ready` external metric into the Kubernetes Custom Metrics API for queue-depth autoscaling.

### Kubernetes security controls

Rather than claiming general "CIS-compliant" status, the manifests implement specific, concrete Kubernetes security controls:
- **Pod security context:** Workloads run as non-root (`runAsNonRoot: true`, `runAsUser: 10001`), prohibit privilege escalation (`allowPrivilegeEscalation: false`), enforce a read-only root filesystem (`readOnlyRootFilesystem: true`), and drop all Linux capabilities (`capabilities: drop: ["ALL"]`).
- **Kernel sandboxing:** Workloads enforce the `RuntimeDefault` seccomp profile (`seccompProfile: type: RuntimeDefault`).
- **Credential & token hygiene:** API token automounting is disabled (`automountServiceAccountToken: false`), and sensitive credentials (PostgreSQL and RabbitMQ connection strings) are injected exclusively via Kubernetes Secrets (`inference-secrets`). Model weights are mounted read-only (`readOnly: true`).

## Current architectural status and limitations

- **Complexity Classifier vs. Generator:** The local DistilBERT LoRA model acts strictly as a sequence classification router (predicting whether a task is simple or complex). It is not a generative natural language answer model.
- **Local Execution Boundary:** `LocalExecutor` serves as an adapter prototype for where a quantized local generative SLM (e.g. Gemma, Mistral, or Llama) connects in a subsequent phase.
- **Frontier Provider Boundary:** `FrontierExecutor` supports pluggable adapters. Automated test suites and local E2E pipelines use `MockFrontierProvider` for deterministic offline verification without external secrets. The `HttpFrontierProvider` adapter exists for OpenAI-compatible endpoints, but live production API execution has not been live-tested with active credentials in this phase.
- **Cloud Deployment Status:** Cloud infrastructure configurations (Terraform, EKS, RDS, IRSA, and GitHub Actions OIDC) exist as code and validate cleanly, but live cloud provisioning is blocked because the target AWS account is currently on the Free Plan with S3 unactivated (`NotSignedUp`). No paid resources or fabricated deployments have been created.

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
