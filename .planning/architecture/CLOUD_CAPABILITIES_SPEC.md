# Cloud Deployment Capabilities + Startup Spec (Azure-Oriented)

This spec defines how the system should detect runtime resources (CPU/RAM/GPU),
derive safe defaults (threads/concurrency/queue sizes), and enforce readiness
before accepting work.

Context: the system is expected to run on deep cloud resources (likely Azure)
and sometimes on constrained laptops. The same container images should work in
both environments.

## Goals

- Avoid unstable behavior under scale (e.g. GROBID restarts, OOM kills).
- Use available resources automatically (CPU cores, memory, GPU/VRAM) without
  requiring manual tuning for every deployment.
- Fail fast when a required capability is missing; otherwise degrade
  gracefully and explicitly.
- Make startup and readiness falsifiable and observable.

## Non-Goals

- Kubernetes-only implementation. The spec must work with Compose and extend to
  AKS.
- Auto-provisioning cloud infra.
- Perfect hardware introspection across all environments. Prefer a simple,
  robust "best effort" capability map.

## Terms

- Capability map: computed facts about the runtime (cpu_cores, mem_bytes,
  gpu_count, gpu_vram_bytes, etc).
- Derived limits: computed configuration (max concurrent extract jobs, thread
  pool sizes, queue sizes).
- Readiness: "safe to accept user work" check; must include dependency checks
  and minimum capabilities.

## Capability Detection

### CPU

- Detect the effective CPU quota, preferring container/cgroup limits.
- Output:
  - cpu_cores_effective: int
  - cpu_cores_physical: int (best effort)

Implementation notes:
- In containers, prefer cgroup CPU quota when present.
- Fallback to `os.cpu_count()`.

### Memory

- Detect effective memory limit, preferring container/cgroup limits.
- Output:
  - mem_bytes_effective: int

### GPU

GPU use is optional unless explicitly enabled.

- Detect CUDA availability.
- Enumerate devices.
- Output per device:
  - gpu_name
  - gpu_vram_bytes_total
  - gpu_compute_capability (best effort)

GPU should be treated as "present" only when:
- devices > 0 AND
- driver/runtime is usable (simple probe) AND
- VRAM total is readable.

### Disk + Temp

- Ensure a writable temp directory exists and has sufficient free space for:
  - worst-case PDF temp files
  - intermediate images (OCR path)

### Networked Dependencies

Dependencies are treated as capabilities for readiness.

- Postgres reachable and migrations applied.
- Object store reachable and bucket verified.
- GROBID reachable and stable.
- Optional model services reachable (embedding/LLM/ColBERT gateway) depending on
  configured mode.

## Derived Limits (Autotuning)

All derived limits must be overridable via env vars.

### Principles

- Prefer fewer concurrent GROBID requests over maximizing throughput.
- Bound concurrency by memory, not only CPU.
- Keep queues bounded; reject/enqueue-fail rather than building infinite
  backlogs.

### Recommended Derived Settings

Inputs:
- cpu_cores_effective
- mem_bytes_effective
- gpu_count / gpu_vram_bytes_total (optional)

Outputs (examples):
- EXTRACT_POOL_WORKERS (default: min(4, cpu_cores_effective))
- EXTRACT_POOL_QUEUE_MAX (default: 4 * EXTRACT_POOL_WORKERS)
- GROBID_CONCURRENCY (default: 1..2; never > 4 unless explicitly set)
- OCR_CONCURRENCY (default: min(2, cpu_cores_effective/2))
- TORCH_NUM_THREADS, OMP_NUM_THREADS, MKL_NUM_THREADS (bounded)

Memory heuristics:
- Reserve a fixed headroom fraction for OS + caches (e.g. 25%).
- Estimate per-job memory budgets:
  - grobid_job_mem_bytes (configurable; conservative default)
  - ocr_job_mem_bytes (configurable)
- Compute max concurrency as:
  floor((mem_effective * (1 - headroom)) / per_job_mem)

## Startup Behavior

Startup should build and log:

- capability_map (JSON)
- derived_limits (JSON)
- selected execution profile (e.g. "cpu-small", "cpu-large", "gpu")

### Startup Sequence

1) Load config.
2) Compute capability map.
3) Compute derived limits.
4) Apply derived limits to runtime:
   - thread env vars
   - worker pool sizes
   - queue sizes
5) Run dependency checks (with retry window):
   - Postgres migrations
   - object store bucket
   - GROBID readiness
   - model service readiness if required
6) Mark service ready.

### Degrade vs Fail

Rules:

- If GPU is required by config and no GPU is usable: fail startup.
- If GPU is optional and missing: log degrade to CPU; continue.
- If GROBID is required and unreachable: fail readiness (service runs, but is
  not ready).
- If object store/Postgres unreachable: fail readiness.

## Readiness / Liveness

Add two HTTP endpoints:

- `GET /health/live`
  - returns 200 if process is running (no deep checks)
- `GET /health/ready`
  - returns 200 only if:
    - migrations applied
    - object store reachable + bucket ok
    - GROBID reachable
    - worker pools initialized
    - optional model services reachable if configured
  - returns 503 with a machine-readable payload listing failed checks

Readiness should be used by cloud load balancers and orchestrators.

## Worker Pools + Parallelization

### Separation

Maintain separate pools for:

- extraction (GROBID + fallback)
- OCR (if enabled)
- embeddings/rerank (GPU or CPU)

### Backpressure

- Each pool has a bounded queue.
- Enqueue failure returns a clear 503 with a retry-after hint.
- Jobs record "queue_full" failure reasons.

### Retry Policy

Define retryable failures and retry windows:

- Retryable:
  - connection reset/refused to GROBID
  - transient 5xx from GROBID
  - object store transient timeouts
- Non-retryable (default):
  - encrypted PDF
  - permanent parse error
  - settings validation error

## Azure Deployment Notes

This spec is Azure-oriented but should not hardcode Azure SDK usage.

### AKS

- Prefer node pools:
  - general CPU pool (api, postgres client, non-ML workers)
  - GPU pool (embedding/LLM workers) if needed
- Use node selectors/taints to place GPU workloads correctly.

### Storage

- Prefer managed Postgres (Flexible Server) for production.
- Object store:
  - S3-compatible (MinIO in dev)
  - In Azure, either:
    - MinIO in AKS with persistent volumes, or
    - an S3-compatible gateway in front of Azure Blob, if acceptable

### Identity

- For production, avoid embedding secrets in env when possible.
- If using Azure-native services, use workload identity or managed identity.

## Observability

- Log capability_map + derived_limits on every boot.
- Include attempt_id/job_id correlation IDs in logs.
- Emit a periodic heartbeat per worker with:
  - in-flight jobs
  - queue depth
  - recent error counts

## Verification Plan

- Constrained env (1 core, low RAM): service starts, derived limits are small,
  readiness passes, ingestion works slowly.
- Deep CPU env (many cores): service starts, derived limits scale but GROBID
  concurrency remains bounded.
- GPU env: GPU detected; GPU-only features enabled; readiness fails if GPU
  required but missing.
- Dependency failure:
  - Postgres down: /health/ready is 503
  - GROBID down: /health/ready is 503; /health/live is 200

## Roadmap Integration

- This spec is intended to land after Phase 9.2 as part of "cloud readiness"
  hardening and worker distribution.
