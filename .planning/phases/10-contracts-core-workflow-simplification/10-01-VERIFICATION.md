---
phase: 10-contracts-core-workflow-simplification
verified: 2026-02-21T08:52:05Z
status: passed
score: 5/5 must-haves verified
---

# Phase 10-01 Verification Report

**Phase Goal:** Stage boundaries have an explicit, versioned JSON envelope contract per stage; artifacts are persisted write-once per (run_id, stage) to works bucket; API stores/fetches JSON without clients touching S3; `/dev/wipe` removes pipeline run rows and artifacts.
**Verified:** 2026-02-21T08:52:05Z
**Status:** passed
**Re-verification:** No — initial verification

## Must-Haves Checklist (With Evidence)

- [x] Each pipeline stage artifact is a versioned JSON envelope (`schema_version` + `artifact_type` + `status` + `warnings`/`error`).
  - Evidence: `backend/contracts/pipeline_v1.py` defines shared `PipelineArtifactEnvelope` with `schema_version`, `artifact_type`, `status`, `warnings`, optional `error` and validates `artifact_type == contracts/{stage}@v{schema_version}` via `@model_validator`.
  - Evidence: `backend/contracts/pipeline_v1.py` validates and normalizes via `validate_v1()` selecting per-stage models (`ExtractArtifact`, `CiteSpansArtifact`, `RetrievalArtifact`, `FilterArtifact`, `RerankArtifact`, `NliArtifact`).

- [x] Stage artifacts persist in the works bucket under `pipeline/{run_id}/{stage}.json` and are immutable per (run_id, stage): second store attempt is rejected with HTTP 409 (no S3 overwrite).
  - Evidence (key format): `backend/contracts/pipeline_v1.py` `stage_object_key(run_id, stage) -> "pipeline/{rid}/{st}.json"`.
  - Evidence (write-once reserve): `backend/spine/pipeline_artifacts.py` `put_stage_payload()` does `INSERT ... ON CONFLICT (run_id, stage) DO NOTHING RETURNING artifact_id`; if no row returned raises `StageArtifactAlreadyExists` *before* calling `object_store_s3.put_bytes()`.
  - Evidence (409 mapping): `backend/main.py` `put_pipeline_stage()` catches `StageArtifactAlreadyExists` and returns `JSONResponse(status_code=409, content={"error":{...}})`.
  - Evidence (no overwrite): `tests/test_pipeline_contract_store.py` `test_put_and_get_stage_payload_is_write_once()` asserts second put raises and `object_store_s3.get_bytes(key)` remains unchanged.
  - Evidence (API-level immutability): `tests/test_pipeline_contract_api.py` asserts second PUT returns 409 and subsequent GET returns the original payload.

- [x] API can store and fetch stage artifacts as JSON bodies (clients never fetch S3 directly).
  - Evidence: `backend/main.py` implements `POST /pipeline/runs`, `PUT /pipeline/runs/{run_id}/stages/{stage}`, `GET /pipeline/runs/{run_id}/stages/{stage}` and returns only the validated envelope dict.
  - Evidence (no S3 pointer in responses): `backend/spine/pipeline_artifacts.py` returns `dict(validated)` from `put_stage_payload()`; the contract models do not include `object_key`, and routes return this dict verbatim.

- [x] Contracts carry stable IDs/aliases needed for later stages (run_id UUID; work_id canonical with doc_id alias; deterministic span_id and candidate_id_for rule specified and used).
  - Evidence (run_id UUID): `backend/contracts/pipeline_v1.py` `PipelineArtifactEnvelope._validate_run_id()` parses `UUID(val)`.
  - Evidence (work_id canonical + doc_id alias): `backend/contracts/pipeline_v1.py` `work_id = Field(validation_alias=AliasChoices("work_id", "doc_id"))`.
  - Evidence (deterministic span ids specified, compatible): `backend/contracts/pipeline_v1.py` `span_id_for()` implements `sha256("|".join([anchor_id, kind, exact, prefix, suffix, fp]))` and documents compatibility with `backend/span_graph_store.py` `_span_id()`.
  - Evidence (deterministic candidate ids specified + enforced): `backend/contracts/pipeline_v1.py` `candidate_id_for()`; `backend/pipeline_contracts/service.py` `_ensure_candidate_ids()` computes and rejects mismatched `candidate_id` values.

- [x] `POST /dev/wipe` removes pipeline run rows and pipeline artifacts (tables truncated + objects deleted).
  - Evidence (tables truncated): `backend/main.py` `/dev/wipe` truncation list includes `pipeline_stage_artifacts` and `pipeline_runs`.
  - Evidence (objects deleted): `backend/main.py` `/dev/wipe` calls `backend/object_store/s3.py` `delete_all()`.
  - Evidence (verified behavior): `tests/test_pipeline_contract_api.py` asserts after wipe, stage GET returns 404 and `object_store_s3.exists(stage_object_key(...)) is False`.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Each pipeline stage artifact is a versioned JSON envelope | ✓ VERIFIED | `backend/contracts/pipeline_v1.py` (`PipelineArtifactEnvelope`, `validate_v1()`) |
| 2 | Artifacts stored at `pipeline/{run_id}/{stage}.json` are write-once per (run_id, stage) and conflicts map to HTTP 409 without S3 overwrite | ✓ VERIFIED | `backend/spine/pipeline_artifacts.py` (`StageArtifactAlreadyExists`, reserve-before-put), `backend/main.py` (409), tests |
| 3 | API stores/fetches JSON bodies only; clients don’t touch S3 | ✓ VERIFIED | `backend/main.py` pipeline routes return envelope dict; no object key in contract |
| 4 | Contracts include stable IDs/aliases and deterministic id helpers/rules | ✓ VERIFIED | `backend/contracts/pipeline_v1.py`, `backend/pipeline_contracts/service.py` |
| 5 | `/dev/wipe` clears pipeline DB rows and pipeline artifacts | ✓ VERIFIED | `backend/main.py` + `backend/object_store/s3.py` + API test |

**Score:** 5/5 truths verified

## Required Artifacts

| Artifact | Expected | Status | Details |
| --- | --- | --- | --- |
| `backend/contracts/pipeline_v1.py` | v1 envelope + per-stage models + deterministic helpers | ✓ VERIFIED | Has envelope + validators + `stage_object_key`/`candidate_id_for`/`span_id_for` |
| `backend/contracts/upgrade.py` | upgrade + validate entrypoints | ✓ VERIFIED | `upgrade_contract_payload()`, `validate_contract_payload()` |
| `backend/spine/pipeline_artifacts.py` | DB reserve -> S3 put -> DB finalize; conflict => 409 path | ✓ VERIFIED | Reserve-before-put + `StageArtifactAlreadyExists` |
| `backend/main.py` | `/pipeline/*` endpoints + `/dev/wipe` includes pipeline tables | ✓ VERIFIED | Routes present; wipe truncates pipeline tables + deletes objects |
| `tests/test_pipeline_contract_store.py` | store/get + immutability + no overwrite | ✓ VERIFIED | Confirms stable S3 key and no overwrite via byte equality |
| `tests/test_pipeline_contract_api.py` | API round-trip + 409 + wipe cleanup | ✓ VERIFIED | Confirms 409 + wipe deletes object and 404s |
| `scripts/dev/verify_10_01_contracts.sh` | compose-backed smoke verifier | ✓ VERIFIED | Prints `OK: verify_10_01_contracts` |

## Key Link Verification

| From | To | Via | Status | Details |
| --- | --- | --- | --- | --- |
| `backend/main.py` | `backend/pipeline_contracts/service.py` | route handlers delegate | ✓ WIRED | `pipeline_contracts_service.create_run/store_stage/fetch_stage` |
| `backend/pipeline_contracts/service.py` | `backend/spine/pipeline_artifacts.py` | `put_stage_payload()` / `get_stage_payload()` | ✓ WIRED | service builds envelope then persists/fetches |
| `backend/spine/pipeline_artifacts.py` | `backend/object_store/s3.py` | reserve row then `put_bytes` | ✓ WIRED | conflict path does not call S3 put |
| `backend/spine/pipeline_artifacts.py` | Postgres | unique `(run_id, stage)` reserve | ✓ WIRED | `ON CONFLICT (run_id, stage) DO NOTHING` + unique index in migration |
| `/dev/wipe` | S3 works bucket | delete objects | ✓ WIRED | calls `object_store_s3.delete_all()` |

## Requirements Coverage

- `.planning/REQUIREMENTS.md` not present in repo; coverage assessed only against the plan-level goal + must-haves.

## Automated Verification Commands Run

- `bash scripts/dev/pytest_docker.sh`
  - Result: `94 passed, 4 skipped` (includes `tests/test_pipeline_contract_store.py` and `tests/test_pipeline_contract_api.py`).
- `bash scripts/dev/verify_10_01_contracts.sh`
  - Result: prints `OK: verify_10_01_contracts`.

## Anti-Patterns Found

- None detected in the phase’s key files (no TODO/FIXME/placeholder markers found).

---

_Verified: 2026-02-21T08:52:05Z_
_Verifier: OpenCode (gsd-verifier)_
