---
status: diagnosed
trigger: "/gsd-debug Permanent fix for chasing duplicate widget keys and upload intent/role friction"
created: 2026-03-02T10:52:03+00:00
updated: 2026-03-02T10:55:37+00:00
---

## Current Focus

hypothesis: duplicate Streamlit keys come from rendering the same run queue in multiple panels with identical scope-derived keys; upload role prompt persists because intake architecture branches into mutually-exclusive citing vs source routes before processing
test: correlate key format with render callsites and check intake unknown-intent branch/routing side effects
expecting: confirm same key namespace reused across rail/tab and confirm explicit unknown->manual intent gating in upload panel
next_action: return root-cause diagnosis and minimal patch plan (no code changes in this run)

## Symptoms

expected: chasing queue renders without Streamlit key collisions; upload flow does not ask manual citing/source role and still infers links automatically
actual: chasing queue throws StreamlitDuplicateElementKey for selectbox keys under duplicated requested-work rows; upload intake still prompts "what role does this document have?"
errors: "StreamlitDuplicateElementKey" with key pattern "workflow::cite:{doc_id}:{citation_index}:default:{...}::assign::{attachment_id}::ref:{doc_id}:{citation_index}:{...}" from ui.py -> chasing_panel.py -> chase_queue.py
reproduction: open chasing queue with duplicate requested-work entries hitting same scope/rid/target-derived key; use upload intake path and observe role prompt before ingest
started: after recent scope/ACL work (exact commit/time not yet confirmed)

## Eliminated

- hypothesis: duplicate queue rows from backend are the primary source of key collisions
  evidence: pipeline_target_status enforces ON CONFLICT (run_id, target_id), and /workflow/runs/{run_id}/status queue is derived from that table, so same target cannot duplicate within one run without data corruption
  timestamp: 2026-03-02T10:55:37+00:00

## Evidence

- timestamp: 2026-03-02T10:52:29+00:00
  checked: user-reported symptom bundle
  found: two persistent issues (duplicate widget key + role prompt friction) with traceback path and expected outcomes provided
  implication: enough context to begin direct code investigation without additional reporter questions

- timestamp: 2026-03-02T10:55:37+00:00
  checked: frontend/components/chase_queue.py render_requested_works_queue key construction
  found: selectbox/button keys are built as f"{scope}::assign::{rid}::{target_id}" and f"{scope}::assign-btn::{rid}::{target_id}"
  implication: uniqueness depends entirely on scope+run_id+target_id namespace

- timestamp: 2026-03-02T10:55:37+00:00
  checked: frontend/components/chasing_panel.py call to render_requested_works_queue
  found: scope is forced to f"workflow::{selected_claim_id}" and does not include parent panel scope (rail vs tab)
  implication: identical claim panel rendered in multiple UI regions reuses identical widget keys

- timestamp: 2026-03-02T10:55:37+00:00
  checked: frontend/ui.py call graph
  found: chasing_panel.render executes in right rail (queue panel rendering) and also in center Review tab; both can render same selected claim in one Streamlit run
  implication: deterministic duplicate widget-key collision path exists even without backend duplicate queue rows

- timestamp: 2026-03-02T10:55:37+00:00
  checked: frontend/ui.py intake flow (_intake_guess_intent, render_intake_panel, _intake_route_citing/_intake_route_source)
  found: unknown uploads are staged as awaiting-intent and require explicit Citing/Source button before route-specific processing starts
  implication: manual role prompt is structurally required by current frontend intake contract

- timestamp: 2026-03-02T10:55:37+00:00
  checked: backend/main.py and backend/attachment_pipeline.py ingest vs attachment paths
  found: /ingest path triggers extraction/resolution workflow for citing docs; /attachments/upload path processes attachment + optional ingest promotion but does not trigger equivalent reference-resolution flow
  implication: fully automatic role removal needs a unified intake path or staged orchestration that guarantees graph-complete processing without user intent

## Resolution

root_cause: |
  Issue A: render_requested_works_queue key namespace is not panel-instance-safe. The same claim workflow queue can render from both rail and review-tab panels in one Streamlit pass, and both use identical key prefixes (scope="workflow::{selected_claim_id}") plus rid/target_id, causing deterministic StreamlitDuplicateElementKey collisions.
  Issue B: intake currently requires intent because unknown uploads are paused at awaiting-intent and routed into two different pipelines with different guarantees: citing (/ingest with extraction/resolution) vs source (/attachments/upload with attachment processing + optional ingest promotion). Without intent, the system cannot decide which processing contract to apply.
fix: 
verification: 
files_changed: []
