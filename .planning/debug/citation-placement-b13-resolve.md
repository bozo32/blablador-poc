---
status: awaiting_human_verify
trigger: |
  Citation placement does not resolve retrieval target after explicit user placement

  Goal
  Make retrieval resolve Hicks (2024) for Rudko target b13 after user placement, so retrieval instructions stop showing fallback upload guidance and show a resolved/complete state.

  Observed behavior
  - In Retrieval instructions, target row shows:
    - "Hicks (2024) incomplete_circle"
    - "Retrieving"
  - Reference summary correctly identifies:
    - "ChatGPT is bullshit (2024)"
    - DOI: 10.1007/s10676-024-09775-5
  - UI still instructs manual upload/rename fallback instead of treating target as resolved.

  Expected behavior
  - After placing Hicks to the active callout target (b13) in Rudko:
    1) ref mapping should persist: ref:{rudko_ingest}:b13 -> ingest_ids=[hicks_ingest]
    2) resolve_reference_to_ingest_id(rudko, b13) should return hicks_ingest
    3) retrieval panel should render resolved/complete (no fallback instructions)

  Recent relevant changes already made
  - Added /ledger/place-reference endpoint
  - Frontend calls ledger_api.place_reference in stray placement flow when relation is "is cited by"
  - Added fallback in graph_store.link_reference_to_ingest to locate ref node via CITES+ref_id if alias missing
  - link_reference_to_ingest now force-updates ingest_ids and sets resolved=true
  - /ledger/place-reference now returns 404 if reference node cannot be linked
  - Frontend gating relaxed to allow selected_doc_id OR selected_callout_tuple.doc_id to match target doc

  Primary hypotheses to test
  1) Frontend is not issuing /ledger/place-reference in the failing flow (missing/incorrect callout tuple or relation mismatch).
  2) Endpoint is called but returns 404 due to missing ref node for ref:{citing}:{reference}.
  3) Ref node is updated but retrieval reads from different project scope or wrong citing/target identifiers.
  4) Resolver path uses stale/mismatched ids (target_id normalization, alias mismatch, or cache/rerun stale state).

  Required debugger outputs
  - Exact minimal repro steps with concrete ids used (citing ingest id, reference_id, cited ingest id).
  - Proof of whether /ledger/place-reference fired (request payload + response code).
  - Graph DB state before/after placement:
    - alias row for ref:{citing}:{reference}
    - graph_nodes.properties.ingest_ids for resolved ref node
    - relevant CITES edges and ref_id values
  - Resolver check result:
    - resolve_reference_to_ingest_id(citing, reference) return value
  - Root cause with confidence level.
  - Smallest safe code fix + verification steps.
  - Regression checks for clustered citations and non-targeted doc-doc placement paths.

  Verification criteria
  - Reproduced failing case turns green/resolved in retrieval instructions.
  - No regression in normal unresolved references.
  - No regression in existing opinion/follow rail behavior.
created: 2026-03-01T18:46:51+00:00
updated: 2026-03-01T19:07:27+00:00
---

## Current Focus

hypothesis: root cause fixed in code; awaiting user confirmation in real Rudko/Hicks workflow
test: user validates b13 placement flow in UI after pulling changes
expecting: Hicks target card shows resolved/check state and no fallback upload guidance for mapped target
next_action: await human verification response

## Symptoms

expected: |
  After placing Hicks (2024) to active Rudko callout target b13, mapping persists as
  ref:{rudko_ingest}:b13 -> ingest_ids=[hicks_ingest], resolver returns hicks_ingest,
  and retrieval instructions render resolved/complete with no fallback upload guidance.
actual: |
  Retrieval row for Hicks (2024) remains in "Retrieving" / incomplete state and UI still
  shows manual upload/rename fallback guidance despite reference summary correctly identifying
  the citation and DOI.
errors: No explicit runtime error reported; unresolved UI state persists post-placement.
reproduction: |
  1) Open Rudko retrieval instructions with target b13 active.
  2) Place Hicks (2024) onto that callout via relation "is cited by" flow.
  3) Observe retrieval panel remains unresolved and continues fallback instructions.
started: Reproducible after recent placement/ledger integration changes.

## Eliminated

- hypothesis: Frontend failing flow never reaches /ledger/place-reference
  evidence: Manual placement payload succeeds with 200 and expected graph mutation; code path includes valid call when relation/is-cited-by + callout tuple match are present
  timestamp: 2026-03-01T18:58:42+00:00

- hypothesis: Backend link_reference_to_ingest fails to persist ref ingest mapping
  evidence: Graph DB after placement shows alias ref:citing:reference exists and node properties.ingest_ids updated to cited ingest id with resolved=true
  timestamp: 2026-03-01T18:58:42+00:00

## Evidence

- timestamp: 2026-03-01T18:47:30+00:00
  checked: initial codebase symbol search for placement, resolver, and retrieval UI status labels
  found: frontend/ui.py contains stray placement branch invoking ledger_api.place_reference; backend/main.py exposes POST /ledger/place-reference; backend/graph_store.py contains link_reference_to_ingest and resolve_reference_to_ingest_id
  implication: end-to-end path exists in code and can be traced for payload/identifier mismatches

- timestamp: 2026-03-01T18:49:41+00:00
  checked: frontend placement branch, retrieval panel rendering, and backend retrieval/resolver endpoints
  found: UI sets retrieval chip from citation context resolution.status (chasing_panel._status_icon) and always prints "Retrieving" text; retrieval dossier endpoint /references/.../retrieval does not consult graph resolver and uses default project only; /ledger/place-reference writes graph mapping in project-scoped GraphStore
  implication: even successful ref->ingest linking may not change retrieval panel state because panel state comes from extraction/resolution data, not graph mapping, and project scoping may diverge

- timestamp: 2026-03-01T18:58:42+00:00
  checked: live API repro in project proj-a with citing=f0fde789-5490-42c8-8ca8-cbb04d6fec74, reference_id=b0, cited=48733816-e4b4-4406-bb3b-72e44ec73083
  found: POST /ledger/place-reference payload returned HTTP 200; graph_aliases has ref:citing:b0 row; graph_nodes for ref node updated ingest_ids to [cited] and resolved=true; CITES edge ref_id=b0 exists; direct scoped GraphStore.resolve_reference_to_ingest_id returns cited ingest id; but POST /graph/resolve-references still returns null mapping (unscoped/default project)
  implication: placement writes correctly, but resolver/read path used by UI is project-mismatched and retrieval rendering is not linked to this mapping

- timestamp: 2026-03-01T19:05:09+00:00
  checked: post-fix API behavior and syntax validation
  found: /graph/resolve-references with X-Project-Id now returns mapping b0->48733816-e4b4-4406-bb3b-72e44ec73083 and keeps unresolved b1->null; modified files compile via py_compile; frontend chasing panel now derives check icon + "Resolved" state from resolver mapping and suppresses upload fallback when resolved_ingest_id is present
  implication: target placement mapping is now consumable by UI and should render resolved state without fallback guidance while preserving unresolved behavior

## Resolution

root_cause:
  Retrieval completion state was not connected to ref->ingest mapping, and graph reference resolver endpoint read from default project scope instead of active project.
fix: |
  1) Scoped /graph/resolve-references by X-Project-Id using a project-specific GraphStore.
  2) Passed project header from frontend graph_api.resolve_references.
  3) Updated chasing panel to query resolver mapping per reference and treat mapped refs as complete/resolved.
  4) Updated retrieval instructions renderer to suppress fallback upload guidance when a resolved ingest mapping exists.
verification:
  API repro confirms mapped reference resolves to cited ingest id in project scope; unresolved reference remains null; syntax checks pass for modified files.
files_changed:
  - backend/main.py
  - frontend/graph_api.py
  - frontend/components/chasing_panel.py
  - frontend/claim_queue.py
  - frontend/ui.py
