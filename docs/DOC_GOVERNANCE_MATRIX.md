# Documentation Governance Matrix

This matrix identifies authority, overlap, staleness risk, and near-term action.

## Fixed Governance Rules

- Precedence: Contracts > Protocol > Spec.
- Single entry: `docs/START_HERE.md` is mandatory for session startup.
- Contract boundary: contract docs define behavioral invariants; protocol docs define operator verification flow.
- Cloud/distributed posture is expressed in spec/rationale docs, not contract docs.

## Conflict/Staleness Matrix

| Document | Tier | Risk | Overlap/Conflict Surface | Status | Action |
|---|---|---|---|---|---|
| `docs/IDENTITY_SCOPE_VISIBILITY_CONTRACT.md` | 1 contract | low | Scope/identity rules may be restated elsewhere | active | Keep as canonical behavioral source |
| `docs/WORKFLOW_PROTOCOL.md` | 2 protocol | medium | May drift from current verifier scripts and endpoint guards | active | Keep canonical for Stage A-F, update when scripts change |
| `docs/REPO_SPEC.md` | 3 spec | medium | Contains behavior summary that can drift from contract/protocol details | active | Keep canonical for product model, add explicit defers to contract/protocol (done) |
| `docs/SYSTEM_DESIGN_RATIONALE.md` | 3 rationale | low | Strategic statements may imply behavior | active | Keep non-normative; defer behavior to contract/protocol |
| `docs/WORKFLOW_STAGE_BACKEND_NOTES.md` | reference | high | WIP/demo notes can conflict with current contract/protocol | provisional | Mark non-normative and candidate for archive indexing |
| `docs/GRAPH_COMPACTION_ROLLBACK_RUNBOOK.md` | protocol extension | low | Could miss evolving scope membership constraints | active | Keep; sync with contract changes as needed |
| `docs/EU_LIBRARY_CITATION_WALKING_PATH.md` | strategy | low | Directional goals may be mistaken for current behavior | active direction | Keep clearly non-normative |
| `docs/GSD.md` | tooling | medium | Can drift from `.opencode` command/workflow behavior | active | Keep as operator shortcut doc; refresh after workflow updates |
| `.planning/README.md` | runtime handoff | medium | Handoff can conflict with `STATE.md` if not refreshed | active | Require update at stop points |
| `.planning/STATE.md` | execution state | medium | Auto-regeneration may drop session context detail | active | Treat as current execution pointer, not full historical authority |
| `.planning/phases/10-*/**` | phase records | medium | Many docs can obscure latest accepted behavior | active + historical | Use phase INDEX + VERIFICATION as latest per phase |
| `.planning/debug/*.md` | debug journals | medium | Investigative notes can be mistaken for accepted policy | active debug | Keep as evidence logs; not normative |

## Priority Issues to Track

1. **Protocol-to-script drift risk**
   - Stage protocol references script behavior that can change quickly.
   - Mitigation: each script change should trigger `WORKFLOW_PROTOCOL.md` consistency check.

2. **Historical note confusion risk**
   - Backend WIP notes and debug journals look authoritative unless labeled.
   - Mitigation: non-normative labeling and archive classification.

3. **Spec detail drift risk**
   - Large spec can lag contract changes in strict scope behavior.
   - Mitigation: keep explicit "defers to contract/protocol" section and periodic contract sweep.
