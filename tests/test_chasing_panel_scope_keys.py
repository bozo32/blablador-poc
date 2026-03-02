from __future__ import annotations

from dataclasses import dataclass

from frontend.components import chasing_panel


@dataclass
class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _StubStreamlit:
    def __init__(self) -> None:
        self.session_state: dict = {}

    def markdown(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None

    def text_area(self, _label, key=None, **kwargs):
        if key is not None:
            return self.session_state.get(key, "")
        return ""

    def button(self, *args, **kwargs):
        return False

    def selectbox(self, _label, options, **kwargs):
        return options[0] if options else None

    def columns(self, spec, **kwargs):
        count = int(spec) if isinstance(spec, int) else len(spec)
        return [_Ctx() for _ in range(count)]

    def expander(self, *args, **kwargs):
        return _Ctx()


def _to_segment_dict(line: str) -> dict:
    text = str(line or "").strip()
    if "." in text:
        left, right = text.split(".", 1)
        return {"segment_id": left.strip(), "claim": right.strip()}
    return {"segment_id": "1a", "claim": text}


def test_requested_works_scope_is_namespaced_by_parent_panel(monkeypatch) -> None:
    stub = _StubStreamlit()
    monkeypatch.setattr(chasing_panel, "st", stub)

    captured_scopes: list[str] = []

    def _capture_render_requested_works_queue(**kwargs):
        captured_scopes.append(str(kwargs.get("scope") or ""))

    monkeypatch.setattr(
        chasing_panel.chase_queue_component,
        "render_requested_works_queue",
        _capture_render_requested_works_queue,
    )

    doc_id = "doc-1"
    citation_index = 1
    target_id = "target-1"
    reviewer_state = "reviewer"
    claim_id = f"cite:{doc_id}:{citation_index}:{reviewer_state}:1a"

    stub.session_state["citation_segments_by_reviewer"] = {
        reviewer_state: {f"{doc_id}::{citation_index}::{target_id}": ["1a. Claim text"]}
    }
    stub.session_state["workflow_selected_claim_id"] = claim_id
    stub.session_state["workflow_runs_by_claim_id"] = {claim_id: "run-123"}
    stub.session_state["workflow_status_cache"] = {"run-123": {"run": {"state": "running"}}}

    for scope in ("rail", "tab"):
        chasing_panel.render(
            doc_id=doc_id,
            citation_index=citation_index,
            target_id=target_id,
            scope=scope,
            get_context_cached=lambda *_: {"citing_sentence": "Sentence."},
            seg_via_llm=lambda *_: ["1a. Claim text"],
            to_segment_dict=_to_segment_dict,
            claim_queue_register=lambda **_: None,
            format_reference_summary=lambda *_: "",
            render_retrieval_instructions=lambda **_: None,
            api_url="http://localhost:8000",
            project_id="project-1",
            selected_model="local",
            active_reviewer_uid="Reviewer",
            reviewers=["Reviewer"],
            rerun=lambda: None,
        )

    assert captured_scopes == [
        f"rail::workflow::{claim_id}",
        f"tab::workflow::{claim_id}",
    ]
    assert captured_scopes[0] != captured_scopes[1]
