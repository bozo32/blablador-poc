from __future__ import annotations

from dataclasses import dataclass

from frontend import ui


@dataclass
class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _Upload:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getbuffer(self):
        return self._data


class _StubStreamlit:
    def __init__(self) -> None:
        self.session_state: dict = {}
        self.captions: list[str] = []
        self.button_labels: list[str] = []

    def markdown(self, *args, **kwargs):
        return None

    def caption(self, value="", *args, **kwargs):
        self.captions.append(str(value))
        return None

    def error(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None

    def file_uploader(self, _label, key=None, on_change=None, **kwargs):
        if key and self.session_state.get(key) and callable(on_change):
            on_change()
        return self.session_state.get(key)

    def columns(self, spec, **kwargs):
        count = int(spec) if isinstance(spec, int) else len(spec)
        return [_Ctx() for _ in range(count)]

    def button(self, label, **kwargs):
        self.button_labels.append(str(label))
        return False

    def expander(self, *args, **kwargs):
        return _Ctx()


def test_unknown_drop_auto_routes_to_citing_without_blocking(monkeypatch) -> None:
    stub = _StubStreamlit()
    stub.session_state["intake_dropzone_nonce"] = 0
    stub.session_state["intake-dropzone::0"] = [_Upload("unknown.pdf", b"%PDF-1.4")]
    monkeypatch.setattr(ui, "st", stub)

    routed: list[str] = []
    monkeypatch.setattr(ui, "_intake_guess_intent", lambda **_: "unknown")
    monkeypatch.setattr(ui, "_intake_route_citing", lambda item_id: routed.append(item_id))
    monkeypatch.setattr(ui, "_intake_route_source", lambda *_, **__: None)
    monkeypatch.setattr(ui, "_rerun", lambda: None)

    ui.render_intake_panel(max_rows=None)

    inbox = stub.session_state.get("intake_inbox") or []
    assert len(inbox) == 1
    item = inbox[0]
    assert item["intent"] == "unknown"
    assert item["stage"] != "awaiting-intent"
    assert item["routed_intent"] == "citing"
    assert item["auto_routed"] is True
    assert item["override_available"] is True
    assert item["intent_guess"] == "unknown"
    assert len(routed) == 1


def test_status_refresh_uses_routed_intent_for_auto_routed_unknown(monkeypatch) -> None:
    stub = _StubStreamlit()
    monkeypatch.setattr(ui, "st", stub)
    monkeypatch.setattr(
        ui,
        "get_document",
        lambda *_args, **_kwargs: {
            "extraction": {"status": "complete", "data": {}},
            "resolution": {"status": "complete"},
        },
    )

    item = {
        "intent": "unknown",
        "routed_intent": "citing",
        "doc_id": "doc-1",
        "stage": "uploading",
    }
    ui._intake_refresh_item_status(item)
    assert item["stage"] == "done"


def test_auto_routed_unknown_keeps_optional_override_controls(monkeypatch) -> None:
    stub = _StubStreamlit()
    stub.session_state["intake_inbox"] = [
        {
            "id": "item-1",
            "filename": "paper.pdf",
            "size": 128,
            "intent": "unknown",
            "routed_intent": "citing",
            "auto_routed": True,
            "override_available": True,
            "stage": "uploading",
            "last_event_at": "2026-01-01T00:00:00Z",
            "error": "",
            "note": "",
            "doc_id": "doc-1",
            "source_queue_item_ids": [],
        }
    ]
    monkeypatch.setattr(ui, "st", stub)
    monkeypatch.setattr(ui, "_intake_refresh_item_status", lambda _item: None)
    monkeypatch.setattr(ui, "_rerun", lambda: None)

    ui.render_intake_panel(max_rows=None)

    assert "Route as Citing" in stub.button_labels
    assert "Route as Source" in stub.button_labels
    assert any("Optional override" in c for c in stub.captions)


def test_empty_intake_copy_mentions_non_blocking_auto_route(monkeypatch) -> None:
    stub = _StubStreamlit()
    monkeypatch.setattr(ui, "st", stub)
    ui.render_intake_panel(max_rows=None)
    assert any("auto-route to Citing" in c for c in stub.captions)
