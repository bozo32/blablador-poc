"""Disk-backed persistence for evidence ranking runs and history."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from backend.settings import AppSettings, settings as app_settings

from . import serializers


class EvidenceRunStore:
    """Persist evidence ranking runs with limited history and delta metadata."""

    def __init__(self, *, settings: AppSettings | None = None) -> None:
        """Create a store that persists runs under the configured directory."""
        self.settings = settings or app_settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_run(
        self,
        claim_id: str,
        *,
        candidates: Sequence[Any],
        metadata: dict | None = None,
    ) -> dict:
        """Persist a ranking run for a claim and keep capped history."""
        if not claim_id:
            raise ValueError("claim_id is required")

        normalized_candidates = [self._normalize_candidate(c) for c in candidates]
        previous = self.latest_run(claim_id)
        previous_candidates = previous.get("candidates", []) if previous else []
        annotated = self._annotate_deltas(normalized_candidates, previous_candidates)

        created_at = _utcnow()
        run_id = self._run_id(created_at)
        payload = {
            "claim_id": claim_id,
            "run_id": run_id,
            "created_at": created_at,
            "metadata": dict(metadata or {}),
            "summary": self._build_summary(annotated),
            "candidates": annotated,
        }

        claim_dir = self._claim_dir(claim_id)
        latest_path = claim_dir / "run.json"
        history_path = self._history_dir(claim_id) / f"{run_id}.json"

        _write_json_atomic(history_path, payload)
        _write_json_atomic(latest_path, payload)
        self._trim_history(claim_id)
        return payload

    def latest_run(self, claim_id: str) -> dict | None:
        """Return the latest persisted run for a claim, if any."""
        latest_path = self._claim_dir(claim_id) / "run.json"
        if not latest_path.exists():
            return None
        return _read_json(latest_path)

    def history(self, claim_id: str) -> list[dict]:
        """Return recent history for a claim, newest first."""
        history_dir = self._history_dir(claim_id)
        if not history_dir.exists():
            return []
        entries: list[tuple[float, dict]] = []
        for path in history_dir.glob("*.json"):
            try:
                payload = _read_json(path)
            except json.JSONDecodeError:
                continue
            entries.append((path.stat().st_mtime, payload))
        entries.sort(key=lambda pair: pair[0], reverse=True)
        return [payload for _, payload in entries]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _root_dir(self) -> Path:
        root = Path(self.settings.EVIDENCE_STORE_DIR)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _claim_dir(self, claim_id: str) -> Path:
        safe_claim = claim_id.replace("/", "_")
        path = self._root_dir() / safe_claim
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _history_dir(self, claim_id: str) -> Path:
        path = self._claim_dir(claim_id) / "history"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _trim_history(self, claim_id: str) -> None:
        depth = max(1, int(getattr(self.settings, "EVIDENCE_HISTORY_DEPTH", 5)))
        history_dir = self._history_dir(claim_id)
        paths = sorted(
            history_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for stale in paths[depth:]:
            try:
                stale.unlink()
            except FileNotFoundError:  # pragma: no cover - race-resistant
                continue

    def _normalize_candidate(self, candidate: Any) -> dict:
        if hasattr(candidate, "id") and hasattr(candidate, "claim_id"):
            # Assume EvidenceCandidate-like dataclass
            try:
                return serializers.serialize_candidate(candidate)
            except Exception:  # pragma: no cover - fallback to dict conversion
                pass
        if hasattr(candidate, "to_dict") and callable(candidate.to_dict):
            return candidate.to_dict()
        if isinstance(candidate, dict):
            return deepcopy(candidate)
        raise TypeError("Unsupported candidate payload")

    def _annotate_deltas(
        self, candidates: Sequence[dict], previous_candidates: Sequence[dict]
    ) -> list[dict]:
        prev_index = {
            cand.get("id"): idx for idx, cand in enumerate(previous_candidates)
        }
        prev_scores = {
            cand.get("id"): _score(cand)
            for cand in previous_candidates
            if cand.get("id") is not None
        }
        annotated: list[dict] = []
        for idx, candidate in enumerate(candidates):
            data = deepcopy(candidate)
            data.setdefault("scores", {})
            data["scores"].setdefault("position", idx + 1)
            prev_idx = prev_index.get(data.get("id"))
            prev_pos = (prev_idx + 1) if prev_idx is not None else None
            current_pos = int(data["scores"].get("position") or idx + 1)
            rank_change = None if prev_pos is None else prev_pos - current_pos
            score_delta = None
            if prev_pos is not None:
                score_delta = _round_score(
                    (_score(data) or 0.0)
                    - (prev_scores.get(data.get("id"), 0.0) or 0.0)
                )
            status: str
            demotion_reason: str | None = None
            if prev_pos is None:
                status = "new"
            elif rank_change is None or rank_change == 0:
                status = "stable"
            elif rank_change > 0:
                status = "promoted"
            else:
                status = "demoted"
                demotion_reason = "rerank-adjustment"

            diversity_note = _diversity_note(data.get("label"))
            data["delta"] = {
                "rank_change": rank_change,
                "score_delta": score_delta,
                "status": status,
                "demotion_reason": demotion_reason,
                "diversity_note": diversity_note,
            }
            annotated.append(data)
        return annotated

    def _build_summary(self, candidates: Sequence[dict]) -> dict:
        labels = Counter(
            (cand.get("label") or "unknown").lower() for cand in candidates
        )
        return {
            "total": len(candidates),
            "label_counts": dict(labels),
        }

    @staticmethod
    def _run_id(created_at: str) -> str:
        compact = created_at.replace("-", "").replace(":", "").replace(".", "")
        return f"{compact}-{uuid4().hex[:8]}"


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _score(candidate: dict | None) -> float | None:
    if not candidate:
        return None
    scores = candidate.get("scores") or {}
    combined = scores.get("combined")
    try:
        return float(combined) if combined is not None else None
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _round_score(value: float) -> float:
    return round(value, 4)


def _diversity_note(label: Any) -> str | None:
    label_text = (label or "").lower()
    if label_text == "contradicts":
        return "contradiction-slot"
    if label_text == "neutral":
        return "neutral-coverage"
    return None


__all__ = ["EvidenceRunStore"]
