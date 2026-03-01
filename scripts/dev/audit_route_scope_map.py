#!/usr/bin/env python3
"""Audit FastAPI route scope handling from backend/main.py.

Generates a JSON snapshot inventory with route-level scope signals and
policy classification suggestions.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}
ADMIN_PREFIXES = (
    "/dev",
    "/maintenance",
    "/background",
    "/scope/observability",
)
FALLBACK_PATTERNS = {
    "default_project_fallback": re.compile(r"DEFAULT_PROJECT_ID"),
    "default_user_fallback": re.compile(r"DEFAULT_USER_ID"),
    "reviewer_default_fallback": re.compile(r'reviewer_uid\s*or\s*[\"\']default[\"\']'),
}


@dataclass
class RouteRecord:
    line: int
    method: str
    path: str
    function: str
    args: list[dict[str, Any]]
    has_x_project_id_arg: bool
    x_project_id_required: bool
    has_user_scope_arg: bool
    fallback_signals: list[str]
    resolve_scope_calls: list[dict[str, Any]]
    current_scope_class: str
    target_policy: str


def _const_value(node: ast.AST) -> Any | None:
    if isinstance(node, ast.Constant):
        return node.value
    return None


def _extract_routes(module: ast.Module, source_lines: list[str]) -> list[RouteRecord]:
    routes: list[RouteRecord] = []
    for node in module.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        decs: list[tuple[str, str, int]] = []
        for dec in node.decorator_list:
            if not (
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Attribute)
                and isinstance(dec.func.value, ast.Name)
                and dec.func.value.id == "app"
                and dec.func.attr in HTTP_METHODS
            ):
                continue
            if not dec.args:
                continue
            route_path = _const_value(dec.args[0])
            if not isinstance(route_path, str):
                continue
            decs.append((dec.func.attr.upper(), route_path, dec.lineno))

        if not decs:
            continue

        args = [a.arg for a in node.args.args]
        defaults = node.args.defaults
        required_cutoff = len(args) - len(defaults)
        arg_records = [
            {"name": name, "required": idx < required_cutoff}
            for idx, name in enumerate(args)
        ]

        has_x_project_id_arg = any(a["name"] == "x_project_id" for a in arg_records)
        x_project_id_required = any(
            a["name"] == "x_project_id" and bool(a["required"]) for a in arg_records
        )
        has_user_scope_arg = any(
            a["name"] in {"reviewer_uid", "x_reviewer_uid", "x_user_id"}
            for a in arg_records
        )

        start = max(1, node.lineno)
        end = max(start, int(getattr(node, "end_lineno", start)))
        fn_src = "\n".join(source_lines[start - 1 : end])

        fallback_signals = [
            key for key, pattern in FALLBACK_PATTERNS.items() if pattern.search(fn_src)
        ]

        resolve_scope_calls: list[dict[str, Any]] = []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == "_resolve_scope_observability":
                call_info = {
                    "line": sub.lineno,
                    "endpoint": None,
                    "require_project": None,
                    "allow_dev_project_default": None,
                    "include_user": None,
                }
                for kw in sub.keywords:
                    if kw.arg in call_info:
                        call_info[kw.arg] = _const_value(kw.value)
                resolve_scope_calls.append(call_info)

        for method, path, line in decs:
            current_scope = _classify_current_scope(
                path=path,
                has_x_project_id_arg=has_x_project_id_arg,
                x_project_id_required=x_project_id_required,
                has_user_scope_arg=has_user_scope_arg,
                fallback_signals=fallback_signals,
                resolve_scope_calls=resolve_scope_calls,
            )
            target_policy = _classify_target_policy(
                method=method,
                path=path,
                has_x_project_id_arg=has_x_project_id_arg,
                has_user_scope_arg=has_user_scope_arg,
                fallback_signals=fallback_signals,
            )

            routes.append(
                RouteRecord(
                    line=line,
                    method=method,
                    path=path,
                    function=node.name,
                    args=arg_records,
                    has_x_project_id_arg=has_x_project_id_arg,
                    x_project_id_required=x_project_id_required,
                    has_user_scope_arg=has_user_scope_arg,
                    fallback_signals=fallback_signals,
                    resolve_scope_calls=resolve_scope_calls,
                    current_scope_class=current_scope,
                    target_policy=target_policy,
                )
            )
    return sorted(routes, key=lambda item: (item.line, item.method, item.path))


def _classify_current_scope(
    *,
    path: str,
    has_x_project_id_arg: bool,
    x_project_id_required: bool,
    has_user_scope_arg: bool,
    fallback_signals: list[str],
    resolve_scope_calls: list[dict[str, Any]],
) -> str:
    if path.startswith(ADMIN_PREFIXES):
        return "admin-dev"

    if fallback_signals:
        return "fallback"

    strict_calls = [
        c
        for c in resolve_scope_calls
        if c.get("require_project") is True
        and c.get("allow_dev_project_default") is not True
    ]
    if x_project_id_required or strict_calls:
        return "strict"

    if has_x_project_id_arg and not strict_calls:
        return "fallback"

    if has_user_scope_arg:
        return "reviewer-scoped"

    return "unscoped"


def _classify_target_policy(
    *,
    method: str,
    path: str,
    has_x_project_id_arg: bool,
    has_user_scope_arg: bool,
    fallback_signals: list[str],
) -> str:
    if path.startswith(ADMIN_PREFIXES):
        return "admin/dev"

    if method == "GET" and not has_x_project_id_arg and not has_user_scope_arg and not fallback_signals:
        return "global-read"

    return "scoped-required"


def _to_json_dict(routes: list[RouteRecord], source_file: Path) -> dict[str, Any]:
    current_scope_totals = Counter(r.current_scope_class for r in routes)
    target_policy_totals = Counter(r.target_policy for r in routes)
    method_totals = Counter(r.method for r in routes)

    return {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_file": str(source_file),
        "route_total": len(routes),
        "totals": {
            "by_method": dict(sorted(method_totals.items())),
            "by_current_scope_class": dict(sorted(current_scope_totals.items())),
            "by_target_policy": dict(sorted(target_policy_totals.items())),
        },
        "routes": [
            {
                "line": r.line,
                "method": r.method,
                "path": r.path,
                "function": r.function,
                "args": r.args,
                "scope_signals": {
                    "has_x_project_id_arg": r.has_x_project_id_arg,
                    "x_project_id_required": r.x_project_id_required,
                    "has_user_scope_arg": r.has_user_scope_arg,
                    "fallback_signals": r.fallback_signals,
                    "resolve_scope_calls": r.resolve_scope_calls,
                },
                "current_scope_class": r.current_scope_class,
                "target_policy": r.target_policy,
            }
            for r in routes
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit route scope map from backend/main.py")
    parser.add_argument(
        "--source",
        default="backend/main.py",
        help="Path to FastAPI module to inspect (default: backend/main.py)",
    )
    parser.add_argument(
        "--out",
        default=".planning/phases/10-contracts-core-workflow-simplification/10-04.5-02-PR-02-ROUTE-POLICY-MAP.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    source_file = Path(args.source)
    out_file = Path(args.out)

    source_text = source_file.read_text(encoding="utf-8")
    source_lines = source_text.splitlines()
    module = ast.parse(source_text)
    routes = _extract_routes(module, source_lines)

    payload = _to_json_dict(routes, source_file)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote route policy snapshot: {out_file}")
    print(f"Total routes: {payload['route_total']}")
    print("Target policy totals:", payload["totals"]["by_target_policy"])
    print("Current scope totals:", payload["totals"]["by_current_scope_class"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
