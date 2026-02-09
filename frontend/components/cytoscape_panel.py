from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import streamlit.components.v1 as components


_COMPONENT_DIR = Path(__file__).resolve().parent / "cytoscape_component"


_cytoscape_component = components.declare_component(
    "cytoscape_component",
    path=str(_COMPONENT_DIR),
)


def render(
    elements: Any,
    *,
    style: Optional[Any] = None,
    layout: Optional[Dict[str, Any]] = None,
    height: int = 560,
    key: str = "cytoscape",
    selection: Optional[Dict[str, Any]] = None,
    focus: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Render Cytoscape and return selection as {type,id}.

    Contract is intentionally tiny to avoid deep Streamlit wiring.
    """
    value = _cytoscape_component(
        elements=elements or [],
        style=style or [],
        layout=layout or {},
        height=int(height),
        selection=selection or {},
        focus=focus or {},
        options=options or {},
        default={},
        key=key,
    )
    return value if isinstance(value, dict) else {}
