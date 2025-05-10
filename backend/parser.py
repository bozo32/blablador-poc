from __future__ import annotations

"""
backend.parser
--------------

Low‑level helpers for turning a GROBID‑TEI file into a list of
{text, meta} chunks that can be embedded and indexed.

* We only depend on lxml + std‑lib – no circular imports.
* Sentences are kept inside paragraphs for now; FAISS will still
  surface the paragraph and NLI will judge the sentence‑level claim.
"""

import re
from pathlib import Path
from typing import Dict, List
from lxml import etree
from .utils import embed  # noqa: F401  (imported for side‑effects elsewhere)
import pandas as pd
from .utils import read_csv

import logging

logger = logging.getLogger(__name__)

def tei_and_csv_to_documents(folder: Path, csv_path: str) -> list[dict]:
    """
    Load TEI-XML chunks + CSV citing sentences into one list of {'text','meta'} dicts.
    """
    tei_docs = []
    for xml_file in folder.glob("*.xml"):
        chunks = tei_to_chunks(xml_file)
        tei_docs.extend(chunks)

    df = read_csv(csv_path)
    # Build docs from CSV citing sentences, including author and year for retriever keys
    # Ensure Cited Year is an integer
    df["Cited Year"] = df["Cited Year"].astype(int)
    citing_docs = []
    for _, row in df.iterrows():
        doc_id = row["TEI File"]
        citing_docs.append({
            "id": doc_id,
            "text": row["tei_sentence"],
            "author": row["Cited Author"],
            "year": row["Cited Year"],
            "tei_file": row["TEI File"],
            "meta": {"id": doc_id}
        })

    return tei_docs + citing_docs

NS = {"tei": "http://www.tei-c.org/ns/1.0"}

_RE_WS = re.compile(r"\s+")



# Helper to strip XML tags and citation markers
def _strip_citations(txt: str) -> str:
    # remove anything in angle brackets and citation markers like #[...] or [\d+]
    txt = re.sub(r"<[^>]+>", "", txt)
    txt = re.sub(r"#\S+", "", txt)
    txt = re.sub(r"\[\d+\]", "", txt)
    return txt

def _clean(txt: str) -> str:
    """Collapse whitespace & strip."""
    txt = _strip_citations(txt)
    return _RE_WS.sub(" ", txt).strip()


def _add_chunk(chunks: List[Dict], text: str, kind: str, chunk_id: str | None = None) -> None:
    """Utility to append non‑empty chunks with a meta tag."""
    text = _clean(text)
    if text:
        meta = {"type": kind}
        if chunk_id is not None:
            meta["id"] = chunk_id
        chunks.append({"text": text, "meta": meta})


# --------------------------------------------------------------------------- #
#   Public API
# --------------------------------------------------------------------------- #
def tei_to_chunks(tei_path: Path) -> List[Dict]:
    """
    Parse a TEI file (from GROBID) and return a list of chunks:

        [{"text": "...", "meta": {"type": "paragraph"}}, ...]

    The extraction order is:
        1. Body paragraphs  (//text//p)
        2. Table rows       (//table//row//cell)
        3. Figure captions  (//figure//head | //figure//figDesc)
    """
    if not tei_path.exists():
        raise FileNotFoundError(tei_path)

    tree = etree.parse(str(tei_path))

    # Assign section IDs
    sec_ctr = 1
    div_meta_map: dict[etree._Element, dict] = {}
    for div in tree.xpath(".//tei:text//tei:body//tei:div", namespaces=NS):
        sid = div.get("{http://www.w3.org/XML/1998/namespace}id") or f"sec{sec_ctr}"
        sec_ctr += 1
        sec_type = div.get("type")
        head_elem = div.find("tei:head", namespaces=NS)
        sec_head = head_elem.text if head_elem is not None else None
        div_meta_map[div] = {"id": sid, "type": sec_type, "head": sec_head}

    chunks: List[Dict] = []

    sent_ctr = 1
    # 1. Sentence chunks with paragraph and section context
    for p in tree.xpath(".//tei:text//tei:body//tei:p", namespaces=NS):
        p_id = p.get("{http://www.w3.org/XML/1998/namespace}id") or f"p{sent_ctr}"
        # find nearest ancestor div for this paragraph
        ancestor_divs = p.xpath("ancestor::tei:div", namespaces=NS)
        section_info = None
        if ancestor_divs:
            div_elem = ancestor_divs[0]
            section_info = div_meta_map.get(div_elem)
        for s in p.xpath(".//tei:s", namespaces=NS):
            sent_text = "".join(s.itertext())
            sent_text = _clean(sent_text)
            chunk_id = s.get("{http://www.w3.org/XML/1998/namespace}id") or f"s{sent_ctr}"
            sent_ctr += 1
            if sent_text:
                meta = {
                    "source": str(tei_path),
                    "type": "sentence",
                    "id": chunk_id,
                    "p_id": p_id,
                    "section_id": section_info["id"] if section_info is not None else None,
                    "section_type": section_info.get("type") if section_info is not None else None,
                    "section_head": section_info.get("head") if section_info is not None else None
                }
                chunks.append({
                    "text": sent_text,
                    "id": chunk_id,
                    "meta": meta
                })
    print(f"Parsed {len(chunks)} sentence chunks from {tei_path.name}")
    return chunks


# ========================================================================= #
#  Data Integrity & Improvement Options                                      #
#                                                                            #
#  Current approach:                                                         #
#    - Extracts only <s> sentence elements, assigning a stable 'id' via     #
#      xml:id (if present) or fallback to numeric index.                    #
#    - Wraps each <s> parse in try/except, logging and skipping malformed.   #
#                                                                            #
#  Integrity considerations:                                                 #
#    - Skipped or malformed sentences result in fewer chunks than expected, #
#      potentially misaligning downstream CSV-to-chunk mappings.            #
#    - Table and figure content (e.g., <table>//<row>//<cell> or <figure>)  #
#      are not handled here; such elements are better parsed via dedicated   #
#      routines (see tei_to_chunks above).                                   #
#    - TEI files from GROBID focus on bibliographic structure over content   #
#      extraction, so some data (captions, footnotes, inline equations)      #
#      may be missing or incomplete.                                         #
#                                                                            #
#  Possible improvements:                                                    #
#    1) Placeholder stub chunks for malformed <s> to preserve alignment.     #
#    2) Raw-XML fallback to salvage content rather than drop entirely.       #
#    3) Pre-validate or auto-correct TEI via XSD/Schematron before parsing.  #
#    4) Extend to include table/figure parsing:                              #
#         - Split <table> rows/cells like tei_to_chunks does for paragraphs. #
#         - Extract <figure> captions with sentence splitting.               #
#    5) Surface errors fail-fast if full completeness is required, forcing   #
#       upstream TEI fixes.                                                  #
#                                                                            #
#  These notes outline trade-offs between robustness, completeness, and      #
#  complexity for evolving the parser as needed.                            #
# ========================================================================= #
def parse_chunks(tei_path: Path) -> List[dict]:
    """
    Parse a TEI file and return a list of chunks with text and metadata.
    """
    if not tei_path.exists():
        raise FileNotFoundError(f"TEI file not found: {tei_path}")

    tree = etree.parse(str(tei_path))
    docs = []

    for idx, s in enumerate(tree.findall('.//s', namespaces=NS)):
        try:
            xml_id = (s.get('{http://www.w3.org/XML/1998/namespace}id')
                      or s.get('xml:id'))
            text = ''.join(s.itertext()).strip()

            chunk = {
                "text": text,
                "xml_id": xml_id,
                # other metadata as needed
            }
            chunk["id"] = xml_id or str(idx)

            docs.append(chunk)
        except Exception as e:
            logger.warning(f"Skipping malformed <s> at index {idx}: {e}")

    print(f"Parsed {len(docs)} chunks from {tei_path.name}")
    return docs


# --------------------------------------------------------------------------- #
#   Figure caption sentence splitter
# --------------------------------------------------------------------------- #
def figure_caption_sentence(text: str, counter: int) -> list[dict]:
    """
    Split a figure caption into sentence-level chunks for indexing.
    Each sentence is returned as a dict with 'text' and 'meta':{'type':'figure', 'fig_id':counter}.
    """
    # naive sentence split on period followed by space; can be refined
    sentences = [s.strip() for s in text.split('. ') if s.strip()]
    chunks = []
    for idx, sent in enumerate(sentences):
        chunks.append({
            "text": sent.rstrip('.') + '.',
            "meta": {"type": "figure", "fig_id": counter, "sent_id": idx}
        })
    return chunks