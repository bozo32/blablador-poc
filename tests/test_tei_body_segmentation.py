from backend import tei_body


def _build_body(tei_xml: str):
    payload = tei_body.build_document_body(tei_xml)
    assert isinstance(payload, dict)
    assert "paragraphs" in payload
    return payload


def _first_paragraph_sentences(payload: dict):
    paragraphs = payload.get("paragraphs") or []
    assert len(paragraphs) == 1
    para = paragraphs[0]
    assert "sentences" in para
    return para["sentences"]


def _collect_citation_segments(payload: dict):
    citations = []
    for para in payload.get("paragraphs") or []:
        for sent in para.get("sentences") or []:
            for seg in sent.get("segments") or []:
                if seg.get("type") == "citation":
                    citations.append((sent, seg))
    return citations


def test_build_document_body_non_suspicious_keeps_tei_sentence_ids_and_citations():
    tei_xml = (
        '<TEI xmlns="http://www.tei-c.org/ns/1.0">'
        "<text><body>"
        "<p xml:id='p1'>"
        "<s xml:id='s1'>First sentence with plain text.</s>"
        "<s xml:id='s2'>Second sentence with a citation "
        "<ref type='bibr' target='#R1'>(Smith, 2020)</ref> and more text.</s>"
        "</p>"
        "</body></text></TEI>"
    )

    payload = _build_body(tei_xml)
    sentences = _first_paragraph_sentences(payload)

    assert [s.get("sentence_id") for s in sentences] == ["s1", "s2"]

    citations = _collect_citation_segments(payload)
    assert len(citations) == 1
    sent, cite = citations[0]
    assert cite.get("citation_index") == 0
    assert cite.get("target_id") == "R1"
    assert cite.get("sentence_id") == "s2"
    assert sent.get("sentence_id") == "s2"
    assert isinstance(sent.get("citation_indices"), list)
    assert 0 in sent.get("citation_indices")


def test_build_document_body_suspicious_single_s_triggers_fallback():
    long_a = "A" * 340
    long_b = "B" * 340
    tei_xml = (
        '<TEI xmlns="http://www.tei-c.org/ns/1.0">'
        "<text><body>"
        "<p xml:id='p1'>"
        "<s xml:id='s1'>"
        "This is sentence one. "
        f"{long_a} "
        "This is sentence two with a citation "
        "<ref type='bibr' target='#R1'>[1]</ref>. "
        f"{long_b} "
        "This is sentence three."
        "</s>"
        "</p>"
        "</body></text></TEI>"
    )

    payload = _build_body(tei_xml)
    sentences = _first_paragraph_sentences(payload)

    # Suspicious single <s> should be split into multiple fallback sentences.
    assert len(sentences) >= 2

    citations = _collect_citation_segments(payload)
    assert len(citations) == 1
    sent, cite = citations[0]
    assert cite.get("citation_index") == 0
    assert cite.get("target_id") == "R1"
    assert cite.get("sentence_id")
    assert cite.get("sentence_id") == sent.get("sentence_id")
