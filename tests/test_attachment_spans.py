from pathlib import Path

from backend import attachment_spans, attachment_store


def _create_ready_attachment(tmp_path: Path, tei_xml: str) -> str:
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-sample")
    record = attachment_store.create_attachment(
        claim_id="claim-1",
        doc_id=None,
        local_path=source,
    )
    artifacts = attachment_store.save_artifacts(
        record["id"],
        tei_xml=tei_xml,
        tei_json={},
        sentences=[],
    )
    attachment_store.mark_matched(record["id"], artifacts=artifacts)
    attachment_spans.clear_attachment_span_cache(record["id"])
    return record["id"]


def test_excerpt_does_not_cross_paragraph_boundaries(tmp_path: Path):
    tei = (
        '<TEI xmlns="http://www.tei-c.org/ns/1.0">'
        "<text><body>"
        "<p xml:id='p1'><s xml:id='s1'>One.</s><s xml:id='s2'>Two.</s></p>"
        "<p xml:id='p2'><s xml:id='s3'>Three.</s><s xml:id='s4'>Four.</s></p>"
        "</body></text></TEI>"
    )
    attachment_id = _create_ready_attachment(tmp_path, tei)
    index = attachment_spans.AttachmentSpanIndex.for_attachment(attachment_id)

    excerpt = index.excerpt("s3", before=2, after=1)
    assert [row["sentence_id"] for row in excerpt] == ["s3", "s4"]
    assert all(row["paragraph_id"] == "p2" for row in excerpt)


def test_section_path_falls_back_to_body(tmp_path: Path):
    tei = (
        '<TEI xmlns="http://www.tei-c.org/ns/1.0">'
        "<text><body><p><s xml:id='s1'>Hello.</s></p></body></text></TEI>"
    )
    attachment_id = _create_ready_attachment(tmp_path, tei)
    index = attachment_spans.AttachmentSpanIndex.for_attachment(attachment_id)
    jump = index.jump("s1")
    assert jump["section_path"] == "Body"


def test_jump_returns_page_label_from_pb(tmp_path: Path):
    tei = (
        '<TEI xmlns="http://www.tei-c.org/ns/1.0">'
        "<text><body><p><pb n='5'/><s xml:id='s1'>Hello.</s></p></body></text></TEI>"
    )
    attachment_id = _create_ready_attachment(tmp_path, tei)
    index = attachment_spans.AttachmentSpanIndex.for_attachment(attachment_id)
    jump = index.jump("s1")
    assert jump["page"] == "5"
