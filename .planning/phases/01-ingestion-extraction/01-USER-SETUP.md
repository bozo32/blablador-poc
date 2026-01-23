# Phase 1: User Setup Required

**Generated:** 2026-01-23
**Phase:** 01-ingestion-extraction
**Status:** Incomplete

Complete these items for the integration to function. Claude automated everything possible; these items require human access to external dashboards/accounts.

## Environment Variables

| Status | Variable | Source | Add to |
|--------|----------|--------|--------|
| [ ] | `GROBID_URL` | Local GROBID service base URL (example: http://localhost:8070) | `.env` |
| [ ] | `CROSSREF_MAILTO` | Email address required by Crossref REST API | `.env` |

## Verification

After completing setup, verify with:

```bash
curl http://localhost:8070/api/isalive
grep CROSSREF_MAILTO .env
```

Expected results:
- Response contains `true` when GROBID is running.
- `CROSSREF_MAILTO` is present in `.env`

---

**Once all items complete:** Mark status as "Complete" at top of file.
