# Phase 1: User Setup Required

**Generated:** 2026-01-23
**Phase:** 01-ingestion-extraction
**Status:** Incomplete

Complete these items for the integration to function. Claude automated everything
possible; these items require human access to local services.

## Environment Variables

| Status | Variable | Source | Add to |
|--------|----------|--------|--------|
| [ ] | `GROBID_URL` | Local GROBID service base URL (example: http://localhost:8070) | `.env` |

## Verification

After completing setup, verify with:

```bash
curl http://localhost:8070/api/isalive
```

Expected results:
- Response contains `true` when GROBID is running.

---

**Once all items complete:** Mark status as "Complete" at top of file.
