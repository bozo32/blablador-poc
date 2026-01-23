# Phase 2: User Setup Required

**Generated:** 2026-01-23
**Phase:** 02-citation-context-navigation
**Status:** Incomplete

Complete these items for the OpenAlex integration to function. Claude automated everything possible; these items require human access to external dashboards/accounts.

## Environment Variables

| Status | Variable | Source | Add to |
|--------|----------|--------|--------|
| [ ] | `OPENALEX_API_KEY` | OpenAlex account → API key | `.env` |

## Account Setup

- [ ] **Create OpenAlex account**
  - URL: https://openalex.org
  - Skip if: Already have an account with an API key

## Verification

After completing setup, verify with:

```bash
grep OPENALEX_API_KEY .env
```

Expected results:
- `OPENALEX_API_KEY` is present in `.env` for OpenAlex requests.

---

**Once all items complete:** Mark status as "Complete" at top of file.
