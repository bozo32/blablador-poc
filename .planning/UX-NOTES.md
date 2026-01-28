---
title: UX Backlog / V2 Notes
updated: 2026-01-27T19:20:00Z
---

## Attachment Workflow Improvements

1. **Global attachment inbox drop zone**  
   *Why*: Scrolling across every claim to drop files is awkward, especially for long documents. A single drop target that auto-matches files to the claims the reviewer flagged for retrieval (with manual reassignment fallback) would speed up bulk uploads and reduce misdrops.

2. **Hide demo placeholder claims once real data flows**  
   *Why*: Seeded demo cards are useful for development but confusing in production; the workspace should mirror the user’s actual claim list so they aren’t unsure which cards are actionable.

3. **Three-pane layout (VS Code style)**  
   *Why*: The original ask called for navigation/project context on the left, claim/attachment workspace center, and citation follow/view panel on the right. Streamlit’s current two-column approach feels cramped; consider either custom CSS or another framework once the operational spine is stable.

4. **Real processing status integration**  
   *Why*: The current queue simulates converting/parsing transitions. Once Phase 4.3 wires to the backend pipeline, replace the fake timers with actual ingestion/parsing progress so reviewers trust the status data.

5. **Attachment timeline ergonomics**  
   *Why*: Timelines scroll off-screen quickly. Explore condensed summaries (e.g., latest status + hover for history) so users can glance at attachment state without expanding each panel.

## Notes

- Revisit after Phase 4 spine (retrieval + parsing) is complete; treat this file as the backlog for a dedicated UX polish phase.
