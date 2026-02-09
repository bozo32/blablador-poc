# Graph-As-Architecture POC

This POC is a tiny, in-repo graph dataset rendered in the Surfing view.

- The boring nodes are the point: `Work` and `CiteAnchor` are stable anchors.
- Plurality begins at segmentation: two reviewers produce different `ClaimAtom`s from the same anchor.
- `ClaimAnchor`s are coordination objects, not truth claims.
- `Assertion`s are attributable links from people to evidence, not automated adjudications.
- "Heat" is disagreement density (mixed relation types) used to route attention, not correctness.

Run the app, switch to Surfing, pick the POC dataset, and click around: the inspector should explain what each node means and why the canvas stays quiet by default.
