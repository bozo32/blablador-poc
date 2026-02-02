# Publishing From Private Repo To Public Repo

You work in the private repo (`blablador-poc`) and selectively publish code/docs to the public repo (`os-ERIN`).

## One-Time Remote Setup

In your private clone:

```bash
# origin = private
git remote -v

# add public remote
git remote add public https://github.com/bozo32/os-ERIN.git
git fetch public
```

## Publish A Change Set

Recommended: publish from a dedicated export branch, and only include public-safe commits.

```bash
git checkout -b export/<topic>

# pick commits you want to publish
git cherry-pick <commit> <commit> <commit>

# push export branch to public
git push public export/<topic>
```

Then open a PR on the public repo from `export/<topic>`.

## Public Safety Checklist

Do NOT publish:

- `.env` or any credentials
- `data/` or large corpora (unless explicitly intended for public release)
- non-fixture PDFs
- private demo traces or copyrighted materials

OK to publish:

- `backend/`, `frontend/`, `tests/`, `scripts/`, `environment.yml`
- `.planning/` planning docs (if you want planning in public)
- `.opencode/` workflow definitions (excluding `node_modules/`)
