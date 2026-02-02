# GSD (Get Shit Done) Setup

This repo vendors the OpenCode GSD workflow under `.opencode/` so phase planning and execution work on any machine without relying on `~/.opencode`.

## Install / Update

1) Ensure OpenCode is installed on your machine.

2) Install the repo-local OpenCode dependencies:

```bash
cd .opencode
bun install
```

3) Run GSD commands from the repo root.

## Notes

- Secrets must remain local. Do not commit `.env`.
- `.opencode/node_modules` is intentionally gitignored.
