# Repo Status

Date: 2026-04-22

This file summarizes the state of the curated GitHub-aligned Gemma4Good repository.

## Repository Role

This repository is the curated public-facing source layer for Gemma4Good.

It is meant to contain:

- the Kaggle-facing notebooks
- the governance and viability source code
- experiment and training utilities that are useful as source-of-truth
- explanatory docs and writeups

It is not meant to contain:

- local logs
- secrets
- heavyweight deployment artifacts
- one-off scratch patch helpers
- generated zip bundles that can be rebuilt from source

## Deployment Context

In the wider HAIC environment:

- `haic-gemma4-v35-gov` is the current promoted interviewer runtime
- the live GGUF and runtime defaults live outside this repo
- Kaggle archives and benchmark artifacts live outside this repo

This repo documents and supports that work, but does not attempt to be the deployment-artifact store.

## Push-Readiness Notes

At the time of this pass, the GitHub-aligned branch contains:

- core source modules
- tests
- notebook assets and helper scripts
- writeup and deployment docs
- curated experiment utilities
- maintainer workflow docs

The branch is organized as a sequence of small local commits rather than a single giant import.

## Remaining Structural Risk

The runtime tree's `master` history and `origin/main` are unrelated histories.

That means:

- do not merge the runtime tree directly into `origin/main`
- do not try to "fix" the history with force-push or rewrite operations
- continue porting from runtime into the GitHub-aligned branch in reviewed slices when needed

## Recommended Publishing Lane

For any future push or PR preparation, use:

`D:\gemma4good\_local_worktrees\clean-github-aligned`

That is the lane intended to line up with the public GitHub repository.
