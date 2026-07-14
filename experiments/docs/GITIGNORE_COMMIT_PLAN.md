# Experiment Result Ignore and Commit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Execute this plan inline. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ignore generated experiment results and caches, then commit the experiment framework without unrelated user changes.

**Architecture:** Repository-level `.gitignore` owns generated-output policy. Experiment source, configs, tests, and documentation remain tracked; generated `results/` and runtime caches remain untracked.

**Tech Stack:** Git ignore patterns and Git CLI.

---

### Task 1: Add ignore rules

**Files:**
- Modify: `.gitignore`

- [ ] Run `git check-ignore -q results/navdp_minco_mock_smoke/suite_config.json`; expect exit 1 before the rule exists.
- [ ] Add `/results/`, `/experiment_outputs/`, `*.pyc`, and `.pytest_cache/`.
- [ ] Run the same check; expect exit 0.

### Task 2: Verify and commit

**Files:**
- Commit: `.gitignore`, `experiments/`, `docs/experiments/04_实验软件架构设计.md`, `docs/experiments/05_实验架构实施计划.md`
- Exclude: `pointgoal_navdp_cluttered_easy/easy_4/metric.csv`

- [ ] Run the unittest suite; expect 15 tests and OK.
- [ ] Stage only the listed deliverables, using `git add -f` for the two design documents because the repository already ignores `docs/*`.
- [ ] Inspect `git diff --cached --name-status` and confirm no result files or `metric.csv`.
- [ ] Commit with message `feat: add NavDP MINCO experiment toolkit`.
