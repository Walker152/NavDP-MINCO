# Remove VS Code Git History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop tracking `.vscode/` and permanently remove its generated database objects from the local `develop` history.

**Architecture:** Commit the current working tree with an explicit `.vscode/` ignore rule, create a recoverable pre-rewrite checkpoint, then use `git filter-repo` to invert `.vscode/` out of `develop`. Validate content and history before deleting recovery references and pruning unreachable objects.

**Tech Stack:** Git, git-filter-repo, POSIX shell

---

### Task 1: Ignore `.vscode` and checkpoint current work

**Files:**
- Modify: `.gitignore`
- Commit: all currently tracked modifications and intended untracked experiment source files

- [ ] **Step 1: Add the ignore rule**

Append this repository-specific rule to `.gitignore`:

```gitignore
# Editor-generated indexes and workspace state.
.vscode/
```

- [ ] **Step 2: Remove currently tracked `.vscode` entries from the index**

Run:

```bash
git rm -r --cached --ignore-unmatch .vscode
```

Expected: `.vscode/browse.vc.db-wal`, `.vscode/c_cpp_properties.json`, and `.vscode/settings.json` are staged as deletions while local files remain.

- [ ] **Step 3: Stage the current source changes**

Run:

```bash
git add -A
git status --short
```

Expected: intended experiment changes and `.gitignore` are staged; ignored generated outputs are absent.

- [ ] **Step 4: Commit the checkpoint**

Run:

```bash
git commit -m "feat: finalize experiment tooling and ignore editor state"
```

Expected: a new `develop` commit and a clean working tree.

### Task 2: Create recovery metadata

**Files:**
- Create: `/tmp/navdp-pre-vscode-filter.txt`
- Create ref: `refs/backup/pre-vscode-filter`

- [ ] **Step 1: Record and reference the original tip**

Run:

```bash
git rev-parse HEAD | tee /tmp/navdp-pre-vscode-filter.txt
git update-ref refs/backup/pre-vscode-filter HEAD
```

Expected: the file and backup ref resolve to the same commit.

- [ ] **Step 2: Verify the checkpoint is clean**

Run:

```bash
git status --short
git fsck --no-dangling
```

Expected: no working-tree output and no repository integrity errors.

### Task 3: Rewrite `develop`

**Files:**
- Rewrite ref: `refs/heads/develop`

- [ ] **Step 1: Remove `.vscode` from every `develop` commit**

Run:

```bash
git filter-repo --force --refs refs/heads/develop --path .vscode --invert-paths
```

Expected: `develop` is rewritten and the working tree is checked out at the rewritten tip.

- [ ] **Step 2: Verify history and current content**

Run:

```bash
git rev-list --objects develop | grep -F ' .vscode/'
git ls-tree -r --name-only develop -- .vscode
git status --short
```

Expected: both path checks and status produce no output. Confirm the rewritten tip still contains the checkpointed source files and `.gitignore` rule.

### Task 4: Release obsolete objects and measure the result

**Files:**
- Delete ref: `refs/backup/pre-vscode-filter`
- Delete recovery refs created by local tooling if they retain pre-rewrite commits

- [ ] **Step 1: Remove refs that retain the old `develop` history**

Run `git for-each-ref --contains <old-tip>` first, then delete only the temporary backup and local checkpoint refs proven to retain the old history. Do not delete `refs/heads/master` or remote-tracking refs.

- [ ] **Step 2: Expire reflogs and prune unreachable objects**

Run:

```bash
git reflog expire --expire=now --all
git gc --prune=now
```

Expected: garbage collection finishes without errors.

- [ ] **Step 3: Perform final validation**

Run:

```bash
git count-objects -vH
du -sh .git
git rev-list --objects develop | grep -F ' .vscode/'
git status --short
```

Expected: `.vscode/` is absent from `develop`, the working tree is clean, and `.git` is substantially smaller than 15 GB.

- [ ] **Step 4: Report push instructions**

Do not push automatically. Report the rewritten tip and, after confirming the intended destination branch, use:

```bash
git push --force-with-lease <remote> develop:develop
```

