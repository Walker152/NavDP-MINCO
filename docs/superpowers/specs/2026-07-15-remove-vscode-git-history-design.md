# Remove VS Code Indexes from Git History

## Goal

Reduce repository and push size by permanently removing `.vscode/` generated index files from the `develop` branch history, while preserving the current source changes and the useful commit history.

## Chosen approach

Use `git filter-repo` to rewrite local history and exclude `.vscode/` from every rewritten commit. This preserves commit topology, authorship, messages, and all content outside `.vscode/`. A temporary backup reference will be created before rewriting and deleted before final garbage collection so the removed database objects can actually be pruned.

## Workflow

1. Add `.vscode/` to `.gitignore` and commit the user's current working-tree changes together with that ignore rule.
2. Verify the new commit contains no `.vscode/` entries.
3. Create a temporary local backup reference for the pre-rewrite `develop` tip and record the commit ID outside Git's object database.
4. Run `git filter-repo` over the intended local branch history with `.vscode/` inverted out.
5. Verify that no reachable commit contains `.vscode/`, and that the current source files remain present.
6. Remove the temporary backup reference, expire reflogs, prune unreachable objects, and consolidate packs.
7. Report the resulting repository size and the exact force-with-lease push command. Do not push automatically.

## Safety and recovery

The current work is committed before rewriting, so it participates in the rewrite and is not lost. The pre-rewrite tip ID is also recorded in `/tmp` for emergency recovery. Garbage collection is delayed until history and working-tree checks pass. Remote branches are not rewritten locally unless they are explicitly part of the selected rewrite, and no network mutation occurs during this operation.

## Validation

- `git status --short` is clean immediately before rewriting.
- `git log --all -- .vscode` produces no results for the rewritten branch scope.
- `git rev-list --objects develop` contains no `.vscode/` path.
- The rewritten tip contains the user's current source changes.
- `git count-objects -vH` and `du -sh .git` show that the historical database objects are no longer retained after pruning.

