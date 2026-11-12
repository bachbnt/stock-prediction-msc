# backup-v1

Snapshot of the old branches, each kept byte-for-byte in its own folder.
All original commits remain reachable as parents of the tip commit (git log --first-parent / git log <sha>).

| Folder | Source branch | Commit | Date | Message |
|---|---|---|---|---|
| `main/` | `main` | `0f34166` | 2022-02-03 | first commit |
| `dev/` | `dev` | `7ba4507` | 2023-06-10 | add report + slide |
| `v2/` | `v2` | `abcd660` | 2022-06-26 | add v2 |
| `v3/` | `v3` | `da7fc07` | 2023-02-19 | v3 |
| `origin-data/` | `origin/data` | `128e854` | 2022-04-24 | env mac m1 |

Restore one branch: `git checkout backup-v1 -- <folder>` or `git worktree add ../x da7fc07`.
