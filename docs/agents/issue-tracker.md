# Issue tracker: GitHub

Issues and specs for this repo live as GitHub issues in **ACGNnsj/Wan2GP**. Use the `gh` CLI for all operations.

## Conventions

- **Create an issue**: `gh issue create --repo ACGNnsj/Wan2GP --title "..." --body "..."`. Use `--body-file` for multi-line bodies.
- **Read an issue**: `gh issue view <number> --repo ACGNnsj/Wan2GP --comments`, also fetching labels.
- **List issues**: `gh issue list --repo ACGNnsj/Wan2GP --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --repo ACGNnsj/Wan2GP --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --repo ACGNnsj/Wan2GP --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --repo ACGNnsj/Wan2GP --comment "..."`

Always pass `--repo ACGNnsj/Wan2GP` explicitly: this workspace is reached over a UNC path, and commands that walk up from the current directory (bare `gh` repo inference) are unreliable here.

## Pull requests as a triage surface

**PRs as a request surface: no.** _(Set to `yes` if this repo treats external PRs as feature requests; `/triage` reads this flag.)_

When set to `yes`, PRs run through the same labels and states as issues, using the `gh pr` equivalents:

- **Read a PR**: `gh pr view <number> --repo ACGNnsj/Wan2GP --comments` and `gh pr diff <number>` for the diff.
- **List external PRs for triage**: `gh pr list --repo ACGNnsj/Wan2GP --state open --json number,title,body,labels,author,authorAssociation,comments` then keep only `authorAssociation` of `CONTRIBUTOR`, `FIRST_TIME_CONTRIBUTOR`, or `NONE` (drop `OWNER`/`MEMBER`/`COLLABORATOR`).
- **Comment / label / close**: `gh pr comment`, `gh pr edit --add-label`/`--remove-label`, `gh pr close`.

GitHub shares one number space across issues and PRs, so a bare `#42` may be either: resolve with `gh pr view 42` and fall back to `gh issue view 42`.

## When a skill says "publish to the issue tracker"

Create a GitHub issue in `ACGNnsj/Wan2GP`.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --repo ACGNnsj/Wan2GP --comments`.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: a single issue labelled `wayfinder:map`, holding the Notes / Decisions-so-far / Fog body. `gh issue create --repo ACGNnsj/Wan2GP --label wayfinder:map`.
- **Child ticket**: an issue linked to the map as a GitHub sub-issue (`gh api` on the sub-issues endpoint). Where sub-issues aren't enabled, add the child to a task list in the map body and put `Part of #<map>` at the top of the child body. Labels: `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`). Once claimed, the ticket is assigned to the driving dev.
- **Blocking**: GitHub's **native issue dependencies**, the canonical, UI-visible representation. Add an edge with `gh api --method POST repos/ACGNnsj/Wan2GP/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`, where `<blocker-db-id>` is the blocker's numeric **database id** (`gh api repos/ACGNnsj/Wan2GP/issues/<n> --jq .id`, _not_ the `#number` or `node_id`). GitHub reports `issue_dependencies_summary.blocked_by` (open blockers only, the live gate). Where dependencies aren't available, fall back to a `Blocked by: #<n>, #<n>` line at the top of the child body. A ticket is unblocked when every blocker is closed.
- **Frontier query**: list the map's open children (`gh issue list --repo ACGNnsj/Wan2GP --state open`, scoped to the map's sub-issues / task list), drop any with an open blocker (`issue_dependencies_summary.blocked_by > 0`, or an open issue in the `Blocked by` line) or an assignee; first in map order wins.
- **Claim**: `gh issue edit <n> --repo ACGNnsj/Wan2GP --add-assignee @me`, the session's first write.
- **Resolve**: `gh issue comment <n> --body "<answer>"`, then `gh issue close <n>`, then append a context pointer (gist + link) to the map's Decisions-so-far.

## Local environment note

- Linux `gh` is installed at `~/.local/bin/gh` (v2.102.0, authenticated as `ACGNnsj` via `~/.config/gh/hosts.yml`). `HTTPS_PROXY=http://127.0.0.1:7897` is already exported in this WSL environment.
- A Windows `gh` also exists (`C:\Users\ASUS\AppData\Local\Programs\GitHub CLI\bin\gh.exe`) for the deck plugin; prefer the Linux one for shell work.
- GitHub Issues were disabled on the fork; they are now enabled (`has_issues: true`).
