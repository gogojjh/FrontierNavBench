# Publishing a snapshot without development history

The public distribution starts from a fresh root commit, without development
history. `main` is the only maintained branch, both locally and on GitHub. All
published changes from the former `opensource` branch are included in `main`.
Older development history is retained in a verified local Git bundle outside
the tracked files. It is not published as extra branches or tags.

Before publishing, verify that the fresh tree contains only the reviewed public
files, run `bash scripts/verify.sh` from it and scan it for credentials. Exclude
`.env`, local robot binaries, caches, staging archives and generated campaigns.
Preserve attribution and third-party notices even when Git history is omitted.
The data's provenance IDs and chained integrity manifests are dataset metadata;
they do not make old Git commits reachable from the public snapshot.

Inspect remote heads and tags immediately before updating them. Routine changes
fast-forward `main`. When explicitly authorized to remove extra branches, first
verify their changes are included in `main` or preserved in the local backup;
then update `main` and delete the obsolete remote heads atomically, using exact
`--force-with-lease=refs/heads/NAME:EXPECTED_SHA` checks for every affected head.
If a head changes, inspect the new work before retrying. Do not force around
branch protection or use `git push --mirror` from an archive.

Scheduled Dependabot version updates are disabled to avoid automatically creating
new branches. Dependencies and pinned CI actions can still be updated manually
on `main`. CI validates pushes to `main` and remains available for pull requests
and manual runs.

Changing branch tips does not guarantee removal from a hosting service's caches,
forks, pull-request refs or artifacts. For this reason, a clean public tree and
secret review are required even when publishing without the development log.

After the first approved publication, make future reviewed changes on top of
that public root. There is no need to squash and force-push every routine update.
GitHub branch rules, private vulnerability reporting and the first semver tag
are repository settings/release operations, separate from local file preparation.
