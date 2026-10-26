# Publishing a snapshot without development history

The public distribution is prepared in a separate, fresh Git repository. Its
`main` and `opensource` branches should point to the same single root commit.
The original local checkout and all of its development branches remain intact.
Do not use `git push --mirror` from the development repository: that would send
its other branches and history as well.

Before publishing, verify that the fresh tree contains only the reviewed public
files, run `bash scripts/verify.sh` from it and scan it for credentials. Exclude
`.env`, local robot binaries, caches, staging archives and generated campaigns.
Preserve attribution and third-party notices even when Git history is omitted.
The data's provenance IDs and chained integrity manifests are dataset metadata;
they do not make old Git commits reachable from the public snapshot.

Inspect remote heads and tags immediately before updating them. Use one atomic
push with explicit `--force-with-lease=refs/heads/NAME:EXPECTED_SHA` for each of
`main` and `opensource`. The expected SHAs must be the heads just reviewed; if
they change, stop and inspect the new work. Do not force around branch protection.
Review any existing tags or additional branches separately because they can
retain old history. No tag or branch deletion is implied by this maintenance.

Changing branch tips does not guarantee removal from a hosting service's caches,
forks, pull-request refs or artifacts. For this reason, a clean public tree and
secret review are required even when publishing without the development log.

After the first approved publication, make future reviewed changes on top of
that public root. There is no need to squash and force-push every routine update.
GitHub branch rules, private vulnerability reporting and the first semver tag
are repository settings/release operations, separate from local file preparation.
