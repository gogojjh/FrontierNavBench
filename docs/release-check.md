# Release preparation checks — 2026-10-26

## Scope and skills

This maintenance applies to the standalone benchmark checkout, not the surrounding
RoboHike modules. The selected skills were installed globally outside the repo:

- [repo-standardizer](https://github.com/github/awesome-copilot/tree/main/skills/repo-standardizer)
  for local documentation, templates and CI configuration.
- [open-sourcing](https://github.com/trailofbits/skills/tree/main/plugins/open-sourcing/skills/open-sourcing)
  for input, licensing, security and release-readiness checks.
- [Superpowers](https://github.com/obra/superpowers/tree/main/skills)
  for debugging, regression verification and independent review.

## Input preservation

All 566 files in the 30 baseline map views were compared byte-for-byte with the
pre-maintenance commit. All 15 numeric episode records and their goal-image
bytes remain unchanged. Only map IDs, directory paths, registry checksum and
source-binding metadata changed. The migration table and updated asset inventory
retain the original fingerprints. Frozen layers v1/v2 remain intact; v3/v4
record the reviewed maintenance and figure-provenance change.

The unused paid keyframe-selection module was removed after checking runtime
imports. Fixed active-map selections, active memory, simulator modules and
executable report tools remain. Tests for retained functionality were ported
from development history, with fake clients kept under `tests/unit/`.

## Security and licensing

Gitleaks 8.30.1 scanned all local Git history and the isolated public snapshot
with complete redaction enabled. Its official ARM64 binary checksum was verified.
The history scan produced seven heuristic matches; review identified a source
checksum, synthetic security/VLM test values and an old episode token. The public
snapshot scan produced four matches: one valid runtime SHA256 and three synthetic
test credentials. Independent review found no real credentials in these matches.
The scans have findings; they are not being described as zero-match scans.

MIT remains the project license. Existing MIT, Apache-2.0 and CC0 asset notices
remain. Official NVIDIA robot files stay excluded and must be obtained under
their own terms. No tracked source file exceeds GitHub's 100 MiB file limit.

## Validation boundaries

| Environment | Executed checks | Result |
| --- | --- | --- |
| Python 3.8, local checkout | Offline regression suite before final figure-only fix | 437 passed, no skipped tests |
| Python 3.8, final targeted follow-up | Input/manifest/figure checks and golden-replay compatibility checks | 18 passed plus 9 passed |
| Fresh Python 3.10 environment, isolated public tree | `VNAV_VERIFY_PYTHON=/tmp/vnav-release-eval310/bin/python bash scripts/verify.sh` | Ruff passed, 447 tests passed in 90.41 s, input checks and CPU smoke passed |

The fresh evaluation environment was created by `setup/evaluation.sh` using its
documented interpreter/environment overrides, then the declared `dev` extra was
installed. The isolated source tree contained no old `.git`, `.env`, staging
archive or pre-existing caches. Dependencies emitted deprecation warnings;
the validation command exited successfully. A golden replay comparison was
adapted for Python/libm differences of at most 1.11e-16: types, sequence lengths,
integer/text values and JSON structure remain exact, floats allow 1e-12 relative
and absolute tolerance, and meaningful drift has explicit rejection tests.

The CPU installation/integrity check verifies 94 runtime files, 1,007 asset
inventory entries, 30 baseline maps, 15 active maps, 15 episodes and three scenes.
Its temporary debug cell completes with `excluded_debug`, with no simulation
started and no model calls. Offline tests isolate credentials and input roots,
use fake VLM replies and reserve independent loopback sockets.

The README's new goal/map figures were reproduced from their recorded inputs;
relocated-map rendering was also checked. Preserved historical result images
remain byte-identical to their identified documentation sources and are labeled
as historical, not current performance measurements.

Local shell syntax, Markdown links, workflow YAML and pinned official Action
commits were checked. The CI definition targets Python 3.8 and 3.10. Remote
GitHub Actions execution, actionlint/zizmor checks, GPU/Isaac simulation and real
robot validation were not run by this maintenance. Remote Actions artifacts,
issue/wiki content and repository protection settings require separate review
before a visibility change. These results describe preparation before any remote
publication; publishing the verified snapshot is a separate step.

The public snapshot follows [the history policy](public-release.md). The original
local history and unrelated work in the parent workspace remain available.
