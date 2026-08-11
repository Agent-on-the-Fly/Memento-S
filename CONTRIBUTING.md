# Contributing to Memento-Skills

Thank you for helping improve Memento-Skills. Bug reports, documentation fixes,
tests, and focused code changes are welcome.

## Development setup

Use Python 3.12 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`.

Set `MEMENTO_HOME` to a disposable directory while developing or testing so a
test run cannot modify a real user profile:

```bash
export MEMENTO_HOME="$PWD/.memento-test"
```

## Required local checks

Run the deterministic cross-platform smoke suite:

```bash
python scripts/check_release_readiness.py
python -m pytest -p no:cacheprovider -q \
  core/skill/tests/schema/test_skill_model.py \
  core/skill/tests/execution/loop/test_loop_detector.py \
  core/skill/tests/execution/state/test_react_state.py \
  tests/test_runtime_mode_paths.py
```

The GitHub workflow installs only the dependencies imported by this smoke suite;
the editable developer installation above includes the complete runtime.

Tests that need network access, credentials, a live LLM, platform services, or
external executables must use the `integration` or `slow` marker and must not be
added to the deterministic smoke job.

## Pull requests

- Keep each pull request focused and describe its user-visible effect.
- Add or update tests for changed behavior.
- Update documentation and third-party notices when dependencies or bundled
  assets change.
- Confirm the cross-platform CI and distribution inspection jobs pass.

## Third-party intake policy

Do not vendor code, prompts, schemas, models, media, or other assets unless all
of the following are recorded and reviewed:

- an OSI-approved license or other terms demonstrably compatible with this
  Apache-2.0 distribution;
- the upstream project URL and an exact commit, tag, or content hash;
- the applicable copyright and license text;
- a list of local modifications; and
- an entry in `THIRD_PARTY_NOTICES.md`.

If redistribution rights are unclear, use an optional external dependency or
leave the component out of the repository.

Unless stated otherwise, contributions are submitted under the Apache License
2.0 that covers this project.
