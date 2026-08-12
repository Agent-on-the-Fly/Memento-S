# Third-Party Notices

Memento-Skills is distributed under the Apache License 2.0, except for the
separately identified components below. A component's own license continues to
govern that component.

## Bundled components

### Anthropic skill-creator

`builtin/skills/skill-creator/` is derived from Anthropic's `skill-creator`:

- Upstream: <https://github.com/anthropics/skills/tree/main/skills/skill-creator>
- Imported revision: `b0cbd3df1533b396d281a6886d5132f623393a9c`
- Copyright: 2026 Anthropic, PBC
- License: Apache License 2.0
- Local license: `builtin/skills/skill-creator/LICENSE.txt`

The local copy modifies `SKILL.md`, `scripts/improve_description.py`,
`scripts/quick_validate.py`, and `scripts/run_loop.py` relative to that revision.

### SheetJS Community Edition

`builtin/skills/skill-creator/eval-viewer/viewer.html` optionally loads SheetJS
Community Edition 0.20.3 from `cdn.sheetjs.com` to render spreadsheet results.
The script is fetched at runtime and is not bundled in this repository.

- Project: <https://sheetjs.com/>
- Copyright: 2012-present SheetJS LLC
- License: Apache License 2.0
- License information: <https://docs.sheetjs.com/docs/miscellany/license/>

### Poppins and Lora fonts

The skill-creator report viewer optionally loads Poppins and Lora from Google
Fonts. Font files are fetched at runtime and are not bundled in this repository.
Both families are distributed under the SIL Open Font License 1.1; their
copyright and license metadata are available from the Google Fonts catalog.

## Optional components not distributed here

The personal-WeChat integration imports a `weixin_sdk` package when one is
installed separately. No copy of that SDK is distributed with Memento-Skills.
Users are responsible for reviewing and complying with the terms of any SDK they
choose to install.

Document, PDF, presentation, and spreadsheet skills previously present in this
repository are not part of this Apache-2.0 distribution. External skills may be
installed only after their licenses and provenance have been reviewed.

## Python dependencies

Python packages installed from package indexes are not vendored in the source
repository. They remain subject to their respective licenses. Binary release
maintainers must generate and review a dependency notice or software bill of
materials for the exact locked dependency set before publication.
