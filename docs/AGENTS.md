# AGENTS Guide for docs/

This file applies to everything under `docs/`.

## Purpose

- Keep documentation changes accurate, minimal, and easy to review.
- Prefer docs-only edits unless a task explicitly requires source-code changes.

## Documentation

```bash
# From docs/
make html-quick   # Fast build (skips API and gallery)
make html-noplot  # Skip gallery only
make html-noapi   # Skip API only
make html         # Full build (warnings are errors)
make doctest      # Run doctests
make clean        # Remove build artefacts
make show         # Open built docs in browser
```

Gallery example tests (from repo root):

```bash
pytest -v docs/gallery_tests/test_gallery_examples.py
```

## Key Locations

- `docs/src/`: Sphinx source content.
- `docs/gallery_code/`: Gallery example scripts.
- `docs/gallery_tests/`: Tests for gallery examples.

## Editing Rules

- Keep changes focused; avoid unrelated refactors and formatting churn.
- Follow existing reStructuredText style and section structure in nearby files.
- Preserve cross-references (`:ref:`, `:doc:`, links) and update targets when moving content.
- Prefer incremental edits to existing pages over creating new top-level pages.

## Gallery Rules

- Add gallery examples in `docs/gallery_code/<section>/`.
- Name new gallery scripts with a `plot_` prefix.
- Keep gallery examples deterministic and testable.

## Safety Checklist Before Finishing

- The relevant docs build command succeeds.
- Any changed gallery example has a passing related test.
- No broken links or references introduced by the change.

## ⚠️ Meta-Instruction: Auto-Update Rule
- **Trigger**: You MUST automatically update this `AGENTS.md` file if a new structural pattern, critical bug fix, or persistent repository rule is established during this session.
- **Constraint 1**: Keep the file under 200 lines. Delete obsolete rules to make room for new ones.
- **Constraint 2**: Do not ask for user permission before updating. Perform the edit silently as your final step.
- **Constraint 3**: Only log global, reusable lessons. Do not log temporary or component-specific fixes.

---

**Last Updated**: 16 June 2026  

