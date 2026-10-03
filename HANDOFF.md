# Handoff: major-overhaul branch

## Scope

Branch: `refactor/major-overhaul`

This branch updates the Gradio administration workflow around one active project and adds PDF-level preprocessing and single-turn Q&A collection.

## Changes

- Project Settings can create, delete, select, and open projects. The opened project is shared by the remaining admin pages; per-page project selectors were removed.
- Connection Settings loads API keys from environment variables, saved settings, or `.env` files. Saved `PROJECTS_ROOT/.connection.json` values now take precedence over dotenv fallbacks, so a stale project `.env` no longer masks a newly saved key after restart.
- PDF preprocessing settings are per document: header/footer ignore percentages and inclusive, zero-based start/end page indexes. Defaults are 0%, page 0 through the last page.
- GraphRAG indexing output is streamed to the page and the log follows new output to the bottom.
- The Q&A Test page is single-question only. Its answer is editable; users can create or load a project question set and save the current question plus edited answer. New entries store both `answer` and `reference_answer`, and cannot be added without a non-empty answer. Question-set controls appear above Evidence details.
- Fixed callback return-count and sampling-dropdown type errors encountered while opening projects.

## Verification performed

- Python syntax compilation for changed application, service, and test modules.
- Gradio app construction and direct callback arity checks.
- Focused persistence checks for API key reload precedence and edited Q&A answers.
- `git diff --check`.

The full test suite was not run. Regression tests were added/updated in `tests/test_connections.py`, `tests/test_documents.py`, `tests/test_packaging.py`, and `tests/test_question_sets.py`.

## Commit range

The branch is based on `a5d05a9` and currently includes:

- `d9ab959` feat: scope admin workflows to active project
- `4c69a96` fix: refresh sampling sections on project open
- `2702426` fix: handle PDF settings dataframe values safely
- `f1eae8a` fix: return all active project view updates
- `9808fb7` fix: prefer saved API key over dotenv fallback
- `647917f` feat: force indexing log to follow new output
- `61d5ffd` feat: collect single Q&A in question sets
- `7d68925` feat: edit answers before saving to question sets

## Follow-up

Run the full test suite and exercise the main UI flows with a real project/PDF and configured API key after deployment. Ensure the application is restarted when testing these changes.
