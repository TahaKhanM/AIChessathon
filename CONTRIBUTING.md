# Contributing

Use CPython 3.12 and install `.[dev]` in a virtual environment. Run `make lint` and `make test` before submitting a change. `make format` applies the shared formatter.

Keep each commit focused on one reviewable change. Describe the observable behaviour and the validation performed. Preserve actual commit dates and component attribution. Separate measured results from hypotheses; include source/model identity and machine details with benchmarks.

For engine changes, check state restoration, draw semantics and abort paths. For numerical changes, preserve exact rounding and export/runtime parity. For data changes, check label perspective and split leakage. Never commit credentials, checkpoints, generated datasets or provider inventories.

The root adapter is intentionally a weights-free baseline. A neural model release needs its own provenance, checksums and complete-package qualification. The README documents the retained CPU reference trainer for reproducible user experiments.

The public history starts with a curated source import. Subsequent commits record actual maintenance work; earlier experimental chronology is not reconstructed.
