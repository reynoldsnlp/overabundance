"""Entry point placeholder.

This project now uses *two* runner scripts because the required transformers
versions are incompatible:

- answerdotai/ModernBERT-base requires transformers>=4.48
- transfo-xl-wt103 requires transformers<=4.12.1

Run one of:

- `uv run run_modernbert.py`
- `uv run run_transfoxl.py`

Both scripts share the experiment logic in `overabundance_common.py` and each
declares its own dependencies via inline script metadata (PEP 723).
"""


def main() -> None:
    print(__doc__.strip())


if __name__ == "__main__":
    main()
