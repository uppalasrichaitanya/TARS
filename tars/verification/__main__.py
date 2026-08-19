"""`python -m tars.verification` — same entry point as the `tars-verify` script."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
