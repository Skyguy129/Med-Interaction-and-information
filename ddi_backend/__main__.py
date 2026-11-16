"""Lets ``python -m ddi_backend ...`` run the command-line interface."""

import sys

from ddi_backend.cli import main

sys.exit(main())
