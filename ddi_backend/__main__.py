# Module docstring: says this file makes `python -m ddi_backend ...` start the command-line interface
"""Lets ``python -m ddi_backend ...`` run the command-line interface."""

import sys  # needed to hand the CLI's return value to the operating system as the exit code

from ddi_backend.cli import main  # the command-line entry point defined in cli.py

sys.exit(main())  # run the CLI and exit with its status code (0 = success)

# ---------------------------------------------------------------------------
# FILE SUMMARY: Entry shim that runs when someone types `python -m ddi_backend`. It does no work itself; it just calls
# the command-line interface in cli.py and passes its exit code back to the shell.
# ASSIGNMENT TOOL: supporting infrastructure (not one of the assignment tools): it is the front door to the
# deterministic core (knowledge graph and rules), the guarded agent and the optional language model and embedding tiers.
# ---------------------------------------------------------------------------
