"""Backward-compatible forwarding shim -- train_probes.py moved to
probing/probe/train_probes.py. Delete this file once every sibling in
prompt_probes/pipeline/ has been migrated (tracked as this plan's Task 7)."""
import sys

from probing.probe import train_probes as _train_probes

sys.modules[__name__] = _train_probes
