"""Backward-compatible forwarding shim -- get_activations.py moved to
probing/probe/get_activations.py. Delete this file once every sibling in
prompt_probes/pipeline/ has been migrated (tracked as this plan's Task 7)."""
import sys

from probing.probe import get_activations as _get_activations

sys.modules[__name__] = _get_activations
