"""Backward-compatible forwarding shim -- common.py moved to
probing/utils/common.py. Delete this file once every sibling in
prompt_probes/pipeline/ has been migrated (tracked as this plan's Task 7)."""
import sys

from probing.utils import common as _common

sys.modules[__name__] = _common
