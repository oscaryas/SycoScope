"""Load Colab Secrets into the kernel's os.environ (never printed). Use via
`%run` from a notebook cell so google.colab.userdata has frontend access and
the variables persist for subsequently launched `!` subprocesses.
Secrets: HF_TOKEN (gated HF repos), ANTHROPIC_API_KEY (Claude judges),
GITHUB_TOKEN (optional; lets the VM push results back to a branch)."""
import os

from google.colab import userdata

status = {}
for key in ("HF_TOKEN", "ANTHROPIC_API_KEY", "GITHUB_TOKEN"):
    try:
        value = userdata.get(key)
        if value:
            os.environ[key] = value
        status[key] = bool(value)
    except Exception as e:  # SecretNotFoundError / NotebookAccessError
        status[key] = f"missing ({type(e).__name__})"
print("secrets:", status)
