"""
vLLM attention backend stub — import after vLLM is on PYTHONPATH.

Usage (future):
  export VLLM_ATTENTION_BACKEND=LUXI_WALLER
  python -m vllm.entrypoints.openai.api_server ...
"""

from __future__ import annotations

BACKEND_NAME = "LUXI_WALLER"


def get_attention_impl():
    """Return callable compatible with vLLM custom attention hook (stub)."""
    try:
        import sys
        from pathlib import Path

        torch_path = Path(__file__).resolve().parents[1] / "torch"
        sys.path.insert(0, str(torch_path))
        from waller_attention import waller_attention_torch  # noqa: F401

        return waller_attention_torch
    except ImportError as e:
        raise RuntimeError(
            f"{BACKEND_NAME} backend requires torch + integrations/torch on PYTHONPATH"
        ) from e


if __name__ == "__main__":
    print(f"{BACKEND_NAME} stub OK — impl={get_attention_impl()}")