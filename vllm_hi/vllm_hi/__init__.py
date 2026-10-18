# SPDX-License-Identifier: Apache-2.0
"""vllm_hi — a package for customizing vllm.

Usage::

    import vllm_hi  # importing automatically applies all inference-time hot-migration customizations (idempotent)

Customizations include:
1. monkey-patch ``vllm.entrypoints.launchers.app.build_app``:
   - add a ``GET /vllm_hi/status`` route to the API server;
   - attach the ``X-VLLM-HI`` header to all HTTP responses.
2. monkey-patch ``OpenAIServingChat.create_chat_completion``:
   - log key information for every chat request on the server side.
3. inject engine_internals patches into every process via the
   ``vllm.general_plugins`` entry point (see ``vllm_hi.patches.plugin``),
   supporting state collection and recovery for inference-time hot migration.
"""

__version__ = "0.1.0"

from vllm_hi import patches  # noqa: F401

# Importing applies immediately (idempotent; repeated imports do not re-apply patches)
patches.apply()

__all__ = ["__version__", "patches"]
