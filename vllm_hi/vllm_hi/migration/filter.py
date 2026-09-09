# SPDX-License-Identifier: Apache-2.0
"""Request-filtering extension point.

During migration you can filter "which requests to migrate" by a custom rule
(e.g. long requests whose length exceeds a threshold). The default implementation
:class:`MigrateAllFilter` migrates all requests; specify a custom class's fully
qualified name (``module.Class``) via the ``VLLM_HI_REQUEST_FILTER`` environment
variable to extend it.
Filtered-out requests have no KV collected and are not migrated; they continue
to be processed on the A instance.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any


class RequestFilter(ABC):
    """Request-filter interface.

    Implement ``should_migrate`` to return True when the request should migrate.
    """

    @abstractmethod
    def should_migrate(self, seq_id: str, request_state: Any) -> bool:
        """Decide whether a request should migrate.

        Args:
            seq_id: Request (sequence) id.
            request_state: :class:`vllm_hi.migration.state_snapshot.RequestState`.
        """
        raise NotImplementedError


class MigrateAllFilter(RequestFilter):
    """Default filter: migrate all (every request migrates)."""

    def should_migrate(self, seq_id: str, request_state: Any) -> bool:
        return True


def load_request_filter(class_path: str | None) -> RequestFilter:
    """Load the request filter by fully qualified class name; fall back to the default migrate-all when unspecified or on load failure."""
    if not class_path:
        return MigrateAllFilter()
    try:
        module_name, _, cls_name = class_path.rpartition(".")
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        instance = cls()
        if not isinstance(instance, RequestFilter):
            raise TypeError(f"{class_path!r} does not implement the RequestFilter interface")
        return instance
    except Exception as exc:  # noqa: BLE001
        from vllm.logger import init_logger

        logger = init_logger("vllm.vllm_hi.migration.filter")
        logger.warning(
            "[vllm_hi Filter] failed to load request filter %r, falling back to migrate-all: %s",
            class_path, exc,
        )
        return MigrateAllFilter()


__all__ = ["RequestFilter", "MigrateAllFilter", "load_request_filter"]
