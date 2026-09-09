# SPDX-License-Identifier: Apache-2.0
"""B-side resumption: register output_processor for migrated requests and warm up the detokenizer.

After B restores scheduler state, the requests continue to be scheduled and
generated in the engine core, but the AsyncLLM frontend's ``output_processor``
does not know these request_ids, so output is silently dropped. For each restored
request this module:

1. Builds an EngineCoreRequest and registers it with the output_processor (with detokenizer/logprobs);
2. Warms up the detokenizer to the ``num_computed_tokens`` state (feeding the already-generated output_token_ids);
3. Sets ``_last_output_text_offset`` so only new tokens are emitted afterward;
4. Returns ``request_id -> RequestOutputCollector`` for the resumption endpoint to consume.
"""

from __future__ import annotations

import time
from typing import Any

from vllm.logger import init_logger

from vllm_hi.common.utils import deserialize_sampling_params

logger = init_logger("vllm.vllm_hi.migration.resume")


def register_resumed_outputs(
    engine: Any,
    request_states: dict[str, Any],
) -> dict[str, Any]:
    """Register output_processor for restored requests, returning ``request_id -> queue``.

    Args:
        engine: The AsyncLLM instance (after B-side instantiation).
        request_states: The restored request states ``{seq_id: RequestState}`` (dataclass).

    Returns:
        Mapping ``request_id -> RequestOutputCollector``.
    """
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.output_processor import RequestOutputCollector

    output_processor = getattr(engine, "output_processor", None)
    if output_processor is None:
        logger.warning("[vllm_hi resume] engine.output_processor is unavailable")
        return {}

    queues: dict[str, Any] = {}
    for req_id, rs in request_states.items():
        try:
            # Finished requests need no resumption (their output was already fully returned); skip to shorten switch time
            status = str(getattr(rs, "status", "")).upper()
            if "FINISH" in status:
                continue
            sampling_params = deserialize_sampling_params(
                getattr(rs, "sampling_params", None)
            )
            prompt_token_ids = list(getattr(rs, "prompt_token_ids", None) or [])
            output_token_ids = list(getattr(rs, "output_token_ids", None) or [])

            if sampling_params is None:
                logger.warning("[vllm_hi resume] request %s has no sampling_params, skipped", req_id)
                continue

            ecr = EngineCoreRequest(
                request_id=req_id,
                prompt_token_ids=prompt_token_ids,
                mm_features=None,
                sampling_params=sampling_params,
                pooling_params=None,
                arrival_time=time.time(),
                lora_request=None,
                cache_salt=None,
                data_parallel_rank=None,
                external_req_id=req_id,
            )

            queue = RequestOutputCollector(sampling_params.output_kind, req_id)
            output_processor.add_request(
                ecr, prompt=None, parent_req=None, request_index=0, queue=queue
            )

            req_state = output_processor.request_states.get(req_id)
            if req_state is None:
                logger.warning("[vllm_hi resume] RequestState missing after registering %s", req_id)
                continue

            det = req_state.detokenizer
            if det is not None and output_token_ids:
                det.update(list(output_token_ids), stop_terminated=False)
                det._last_output_text_offset = len(det.output_text)
                req_state.is_prefilling = False
                req_state.sent_tokens_offset = det.num_output_tokens()

            queues[req_id] = queue
        except Exception as exc:  # noqa: BLE001
            logger.warning("[vllm_hi resume] failed to register request %s: %s", req_id, exc)

    logger.info(
        "[vllm_hi resume] registered output consumers for %d restored requests", len(queues)
    )
    return queues