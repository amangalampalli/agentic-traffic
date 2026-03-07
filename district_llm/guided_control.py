from __future__ import annotations

from typing import Any

import numpy as np

from district_llm.schema import DistrictAction


class DistrictGuidedLocalController:
    """
    Wrap a low-level controller and bias its actions with district directives.

    The shared DQN still produces the base per-intersection action, and the
    district plan only nudges hold/switch decisions toward the requested phase.
    """

    def __init__(self, base_teacher):
        self.base_teacher = base_teacher

    def act(
        self,
        observation_batch: dict[str, Any],
        district_actions: dict[str, DistrictAction] | None = None,
    ) -> np.ndarray:
        base_actions = np.asarray(self.base_teacher.act(observation_batch), dtype=np.int64)
        if not district_actions:
            return base_actions

        guided_actions = base_actions.copy()
        for index, district_id in enumerate(observation_batch["district_ids"]):
            directive = district_actions.get(district_id)
            if directive is None:
                continue
            guided_actions[index] = self._apply_directive(
                observation_batch=observation_batch,
                index=index,
                base_action=int(base_actions[index]),
                directive=directive,
            )
        return guided_actions

    @staticmethod
    def _apply_directive(
        observation_batch: dict[str, Any],
        index: int,
        base_action: int,
        directive: DistrictAction,
    ) -> int:
        action_mask = observation_batch["action_mask"][index]
        current_phase = int(observation_batch["current_phase"][index])
        can_switch = bool(action_mask[1] > 0.0)

        if directive.strategy == "hold" or directive.phase_bias == "NONE":
            return int(base_action)

        if directive.phase_bias == "NS":
            if current_phase == 0:
                return 0
            return 1 if can_switch else 0

        if directive.phase_bias == "EW":
            if current_phase != 0:
                return 0
            return 1 if can_switch else 0

        return int(base_action)
