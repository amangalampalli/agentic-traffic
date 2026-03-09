from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from district_llm.inference import DistrictLLMInference
from district_llm.schema import DistrictAction
from models import (
    AgenticTrafficAction,
    AgenticTrafficObservation,
    AgenticTrafficState,
)
from openenv.core.env_server.interfaces import Environment
from openenv_app.openenv_wrapper import OpenEnvTrafficWrapper


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DATA_DIR", "") or (REPO_ROOT / "data" / "generated"))
SPLITS_DIR = Path(os.environ.get("SPLITS_DIR", "") or (REPO_ROOT / "data" / "splits"))
DISTRICT_LLM_ADAPTER_PATH = Path(
    os.environ.get("DISTRICT_LLM_ADAPTER_PATH", "")
    or (REPO_ROOT / "artifacts" / "district_llm_adapter_v3" / "main_run" / "adapter")
)
DISTRICT_LLM_DEVICE = os.environ.get("DISTRICT_LLM_DEVICE")


class AgenticTrafficEnvironment(
    Environment[AgenticTrafficAction, AgenticTrafficObservation, AgenticTrafficState]
):
    """Minimal OpenEnv-compatible wrapper around the existing district controller stack."""

    def __init__(self) -> None:
        super().__init__()
        self.wrapper = OpenEnvTrafficWrapper(
            generated_root=DATA_DIR,
            splits_root=SPLITS_DIR,
        )
        self._state = AgenticTrafficState()
        self._llm_inference: DistrictLLMInference | None = None
        self._llm_load_attempted = False
        self._llm_error: str | None = None

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> AgenticTrafficObservation:
        payload = self.wrapper.reset(
            seed=seed,
            city_id=kwargs.get("city_id"),
            scenario_name=kwargs.get("scenario_name"),
        )
        self._state.episode_id = episode_id or str(uuid.uuid4())
        self._state.step_count = 0
        self._sync_state()
        observation = AgenticTrafficObservation.model_validate(payload["observation"])
        observation.reward = None
        observation.done = False
        observation.metadata["llm"] = self._llm_status()
        return observation

    def step(
        self,
        action: AgenticTrafficAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> AgenticTrafficObservation:
        del timeout_s, kwargs
        payload = self.wrapper.step(action=self._build_step_payload(action))
        self._state.step_count += 1
        self._sync_state()
        observation = AgenticTrafficObservation.model_validate(payload["observation"])
        observation.done = bool(payload.get("done", False))
        observation.reward = float(payload.get("reward", 0.0))
        observation.metadata["llm"] = self._llm_status()
        return observation

    @property
    def state(self) -> AgenticTrafficState:
        self._sync_state()
        return self._state

    def _build_step_payload(self, action: AgenticTrafficAction) -> dict[str, Any]:
        district_actions = dict(action.district_actions)
        llm_generated_actions: dict[str, Any] = {}

        if action.use_llm:
            llm_generated_actions = self._generate_llm_actions(
                existing_actions=district_actions,
                max_new_tokens=action.llm_max_new_tokens,
            )
            for district_id, directive in llm_generated_actions.items():
                district_actions.setdefault(district_id, directive)

        payload = {"district_actions": district_actions}
        payload["metadata"] = {
            "use_llm": bool(action.use_llm),
            "llm_generated_districts": sorted(llm_generated_actions),
            "llm": self._llm_status(),
        }
        return payload

    def _generate_llm_actions(
        self,
        existing_actions: dict[str, Any],
        max_new_tokens: int,
    ) -> dict[str, Any]:
        if not self.wrapper.last_summaries:
            return {}

        inference = self._get_llm_inference()
        if inference is None:
            return {}

        generated_actions: dict[str, Any] = {}
        for district_id, summary in self.wrapper.last_summaries.items():
            if district_id in existing_actions:
                continue
            result = inference.predict_with_result(summary=summary, max_new_tokens=max_new_tokens)
            generated_actions[district_id] = result.action.to_dict()
        return generated_actions

    def _get_llm_inference(self) -> DistrictLLMInference | None:
        if self._llm_inference is not None:
            return self._llm_inference
        if self._llm_load_attempted:
            return None

        self._llm_load_attempted = True
        if not DISTRICT_LLM_ADAPTER_PATH.exists():
            self._llm_error = f"Adapter not found at {DISTRICT_LLM_ADAPTER_PATH}"
            return None

        try:
            self._llm_inference = DistrictLLMInference(
                model_name_or_path=str(DISTRICT_LLM_ADAPTER_PATH),
                device=DISTRICT_LLM_DEVICE,
                fallback_action=DistrictAction.default_hold(
                    duration_steps=self.wrapper.district_decision_interval
                ),
            )
        except Exception as exc:
            self._llm_error = f"{type(exc).__name__}: {exc}"
            self._llm_inference = None
        return self._llm_inference

    def _llm_status(self) -> dict[str, Any]:
        return {
            "adapter_path": str(DISTRICT_LLM_ADAPTER_PATH),
            "adapter_present": DISTRICT_LLM_ADAPTER_PATH.exists(),
            "loaded": self._llm_inference is not None,
            "load_attempted": self._llm_load_attempted,
            "error": self._llm_error,
        }

    def _sync_state(self) -> None:
        payload = self.wrapper.state()["state"]
        self._state = AgenticTrafficState.model_validate(
            {
                **payload,
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
                "llm": self._llm_status(),
            }
        )
