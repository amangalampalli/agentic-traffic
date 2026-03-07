from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from agents.local_policy import BaseLocalPolicy
from training.models import RunningNormalizer, TrafficControlQNetwork


class TorchLocalPolicyProtocol(Protocol):
    def act(
        self,
        observations: torch.Tensor,
        district_type_indices: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
        epsilon: float = 0.0,
    ) -> torch.Tensor:
        ...


def evaluate_policy(
    env_factory,
    actor: TrafficControlQNetwork | BaseLocalPolicy | TorchLocalPolicyProtocol,
    device: torch.device | None = None,
    obs_normalizer: RunningNormalizer | None = None,
    deterministic: bool = True,
) -> dict[str, float | str]:
    env = env_factory()
    observation_batch = env.reset()
    done = False
    final_info = env.last_info

    while not done:
        if isinstance(actor, BaseLocalPolicy):
            actions = actor.act(observation_batch)
        else:
            raw_obs = observation_batch["observations"].astype(np.float32)
            normalized_obs = obs_normalizer.normalize(raw_obs) if obs_normalizer else raw_obs
            obs_tensor = torch.as_tensor(normalized_obs, dtype=torch.float32, device=device)
            district_type_tensor = torch.as_tensor(
                observation_batch["district_type_indices"],
                dtype=torch.int64,
                device=device,
            )
            action_mask_tensor = torch.as_tensor(
                observation_batch["action_mask"],
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                action_tensor = actor.act(
                    observations=obs_tensor,
                    district_type_indices=district_type_tensor,
                    action_mask=action_mask_tensor,
                    deterministic=deterministic,
                    epsilon=0.0,
                )
            actions = action_tensor.cpu().numpy()

        observation_batch, _, done, final_info = env.step(actions)

    metrics = {
        key: float(value)
        for key, value in final_info["metrics"].items()
        if value is not None and isinstance(value, (int, float))
    }
    metrics.update(
        {
            "city_id": env.city_id,
            "scenario_name": env.scenario_name,
            "episode_return": float(env.episode_return),
            "total_episode_return": float(env.total_episode_return),
            "decision_steps": float(env.decision_step_count),
        }
    )
    return metrics
