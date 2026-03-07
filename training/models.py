from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from env.intersection_config import DISTRICT_TYPES

POLICY_ARCHES: tuple[str, ...] = (
    "multi_head",
    "single_head",
    "single_head_with_district_feature",
)


class TrafficControlQNetwork(nn.Module):
    """Parameter-shared dueling Q-network for intersection-level control."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 2,
        district_types: tuple[str, ...] = DISTRICT_TYPES,
        policy_arch: str = "single_head_with_district_feature",
        dueling: bool = True,
    ):
        super().__init__()
        if policy_arch not in POLICY_ARCHES:
            raise ValueError(
                f"Unsupported policy architecture: {policy_arch}. "
                f"Expected one of {POLICY_ARCHES}."
            )

        layers: list[nn.Module] = []
        input_dim = observation_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            input_dim = hidden_dim

        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.district_types = tuple(district_types)
        self.policy_arch = policy_arch
        self.dueling = bool(dueling)

        self.backbone = nn.Sequential(*layers)
        if self.policy_arch == "multi_head":
            self.advantage_heads = nn.ModuleList(
                [nn.Linear(hidden_dim, action_dim) for _ in self.district_types]
            )
            self.value_heads = nn.ModuleList(
                [nn.Linear(hidden_dim, 1) for _ in self.district_types]
            )
            self.advantage_head = None
            self.value_head = None
        else:
            self.advantage_heads = None
            self.value_heads = None
            self.advantage_head = nn.Linear(hidden_dim, action_dim)
            self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        observations: torch.Tensor,
        district_type_indices: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.backbone(observations)
        advantages, values = self._q_streams(features, district_type_indices)
        if self.dueling:
            q_values = values + advantages - advantages.mean(dim=-1, keepdim=True)
        else:
            q_values = advantages
        if action_mask is not None:
            q_values = self._apply_action_mask(q_values, action_mask)
        return q_values

    def act(
        self,
        observations: torch.Tensor,
        district_type_indices: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
        epsilon: float = 0.0,
    ) -> torch.Tensor:
        q_values = self.forward(
            observations=observations,
            district_type_indices=district_type_indices,
            action_mask=action_mask,
        )
        greedy_actions = q_values.argmax(dim=-1)
        if deterministic or epsilon <= 0.0:
            return greedy_actions

        random_mask = torch.rand(greedy_actions.shape[0], device=greedy_actions.device) < float(
            epsilon
        )
        if not random_mask.any():
            return greedy_actions

        actions = greedy_actions.clone()
        valid_action_mask = (
            action_mask if action_mask is not None else torch.ones_like(q_values, dtype=torch.float32)
        )
        random_rows = torch.nonzero(random_mask, as_tuple=False).flatten()
        for row_index in random_rows.tolist():
            valid_actions = torch.nonzero(valid_action_mask[row_index] > 0.0, as_tuple=False).flatten()
            if valid_actions.numel() == 0:
                actions[row_index] = 0
                continue
            sample_index = torch.randint(
                low=0,
                high=valid_actions.numel(),
                size=(1,),
                device=actions.device,
            )
            actions[row_index] = valid_actions[sample_index].item()
        return actions

    def q_values_for_actions(
        self,
        observations: torch.Tensor,
        district_type_indices: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_values = self.forward(
            observations=observations,
            district_type_indices=district_type_indices,
            action_mask=action_mask,
        )
        return q_values.gather(dim=1, index=actions.view(-1, 1)).squeeze(1)

    def _q_streams(
        self,
        features: torch.Tensor,
        district_type_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.policy_arch == "multi_head":
            all_advantages = torch.stack(
                [head(features) for head in self.advantage_heads],
                dim=1,
            )
            all_values = torch.stack(
                [head(features) for head in self.value_heads],
                dim=1,
            )
            gather_adv = district_type_indices.view(-1, 1, 1).expand(-1, 1, self.action_dim)
            gather_val = district_type_indices.view(-1, 1, 1)
            advantages = all_advantages.gather(dim=1, index=gather_adv).squeeze(1)
            values = all_values.gather(dim=1, index=gather_val).squeeze(1)
            return advantages, values

        return self.advantage_head(features), self.value_head(features)

    def _apply_action_mask(
        self,
        q_values: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        masked_q_values = q_values.masked_fill(action_mask <= 0.0, -1.0e9)
        all_invalid = action_mask.sum(dim=-1) <= 0.0
        if all_invalid.any():
            masked_q_values[all_invalid, 0] = 0.0
        return masked_q_values


@dataclass
class RunningNormalizer:
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        self.count = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None

    def update(self, batch: np.ndarray) -> None:
        array = np.asarray(batch, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("Normalizer expects a 2D batch of observations.")

        if self.mean is None:
            self.mean = np.zeros(array.shape[1], dtype=np.float64)
            self.m2 = np.zeros(array.shape[1], dtype=np.float64)

        for row in array:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            delta2 = row - self.mean
            self.m2 += delta * delta2

    def normalize(self, batch: np.ndarray) -> np.ndarray:
        array = np.asarray(batch, dtype=np.float32)
        if self.mean is None or self.m2 is None or self.count < 2:
            return array

        variance = self.m2 / max(1, self.count - 1)
        std = np.sqrt(np.maximum(variance, self.epsilon))
        return ((array - self.mean.astype(np.float32)) / std.astype(np.float32)).astype(
            np.float32
        )

    def state_dict(self) -> dict:
        return {
            "count": self.count,
            "mean": self.mean,
            "m2": self.m2,
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.count = int(state_dict["count"])
        self.mean = state_dict["mean"]
        self.m2 = state_dict["m2"]
        self.epsilon = float(state_dict.get("epsilon", self.epsilon))
