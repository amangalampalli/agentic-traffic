from __future__ import annotations

import argparse
from typing import Any, Callable

from district_llm.prompting import format_district_prompt
from district_llm.schema import DistrictAction, DistrictStateSummary
from district_llm.summary_builder import DistrictStateSummaryBuilder
from env.observation_builder import ObservationConfig
from env.reward import RewardConfig
from env.traffic_env import EnvConfig, TrafficEnv
from training.dataset import CityFlowDataset


def _extract_json_object(payload: str) -> str:
    start = payload.find("{")
    end = payload.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return payload[start : end + 1]


class DistrictLLMInference:
    def __init__(
        self,
        generator_fn: Callable[[str], str] | None = None,
        model_name_or_path: str | None = None,
        device: str | None = None,
        fallback_action: DistrictAction | None = None,
    ):
        self.fallback_action = fallback_action or DistrictAction.default_hold()
        self.generator_fn = generator_fn
        self.tokenizer = None
        self.model = None
        self.device = device or "cpu"

        if self.generator_fn is None:
            if not model_name_or_path:
                raise ValueError("Provide either generator_fn or model_name_or_path.")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(self.device)
            self.model.eval()

    def generate_raw(self, prompt: str, max_new_tokens: int = 128) -> str:
        if self.generator_fn is not None:
            return self.generator_fn(prompt)
        import torch

        assert self.model is not None and self.tokenizer is not None
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def parse_action(self, payload: str) -> DistrictAction:
        try:
            return DistrictAction.from_json(_extract_json_object(payload))
        except Exception:
            return self.fallback_action

    def predict(self, summary: DistrictStateSummary, max_new_tokens: int = 128) -> DistrictAction:
        prompt = format_district_prompt(summary)
        raw = self.generate_raw(prompt=prompt, max_new_tokens=max_new_tokens)
        return self.parse_action(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-sample district LLM inference.")
    parser.add_argument("--model", required=True, help="Model name, local path, or LoRA adapter path.")
    parser.add_argument("--generated-root", default="data/generated")
    parser.add_argument("--splits-root", default="data/splits")
    parser.add_argument("--city-id", required=True)
    parser.add_argument("--scenario-name", required=True)
    parser.add_argument("--district-id", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def build_env(scenario_spec) -> TrafficEnv:
    env_config = EnvConfig(
        simulator_interval=1,
        decision_interval=5,
        min_green_time=10,
        thread_num=1,
        observation=ObservationConfig(),
        reward=RewardConfig(variant="wait_queue_throughput"),
    )
    return TrafficEnv(
        city_id=scenario_spec.city_id,
        scenario_name=scenario_spec.scenario_name,
        city_dir=scenario_spec.city_dir,
        scenario_dir=scenario_spec.scenario_dir,
        config_path=scenario_spec.config_path,
        roadnet_path=scenario_spec.roadnet_path,
        district_map_path=scenario_spec.district_map_path,
        metadata_path=scenario_spec.metadata_path,
        env_config=env_config,
    )


def main() -> None:
    args = parse_args()
    dataset = CityFlowDataset(
        generated_root=args.generated_root,
        splits_root=args.splits_root,
    )
    scenario_spec = dataset.build_scenario_spec(args.city_id, args.scenario_name)
    env = build_env(scenario_spec)
    summary_builder = DistrictStateSummaryBuilder()
    observation_batch = env.reset()
    summaries = summary_builder.build_all(env, observation_batch)
    if args.district_id not in summaries:
        raise ValueError(f"Unknown district_id '{args.district_id}' for {args.city_id}/{args.scenario_name}.")
    inference = DistrictLLMInference(
        model_name_or_path=args.model,
        device=args.device,
        fallback_action=DistrictAction.default_hold(),
    )
    action = inference.predict(
        summary=summaries[args.district_id],
        max_new_tokens=args.max_new_tokens,
    )
    print(action.to_pretty_json())


if __name__ == "__main__":
    main()
