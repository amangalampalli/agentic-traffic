"""In-memory replay cache for the OpenEnv API.

Runs a full CityFlow simulation on demand and caches the result so repeated
requests for the same (city, scenario, policy) triple are served instantly.

Concurrency design
------------------
A global dict-level lock (``_registry_lock``) protects only the
``_in_flight`` and ``_cache`` dicts.  The actual simulation runs *outside*
any lock, guarded by a per-key ``threading.Event``.  This means:

- Two requests for the **same** key: the second waits on the Event; only
  one simulation runs.
- Two requests for **different** keys: both simulations run in parallel.
"""
from __future__ import annotations

import json
import logging
import tempfile
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

# Completed results: key → (replay_text, roadnet_log_dict, metrics_dict)
_cache: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {}

# In-flight simulations: key → Event that is set() once the result is cached.
_in_flight: dict[str, threading.Event] = {}

# Lock protecting _cache and _in_flight (held only for dict reads/writes).
_registry_lock = threading.Lock()


def _cache_key(city_id: str, scenario_name: str, policy_name: str) -> str:
    return f"{city_id}/{scenario_name}/{policy_name}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_cached(
    city_id: str, scenario_name: str, policy_name: str
) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    with _registry_lock:
        return _cache.get(_cache_key(city_id, scenario_name, policy_name))


def run_and_cache(
    city_id: str,
    scenario_name: str,
    policy_name: str,
    generated_root: Path,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Return (replay_text, roadnet_log, metrics), running the simulation if needed."""
    key = _cache_key(city_id, scenario_name, policy_name)

    while True:
        with _registry_lock:
            if key in _cache:
                return _cache[key]

            if key in _in_flight:
                event = _in_flight[key]
            else:
                event = threading.Event()
                _in_flight[key] = event
                event = None  # sentinel: we are the runner, not a waiter

        if event is not None:
            event.wait()
            continue

        # Runner path: execute the simulation outside all locks.
        try:
            result_tuple = _run_simulation(city_id, scenario_name, policy_name, generated_root)
        except Exception:
            with _registry_lock:
                _in_flight.pop(key, None)
            raise

        with _registry_lock:
            _cache[key] = result_tuple
            done_event = _in_flight.pop(key)

        done_event.set()
        return result_tuple


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _run_simulation(
    city_id: str,
    scenario_name: str,
    policy_name: str,
    generated_root: Path,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Run one full episode and read results into memory."""
    from server.policy_runner import run_policy_for_city

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        result = run_policy_for_city(
            city_id=city_id,
            scenario_name=scenario_name,
            policy_name=policy_name,
            generated_root=generated_root,
            output_root=tmp_path,
        )

        if not result.replay_path.exists():
            raise FileNotFoundError(
                f"CityFlow did not write a replay file for "
                f"{city_id}/{scenario_name}/{policy_name}."
            )

        replay_text = result.replay_path.read_text(encoding="utf-8")
        roadnet_log: dict[str, Any] = (
            json.loads(result.roadnet_log_path.read_text(encoding="utf-8"))
            if result.roadnet_log_path.exists()
            else {}
        )

    logger.info("Simulation complete: %s/%s/%s", city_id, scenario_name, policy_name)
    return replay_text, roadnet_log, result.metrics
