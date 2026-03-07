from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from agents.district_coordinator import RuleBasedDistrictCoordinator
from agents.local_policy import SharedHeuristicLocalPolicy
from dashboard.metrics import flatten_directives, summarize_history
from training.rollout import run_episode


def make_env():
    from env.traffic_env import TrafficEnv
    from env.intersection_config import IntersectionConfig, DistrictConfig

    intersections = {
        "I1": IntersectionConfig(
            intersection_id="I1",
            district_id="D0",
            incoming_lanes=["I1_N", "I1_S", "I1_E", "I1_W"],
            outgoing_lanes=[],
            neighbors=["I2"],
            is_border=False,
        ),
        "I2": IntersectionConfig(
            intersection_id="I2",
            district_id="D0",
            incoming_lanes=["I2_N", "I2_S", "I2_E", "I2_W"],
            outgoing_lanes=[],
            neighbors=["I1", "I3"],
            is_border=True,
        ),
        "I3": IntersectionConfig(
            intersection_id="I3",
            district_id="D1",
            incoming_lanes=["I3_N", "I3_S", "I3_E", "I3_W"],
            outgoing_lanes=[],
            neighbors=["I2", "I4"],
            is_border=True,
        ),
        "I4": IntersectionConfig(
            intersection_id="I4",
            district_id="D1",
            incoming_lanes=["I4_N", "I4_S", "I4_E", "I4_W"],
            outgoing_lanes=[],
            neighbors=["I3"],
            is_border=False,
        ),
    }

    districts = {
        "D0": DistrictConfig(
            district_id="D0",
            intersection_ids=["I1", "I2"],
            neighbor_districts=["D1"],
        ),
        "D1": DistrictConfig(
            district_id="D1",
            intersection_ids=["I3", "I4"],
            neighbor_districts=["D0"],
        ),
    }

    return TrafficEnv(
        config_path="data/cityflow/config.json",
        intersections=intersections,
        districts=districts,
        coordination_interval=20,
        max_steps=200,
    )


def build_history_frames(history: list[dict]):
    metric_rows = []
    for row in history:
        metrics = row.get("metrics", {})
        metric_rows.append(
            {
                "step": row.get("step", 0),
                "total_waiting": float(metrics.get("total_waiting", 0.0)),
                "total_queue": float(metrics.get("total_queue", 0.0)),
                "mean_reward": float(metrics.get("mean_reward", 0.0)),
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    directives_df = pd.DataFrame(flatten_directives(history))
    return metrics_df, directives_df


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _list_generated_cities(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("city_")
    )


def _district_color_map(district_ids: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.cm.get_cmap("tab20", max(1, len(district_ids)))
    return {did: cmap(idx) for idx, did in enumerate(district_ids)}


def _plot_city_geometry(
    roadnet: dict,
    district_map: dict | None,
    show_gateways: bool,
    show_districts: bool,
    show_labels: bool,
):
    fig, ax = plt.subplots(figsize=(10, 10))
    intersections = {
        node["id"]: (float(node["point"]["x"]), float(node["point"]["y"]))
        for node in roadnet.get("intersections", [])
    }
    roads = roadnet.get("roads", [])
    intersection_to_district = (
        district_map.get("intersection_to_district", {}) if district_map else {}
    )
    district_ids = sorted(set(intersection_to_district.values()))
    district_colors = _district_color_map(district_ids) if district_ids else {}
    gateway_nodes = set(district_map.get("gateway_intersections", [])) if district_map else set()
    gateway_roads = set(district_map.get("gateway_roads", [])) if district_map else set()

    for road in roads:
        points = road.get("points", [])
        if len(points) < 2:
            continue
        x = [points[0]["x"], points[-1]["x"]]
        y = [points[0]["y"], points[-1]["y"]]
        color = "#7f8c8d"
        width = 0.8
        alpha = 0.45
        if show_gateways and road["id"] in gateway_roads:
            color = "#f39c12"
            width = 1.8
            alpha = 0.95
        elif show_districts:
            start = road.get("startIntersection")
            end = road.get("endIntersection")
            ds = intersection_to_district.get(start)
            de = intersection_to_district.get(end)
            if ds and ds == de:
                color = district_colors.get(ds, color)
                width = 1.0
                alpha = 0.7
            elif ds and de and ds != de:
                color = "#2c3e50"
                width = 1.25
                alpha = 0.9
        ax.plot(x, y, color=color, linewidth=width, alpha=alpha, solid_capstyle="round")

    if show_districts and district_colors:
        for district_id in district_ids:
            nodes = [
                nid for nid, did in intersection_to_district.items() if did == district_id and nid in intersections
            ]
            if not nodes:
                continue
            xs = [intersections[n][0] for n in nodes]
            ys = [intersections[n][1] for n in nodes]
            ax.scatter(xs, ys, s=14, color=district_colors[district_id], alpha=0.8, label=district_id)
    else:
        xs = [p[0] for p in intersections.values()]
        ys = [p[1] for p in intersections.values()]
        ax.scatter(xs, ys, s=8, color="#34495e", alpha=0.65)

    if show_gateways and gateway_nodes:
        gxs = [intersections[n][0] for n in sorted(gateway_nodes) if n in intersections]
        gys = [intersections[n][1] for n in sorted(gateway_nodes) if n in intersections]
        ax.scatter(gxs, gys, s=44, color="#d35400", edgecolors="#1c2833", linewidths=0.6, zorder=10, label="gateway")

    if show_labels:
        for nid, (x, y) in intersections.items():
            if nid in gateway_nodes:
                ax.text(x, y, nid, fontsize=6, color="#922b21")

    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Roadnet Viewer")
    ax.grid(True, alpha=0.15)
    if show_districts or show_gateways:
        ax.legend(loc="upper right", fontsize=7, frameon=True)
    return fig


def main():
    st.set_page_config(page_title="DistrictFlow Dashboard", layout="wide")
    st.title("DistrictFlow Dashboard")
    st.caption("Multi-agent traffic control with district-level coordination")

    col1, col2, col3 = st.columns(3)
    seed = col1.number_input("Seed", min_value=0, max_value=100000, value=0, step=1)
    max_steps = col2.slider(
        "Max steps", min_value=50, max_value=500, value=200, step=10
    )
    use_district_coordination = col3.checkbox(
        "Enable district coordination", value=True
    )

    st.subheader("Generated City Viewer")
    root_dir = Path(
        st.text_input("Generated dataset dir", value="data/generated")
    )
    cities = _list_generated_cities(root_dir)
    if not cities:
        st.info("No generated cities found in the selected directory.")
    else:
        selected_city = st.selectbox("City", options=cities, index=0)
        show_districts = st.checkbox("Show district overlay", value=True)
        show_gateways = st.checkbox("Show perimeter gateways", value=True)
        show_gateway_labels = st.checkbox("Label gateways", value=False)
        city_dir = root_dir / selected_city
        roadnet_path = city_dir / "roadnet.json"
        district_map_path = city_dir / "district_map.json"

        if roadnet_path.exists():
            roadnet = _load_json(roadnet_path)
            district_map = _load_json(district_map_path) if district_map_path.exists() else None
            fig = _plot_city_geometry(
                roadnet=roadnet,
                district_map=district_map,
                show_gateways=show_gateways,
                show_districts=show_districts,
                show_labels=show_gateway_labels,
            )
            st.pyplot(fig, use_container_width=True)
        else:
            st.warning(f"Missing roadnet file: {roadnet_path}")

    if st.button("Run Simulation", use_container_width=True):
        env = make_env()
        env.max_steps = max_steps

        local_policy = SharedHeuristicLocalPolicy()

        district_coordinators = {}
        if use_district_coordination:
            district_coordinators = {
                "D0": RuleBasedDistrictCoordinator(),
                "D1": RuleBasedDistrictCoordinator(),
            }

        result = run_episode(
            env=env,
            local_policy=local_policy,
            district_coordinators=district_coordinators,
            seed=int(seed),
            max_steps=max_steps,
            record_history=True,
            policy_update=False,
        )

        summary = summarize_history(result.history)
        metrics_df, directives_df = build_history_frames(result.history)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Avg Waiting", f"{summary['avg_total_waiting']:.2f}")
        s2.metric("Avg Queue", f"{summary['avg_total_queue']:.2f}")
        s3.metric("Avg Reward", f"{summary['avg_mean_reward']:.2f}")
        s4.metric("Steps", int(summary["num_steps"]))

        if not metrics_df.empty:
            st.subheader("Simulation Metrics")

            fig1 = plt.figure()
            plt.plot(metrics_df["step"], metrics_df["total_waiting"])
            plt.xlabel("Step")
            plt.ylabel("Total Waiting")
            plt.title("Total Waiting Over Time")
            st.pyplot(fig1)

            fig2 = plt.figure()
            plt.plot(metrics_df["step"], metrics_df["total_queue"])
            plt.xlabel("Step")
            plt.ylabel("Total Queue")
            plt.title("Total Queue Over Time")
            st.pyplot(fig2)

            fig3 = plt.figure()
            plt.plot(metrics_df["step"], metrics_df["mean_reward"])
            plt.xlabel("Step")
            plt.ylabel("Mean Reward")
            plt.title("Mean Reward Over Time")
            st.pyplot(fig3)

            st.subheader("Raw Metrics")
            st.dataframe(metrics_df, use_container_width=True)

        if not directives_df.empty:
            st.subheader("District Directives")
            st.dataframe(directives_df, use_container_width=True)

        st.subheader("Final Info")
        st.json(result.final_info)


if __name__ == "__main__":
    main()
