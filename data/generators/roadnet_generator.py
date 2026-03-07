"""Road network topology generation and CityFlow roadnet assembly."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from itertools import product

from .schemas import CityGraph, RoadRecord, TopologyType
from .utils import euclidean


class RoadnetGenerator:
    """Generate city intersection graph and convert to CityFlow roadnet format."""

    def generate(
        self,
        city_id: str,
        seed: int,
        topology: TopologyType,
        target_intersections: int,
        ring_diagonal_keep_prob: float = 0.07,
        ring_max_diagonal_fraction: float = 0.03,
    ) -> CityGraph:
        rng = random.Random(seed)
        coords, undirected_edges, arterial_pairs = self._build_topology(
            topology=topology,
            target_nodes=target_intersections,
            rng=rng,
            ring_diagonal_keep_prob=ring_diagonal_keep_prob,
            ring_max_diagonal_fraction=ring_max_diagonal_fraction,
        )
        (
            coords,
            undirected_edges,
            gateway_pairs,
            gateway_nodes,
        ) = self._augment_with_perimeter_gateways(
            coords=coords,
            undirected_edges=undirected_edges,
            rng=rng,
        )
        adjacency = self._to_adjacency(coords, undirected_edges)
        directed_roads, arterial_road_ids, gateway_road_ids = self._build_directed_roads(
            coords=coords,
            undirected_edges=undirected_edges,
            arterial_pairs=arterial_pairs,
            gateway_pairs=gateway_pairs,
        )
        roadnet = self._build_roadnet(
            coords=coords,
            adjacency=adjacency,
            directed_roads=directed_roads,
        )
        return CityGraph(
            city_id=city_id,
            topology=topology,
            seed=seed,
            intersections=coords,
            adjacency=adjacency,
            directed_roads=directed_roads,
            roadnet=roadnet,
            arterial_roads=arterial_road_ids,
            gateway_intersections=gateway_nodes,
            gateway_roads=gateway_road_ids,
        )

    def _build_topology(
        self,
        topology: TopologyType,
        target_nodes: int,
        rng: random.Random,
        ring_diagonal_keep_prob: float,
        ring_max_diagonal_fraction: float,
    ) -> tuple[
        dict[str, tuple[float, float]],
        list[tuple[str, str]],
        set[frozenset[str]],
    ]:
        if topology == "rectangular_grid":
            return self._rectangular_grid(target_nodes, rng)
        if topology == "irregular_grid":
            return self._irregular_grid(target_nodes, rng)
        if topology == "arterial_local":
            return self._arterial_local(target_nodes, rng)
        if topology == "ring_road":
            return self._ring_road(
                target_nodes=target_nodes,
                rng=rng,
                ring_diagonal_keep_prob=ring_diagonal_keep_prob,
                ring_max_diagonal_fraction=ring_max_diagonal_fraction,
            )
        return self._mixed(target_nodes, rng)

    def _dimensions(self, target_nodes: int) -> tuple[int, int]:
        cols = max(3, int(round(math.sqrt(target_nodes))))
        rows = max(3, int(math.ceil(target_nodes / cols)))
        return rows, cols

    def _grid_coords(
        self,
        rows: int,
        cols: int,
        spacing: float,
        jitter: float,
        rng: random.Random,
    ) -> dict[str, tuple[float, float]]:
        coords: dict[str, tuple[float, float]] = {}
        idx = 0
        for r, c in product(range(rows), range(cols)):
            x = c * spacing + rng.uniform(-jitter, jitter)
            y = r * spacing + rng.uniform(-jitter, jitter)
            coords[f"i_{idx:04d}"] = (x, y)
            idx += 1
        return coords

    def _smooth_axis_offsets(
        self,
        size: int,
        max_offset: float,
        step_limit: float,
        rng: random.Random,
    ) -> list[float]:
        offsets = [0.0] * size
        drift = 0.0
        for idx in range(1, size - 1):
            drift += rng.uniform(-step_limit, step_limit)
            drift = max(-max_offset, min(max_offset, drift))
            offsets[idx] = drift

        if size > 2:
            mean_mid = sum(offsets[1:-1]) / (size - 2)
            for idx in range(1, size - 1):
                centered = offsets[idx] - mean_mid
                offsets[idx] = max(-max_offset, min(max_offset, centered))

        offsets[0] = 0.0
        offsets[-1] = 0.0
        return offsets

    def _boundary_stability_weight(self, idx: int, size: int) -> float:
        distance = min(idx, size - 1 - idx)
        if distance <= 0:
            return 0.0
        if distance == 1:
            return 0.2
        if distance == 2:
            return 0.45
        if distance == 3:
            return 0.72
        return 1.00

    def _axis_positions(
        self,
        size: int,
        spacing: float,
        gap_variation: float,
        rng: random.Random,
    ) -> list[float]:
        if size <= 1:
            return [0.0]

        gap_profile = self._smooth_drop_profile(size - 1, rng)
        positions = [0.0]
        min_gap = spacing * (1.0 - gap_variation)
        max_gap = spacing * (1.0 + gap_variation)

        for gap_idx in range(size - 1):
            centered = (gap_profile[gap_idx] - 0.5) * 2.0
            edge_distance = min(gap_idx, (size - 2) - gap_idx)
            if edge_distance <= 0:
                edge_weight = 0.32
            elif edge_distance == 1:
                edge_weight = 0.55
            elif edge_distance == 2:
                edge_weight = 0.8
            else:
                edge_weight = 1.0

            local_noise = rng.uniform(-0.22, 0.22) * gap_variation * edge_weight
            gap = spacing * (1.0 + (centered * gap_variation * edge_weight) + local_noise)
            gap = max(min_gap, min(max_gap, gap))
            positions.append(positions[-1] + gap)

        nominal_span = spacing * (size - 1)
        actual_span = positions[-1]
        if actual_span > 1e-9:
            scale = nominal_span / actual_span
            positions = [p * scale for p in positions]
        return positions

    def _irregular_grid_coords(
        self,
        rows: int,
        cols: int,
        spacing: float,
        rng: random.Random,
    ) -> dict[str, tuple[float, float]]:
        # Keep intersections mostly on row/column lines while varying block size.
        row_positions = self._axis_positions(rows, spacing, gap_variation=0.16, rng=rng)
        col_positions = self._axis_positions(cols, spacing, gap_variation=0.16, rng=rng)

        # Small line-wise drift and very small local jitter.
        max_row_offset = spacing * 0.018
        max_col_offset = spacing * 0.018
        row_step = spacing * 0.006
        col_step = spacing * 0.006
        local_jitter = spacing * 0.0045

        row_offsets = self._smooth_axis_offsets(rows, max_row_offset, row_step, rng)
        col_offsets = self._smooth_axis_offsets(cols, max_col_offset, col_step, rng)

        coords: dict[str, tuple[float, float]] = {}
        idx = 0
        for r, c in product(range(rows), range(cols)):
            # Keep perimeter nodes stable while allowing interior irregularity.
            perimeter_weight = min(
                self._boundary_stability_weight(r, rows),
                self._boundary_stability_weight(c, cols),
            )
            jitter_weight = perimeter_weight * perimeter_weight
            x = col_positions[c] + (col_offsets[c] * perimeter_weight)
            y = row_positions[r] + (row_offsets[r] * perimeter_weight)
            x += rng.uniform(-local_jitter, local_jitter) * jitter_weight
            y += rng.uniform(-local_jitter, local_jitter) * jitter_weight
            coords[f"i_{idx:04d}"] = (x, y)
            idx += 1
        return coords

    def _grid_edges(self, rows: int, cols: int) -> list[tuple[int, int]]:
        edges: list[tuple[int, int]] = []
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if c + 1 < cols:
                    edges.append((idx, idx + 1))
                if r + 1 < rows:
                    edges.append((idx, idx + cols))
        return edges

    def _estimate_spacing(
        self,
        coords: dict[str, tuple[float, float]],
        undirected_edges: list[tuple[str, str]],
    ) -> float:
        if not undirected_edges:
            return 120.0
        lengths = [
            euclidean(coords[a], coords[b])
            for a, b in undirected_edges
            if a in coords and b in coords
        ]
        if not lengths:
            return 120.0
        lengths.sort()
        return lengths[len(lengths) // 2]

    def _select_spread_nodes(
        self,
        candidates: list[str],
        coords: dict[str, tuple[float, float]],
        count: int,
        axis: str,
    ) -> list[str]:
        if count <= 0 or not candidates:
            return []
        if len(candidates) <= count:
            return candidates
        axis_idx = 0 if axis == "x" else 1
        ordered = sorted(
            candidates,
            key=lambda nid: (coords[nid][axis_idx], nid),
        )
        selected: list[str] = []
        for i in range(count):
            pos = int(round((i * (len(ordered) - 1)) / max(1, count - 1)))
            selected.append(ordered[pos])
        deduped = sorted(set(selected), key=lambda nid: ordered.index(nid))
        if len(deduped) >= count:
            return deduped[:count]
        for node in ordered:
            if node in deduped:
                continue
            deduped.append(node)
            if len(deduped) >= count:
                break
        return deduped

    def _augment_with_perimeter_gateways(
        self,
        coords: dict[str, tuple[float, float]],
        undirected_edges: list[tuple[str, str]],
        rng: random.Random,
    ) -> tuple[
        dict[str, tuple[float, float]],
        list[tuple[str, str]],
        set[frozenset[str]],
        set[str],
    ]:
        if not coords:
            return coords, undirected_edges, set(), set()

        min_x = min(x for x, _ in coords.values())
        max_x = max(x for x, _ in coords.values())
        min_y = min(y for _, y in coords.values())
        max_y = max(y for _, y in coords.values())
        spacing = self._estimate_spacing(coords, undirected_edges)
        threshold = max(4.0, spacing * 0.08)
        per_side_target = 2

        side_to_candidates: dict[str, list[str]] = {
            "west": [],
            "east": [],
            "south": [],
            "north": [],
        }
        for node_id, (x, y) in coords.items():
            distances = {
                "west": abs(x - min_x),
                "east": abs(x - max_x),
                "south": abs(y - min_y),
                "north": abs(y - max_y),
            }
            side = min(distances, key=distances.get)
            if distances[side] <= threshold:
                side_to_candidates[side].append(node_id)

        selected_anchors: list[tuple[str, str]] = []
        used_anchors: set[str] = set()
        for side in ("west", "east", "south", "north"):
            candidates = [n for n in side_to_candidates[side] if n not in used_anchors]
            axis = "y" if side in {"west", "east"} else "x"
            chosen = self._select_spread_nodes(
                candidates=candidates,
                coords=coords,
                count=per_side_target,
                axis=axis,
            )
            for anchor in chosen:
                if anchor in used_anchors:
                    continue
                used_anchors.add(anchor)
                selected_anchors.append((side, anchor))

        if not selected_anchors:
            return coords, undirected_edges, set(), set()

        offset = max(45.0, spacing * 0.82)
        gateway_pairs: set[frozenset[str]] = set()
        gateway_nodes: set[str] = set()
        next_idx = 0
        for side, anchor in selected_anchors:
            ax, ay = coords[anchor]
            if side == "west":
                gx, gy = min_x - offset, ay + rng.uniform(-spacing * 0.03, spacing * 0.03)
            elif side == "east":
                gx, gy = max_x + offset, ay + rng.uniform(-spacing * 0.03, spacing * 0.03)
            elif side == "south":
                gx, gy = ax + rng.uniform(-spacing * 0.03, spacing * 0.03), min_y - offset
            else:
                gx, gy = ax + rng.uniform(-spacing * 0.03, spacing * 0.03), max_y + offset

            gateway_id = f"g_{next_idx:04d}"
            next_idx += 1
            coords[gateway_id] = (gx, gy)
            undirected_edges.append((anchor, gateway_id))
            gateway_pairs.add(frozenset((anchor, gateway_id)))
            gateway_nodes.add(gateway_id)

        return coords, undirected_edges, gateway_pairs, gateway_nodes

    def _arterial_indices(self, size: int) -> list[int]:
        candidates = {size // 3, size // 2, (2 * size) // 3}
        selected = sorted(i for i in candidates if 0 < i < size - 1)
        if not selected and size > 2:
            selected = [size // 2]
        return selected

    def _smooth_drop_profile(self, size: int, rng: random.Random) -> list[float]:
        values: list[float] = []
        state = rng.uniform(-0.25, 0.25)
        for _ in range(size):
            state = 0.78 * state + 0.22 * rng.uniform(-1.0, 1.0)
            values.append(state)
        low = min(values)
        high = max(values)
        if abs(high - low) < 1e-6:
            return [0.5] * size
        return [(value - low) / (high - low) for value in values]

    def _edge_orientation(
        self,
        a: int,
        b: int,
        cols: int,
    ) -> tuple[str, int, int]:
        ra, ca = divmod(a, cols)
        rb, cb = divmod(b, cols)
        if ra == rb:
            return ("horizontal", ra, min(ca, cb))
        return ("vertical", ca, min(ra, rb))

    def _is_edge_protected(
        self,
        a: int,
        b: int,
        rows: int,
        cols: int,
        arterial_rows: set[int],
        arterial_cols: set[int],
    ) -> tuple[bool, bool]:
        ra, ca = divmod(a, cols)
        rb, cb = divmod(b, cols)
        if ra == rb:
            on_perimeter = ra in {0, rows - 1}
            on_arterial = ra in arterial_rows
        else:
            on_perimeter = ca in {0, cols - 1}
            on_arterial = ca in arterial_cols
        return on_perimeter or on_arterial, on_arterial

    def _has_path_without_edge(
        self,
        start: int,
        goal: int,
        adjacency: dict[int, set[int]],
        edge: tuple[int, int],
    ) -> bool:
        blocked_u, blocked_v = edge
        stack = [start]
        visited = {start}
        while stack:
            node = stack.pop()
            if node == goal:
                return True
            for nxt in adjacency[node]:
                if (
                    (node == blocked_u and nxt == blocked_v)
                    or (node == blocked_v and nxt == blocked_u)
                ):
                    continue
                if nxt in visited:
                    continue
                visited.add(nxt)
                stack.append(nxt)
        return False

    def _is_connected_without_str_edge(
        self,
        start: str,
        goal: str,
        adjacency: dict[str, set[str]],
        edge: tuple[str, str],
    ) -> bool:
        blocked_u, blocked_v = edge
        stack = [start]
        visited = {start}
        while stack:
            node = stack.pop()
            if node == goal:
                return True
            for nxt in adjacency[node]:
                if (
                    (node == blocked_u and nxt == blocked_v)
                    or (node == blocked_v and nxt == blocked_u)
                ):
                    continue
                if nxt in visited:
                    continue
                visited.add(nxt)
                stack.append(nxt)
        return False

    def _rectangular_grid(
        self, target_nodes: int, rng: random.Random
    ) -> tuple[dict[str, tuple[float, float]], list[tuple[str, str]], set[frozenset[str]]]:
        rows, cols = self._dimensions(target_nodes)
        coords = self._grid_coords(rows, cols, spacing=120.0, jitter=6.0, rng=rng)
        edges_raw = self._grid_edges(rows, cols)
        id_list = list(coords.keys())
        edges = [(id_list[a], id_list[b]) for a, b in edges_raw]
        return coords, edges, set()

    def _irregular_grid(
        self, target_nodes: int, rng: random.Random
    ) -> tuple[dict[str, tuple[float, float]], list[tuple[str, str]], set[frozenset[str]]]:
        rows, cols = self._dimensions(int(target_nodes * 1.08))
        coords = self._irregular_grid_coords(rows, cols, spacing=115.0, rng=rng)
        edges_raw = self._grid_edges(rows, cols)
        filtered: set[tuple[int, int]] = set(edges_raw)
        adjacency: dict[int, set[int]] = defaultdict(set)
        for a, b in edges_raw:
            adjacency[a].add(b)
            adjacency[b].add(a)

        arterial_rows = set(self._arterial_indices(rows))
        arterial_cols = set(self._arterial_indices(cols))
        row_profile = self._smooth_drop_profile(rows, rng)
        col_profile = self._smooth_drop_profile(cols, rng)
        base_drop_prob = 0.11

        interior_rows = [r for r in range(2, rows - 2) if r not in arterial_rows]
        interior_cols = [c for c in range(2, cols - 2) if c not in arterial_cols]
        row_gap_rows = set(
            rng.sample(interior_rows, k=min(max(1, rows // 7), len(interior_rows)))
        ) if interior_rows else set()
        col_gap_cols = set(
            rng.sample(interior_cols, k=min(max(1, cols // 7), len(interior_cols)))
        ) if interior_cols else set()

        arterial_pairs: set[frozenset[str]] = set()
        removable: list[tuple[int, int]] = []
        for a, b in edges_raw:
            protected, arterial = self._is_edge_protected(
                a=a,
                b=b,
                rows=rows,
                cols=cols,
                arterial_rows=arterial_rows,
                arterial_cols=arterial_cols,
            )
            if arterial:
                ida = f"i_{a:04d}"
                idb = f"i_{b:04d}"
                arterial_pairs.add(frozenset((ida, idb)))
            if not protected:
                removable.append((a, b))

        rng.shuffle(removable)
        for a, b in removable:
            orientation, major_idx, minor_idx = self._edge_orientation(a, b, cols)
            if orientation == "horizontal":
                drop_prob = base_drop_prob + (0.11 * row_profile[major_idx]) + (
                    0.08 * col_profile[minor_idx]
                )
                if major_idx in row_gap_rows:
                    drop_prob += 0.16
            else:
                drop_prob = base_drop_prob + (0.11 * col_profile[major_idx]) + (
                    0.08 * row_profile[minor_idx]
                )
                if major_idx in col_gap_cols:
                    drop_prob += 0.16

            # Keep perimeter-adjacent links denser to avoid sparse fringes.
            ra, ca = divmod(a, cols)
            rb, cb = divmod(b, cols)
            boundary_distance = min(
                ra,
                rb,
                rows - 1 - ra,
                rows - 1 - rb,
                ca,
                cb,
                cols - 1 - ca,
                cols - 1 - cb,
            )
            if boundary_distance <= 1:
                drop_prob *= 0.45
            elif boundary_distance == 2:
                drop_prob *= 0.75

            if rng.random() > min(0.46, max(0.0, drop_prob)):
                continue
            if len(adjacency[a]) <= 2 or len(adjacency[b]) <= 2:
                continue
            if not self._has_path_without_edge(a, b, adjacency, (a, b)):
                continue

            adjacency[a].remove(b)
            adjacency[b].remove(a)
            filtered.remove((a, b))

        id_list = list(coords.keys())
        edges = [(id_list[a], id_list[b]) for a, b in sorted(filtered)]
        return coords, edges, arterial_pairs

    def _arterial_local(
        self, target_nodes: int, rng: random.Random
    ) -> tuple[dict[str, tuple[float, float]], list[tuple[str, str]], set[frozenset[str]]]:
        rows, cols = self._dimensions(target_nodes)
        coords = self._grid_coords(rows, cols, spacing=130.0, jitter=8.0, rng=rng)
        edges_raw = self._grid_edges(rows, cols)
        arterial_rows = {rows // 3, (2 * rows) // 3}
        arterial_cols = {cols // 3, (2 * cols) // 3}
        id_list = list(coords.keys())
        edges: list[tuple[str, str]] = []
        arterial_pairs: set[frozenset[str]] = set()

        for a, b in edges_raw:
            ra, ca = divmod(a, cols)
            rb, cb = divmod(b, cols)
            ida, idb = id_list[a], id_list[b]
            edges.append((ida, idb))
            if (
                ra in arterial_rows
                or rb in arterial_rows
                or ca in arterial_cols
                or cb in arterial_cols
            ):
                arterial_pairs.add(frozenset((ida, idb)))
            elif rng.random() < 0.06:
                arterial_pairs.add(frozenset((ida, idb)))
        return coords, edges, arterial_pairs

    def _ring_road(
        self,
        target_nodes: int,
        rng: random.Random,
        ring_diagonal_keep_prob: float,
        ring_max_diagonal_fraction: float,
    ) -> tuple[dict[str, tuple[float, float]], list[tuple[str, str]], set[frozenset[str]]]:
        ring_nodes = max(14, int(target_nodes * 0.22))
        inner_nodes = max(24, target_nodes - ring_nodes)
        rows, cols = self._dimensions(inner_nodes)
        inner_coords = self._grid_coords(rows, cols, spacing=90.0, jitter=10.0, rng=rng)
        inner_ids = list(inner_coords.keys())
        inner_index = {node: idx for idx, node in enumerate(inner_ids)}
        min_x = min(x for x, _ in inner_coords.values())
        max_x = max(x for x, _ in inner_coords.values())
        min_y = min(y for _, y in inner_coords.values())
        max_y = max(y for _, y in inner_coords.values())
        center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
        radius = max(max_x - min_x, max_y - min_y) * 0.70

        def norm_edge(a: str, b: str) -> tuple[str, str]:
            return (a, b) if a < b else (b, a)

        coords: dict[str, tuple[float, float]] = dict(inner_coords)
        edge_set: set[tuple[str, str]] = set()
        arterial_pairs: set[frozenset[str]] = set()

        for a, b in self._grid_edges(rows, cols):
            edge_set.add(norm_edge(inner_ids[a], inner_ids[b]))

        ring_ids: list[str] = []
        start_idx = len(coords)
        for i in range(ring_nodes):
            angle = (2.0 * math.pi * i) / ring_nodes
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            rid = f"i_{start_idx + i:04d}"
            ring_ids.append(rid)
            coords[rid] = (x, y)

        for i, rid in enumerate(ring_ids):
            nxt = ring_ids[(i + 1) % ring_nodes]
            edge = norm_edge(rid, nxt)
            edge_set.add(edge)
            arterial_pairs.add(frozenset(edge))

        boundary_inner_nodes = [
            nid
            for nid in inner_ids
            if ((inner_index[nid] // cols) in {0, rows - 1})
            or ((inner_index[nid] % cols) in {0, cols - 1})
        ]
        spokes = max(6, ring_nodes // 3)
        anchor_pool = boundary_inner_nodes[:] if boundary_inner_nodes else inner_ids
        anchor_ids = [
            anchor_pool[(idx * len(anchor_pool)) // spokes]
            for idx in range(spokes)
        ]

        protected_spokes: set[tuple[str, str]] = set()
        for i in range(spokes):
            ring_node = ring_ids[(i * ring_nodes) // spokes]
            inner_node = anchor_ids[i]
            edge = norm_edge(ring_node, inner_node)
            edge_set.add(edge)
            arterial_pairs.add(frozenset(edge))
            protected_spokes.add(edge)

        # Optional non-primary diagonals: sparse extra radials + sparse interior diagonals.
        diagonal_candidates: list[tuple[tuple[str, str], float]] = []
        extra_spokes = max(2, min(5, ring_nodes // 4))
        for i in range(extra_spokes):
            ring_node = ring_ids[(i * ring_nodes) // extra_spokes]
            inner_node = anchor_pool[(i * len(anchor_pool)) // extra_spokes]
            edge = norm_edge(ring_node, inner_node)
            if edge in edge_set or edge in protected_spokes:
                continue
            score = 1.2 + (0.003 * euclidean(coords[ring_node], coords[inner_node]))
            diagonal_candidates.append((edge, score))

        def _orientation(
            p: tuple[float, float],
            q: tuple[float, float],
            r: tuple[float, float],
        ) -> int:
            value = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
            if abs(value) < 1e-9:
                return 0
            return 1 if value > 0 else 2

        def _on_segment(
            p: tuple[float, float],
            q: tuple[float, float],
            r: tuple[float, float],
        ) -> bool:
            return (
                min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
                and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
            )

        def _segments_intersect(
            p1: tuple[float, float],
            q1: tuple[float, float],
            p2: tuple[float, float],
            q2: tuple[float, float],
        ) -> bool:
            o1 = _orientation(p1, q1, p2)
            o2 = _orientation(p1, q1, q2)
            o3 = _orientation(p2, q2, p1)
            o4 = _orientation(p2, q2, q1)
            if o1 != o2 and o3 != o4:
                return True
            if o1 == 0 and _on_segment(p1, p2, q1):
                return True
            if o2 == 0 and _on_segment(p1, q2, q1):
                return True
            if o3 == 0 and _on_segment(p2, p1, q2):
                return True
            if o4 == 0 and _on_segment(p2, q1, q2):
                return True
            return False

        def _has_nonendpoint_intersection(
            edge: tuple[str, str],
            other_edges: set[tuple[str, str]],
        ) -> bool:
            a, b = edge
            p1, q1 = coords[a], coords[b]
            for u, v in other_edges:
                # Shared endpoint is expected at valid junctions.
                if len({a, b, u, v}) < 4:
                    continue
                p2, q2 = coords[u], coords[v]
                if _segments_intersect(p1, q1, p2, q2):
                    return True
            return False

        for r in range(rows - 1):
            for c in range(cols - 1):
                if rng.random() > 0.05:
                    continue
                tl = inner_ids[(r * cols) + c]
                tr = inner_ids[(r * cols) + c + 1]
                bl = inner_ids[((r + 1) * cols) + c]
                br = inner_ids[((r + 1) * cols) + c + 1]
                a, b = (tl, br) if rng.random() < 0.5 else (tr, bl)
                edge = norm_edge(a, b)
                if edge in edge_set:
                    continue
                boundary_bonus = 0.30 if (r in {0, rows - 2} or c in {0, cols - 2}) else 0.0
                diagonal_candidates.append((edge, 1.0 + boundary_bonus))

        keep_prob = max(0.0, min(1.0, ring_diagonal_keep_prob))
        kept_candidates: list[tuple[tuple[str, str], float]] = []
        geometry_edges = set(edge_set)
        for edge, score in diagonal_candidates:
            if rng.random() > keep_prob:
                continue
            if _has_nonendpoint_intersection(edge, geometry_edges):
                continue
            kept_candidates.append((edge, score))
            geometry_edges.add(edge)
        if diagonal_candidates and not kept_candidates:
            for edge, score in sorted(diagonal_candidates, key=lambda item: item[1], reverse=True):
                if not _has_nonendpoint_intersection(edge, edge_set):
                    kept_candidates.append((edge, score))
                    break

        for edge, _ in kept_candidates:
            edge_set.add(edge)

        # Prune weaker optional diagonals while preserving connectivity.
        max_fraction = max(0.0, min(1.0, ring_max_diagonal_fraction))
        max_kept = max(1, int(round(max_fraction * max(1, len(diagonal_candidates)))))
        hard_cap = max(1, min(3, len(inner_ids) // 60))
        max_kept = min(max_kept, hard_cap)
        if len(kept_candidates) > max_kept:
            adjacency: dict[str, set[str]] = {nid: set() for nid in coords}
            for u, v in edge_set:
                adjacency[u].add(v)
                adjacency[v].add(u)
            kept = len(kept_candidates)
            for edge, _ in sorted(kept_candidates, key=lambda item: item[1]):
                if kept <= max_kept:
                    break
                u, v = edge
                if edge not in edge_set:
                    continue
                if len(adjacency[u]) <= 1 or len(adjacency[v]) <= 1:
                    continue
                if not self._is_connected_without_str_edge(u, v, adjacency, edge):
                    continue
                adjacency[u].remove(v)
                adjacency[v].remove(u)
                edge_set.remove(edge)
                kept -= 1

        edges = sorted(edge_set)
        return coords, edges, arterial_pairs

    def _mixed(
        self, target_nodes: int, rng: random.Random
    ) -> tuple[dict[str, tuple[float, float]], list[tuple[str, str]], set[frozenset[str]]]:
        coords, edges, arterial_pairs = self._arterial_local(target_nodes, rng)
        ids = list(coords.keys())
        for _ in range(max(3, len(ids) // 20)):
            a, b = rng.sample(ids, 2)
            if euclidean(coords[a], coords[b]) < 220.0:
                edge = (a, b) if a < b else (b, a)
                if edge not in edges:
                    edges.append(edge)
                    if rng.random() < 0.4:
                        arterial_pairs.add(frozenset(edge))
        return coords, edges, arterial_pairs

    def _to_adjacency(
        self,
        coords: dict[str, tuple[float, float]],
        undirected_edges: list[tuple[str, str]],
    ) -> dict[str, set[str]]:
        adjacency: dict[str, set[str]] = {nid: set() for nid in coords}
        for a, b in undirected_edges:
            if a == b:
                continue
            adjacency[a].add(b)
            adjacency[b].add(a)
        return adjacency

    def _build_directed_roads(
        self,
        coords: dict[str, tuple[float, float]],
        undirected_edges: list[tuple[str, str]],
        arterial_pairs: set[frozenset[str]],
        gateway_pairs: set[frozenset[str]],
    ) -> tuple[dict[str, RoadRecord], set[str], set[str]]:
        roads: dict[str, RoadRecord] = {}
        arterial_ids: set[str] = set()
        gateway_ids: set[str] = set()
        for a, b in undirected_edges:
            for start, end in ((a, b), (b, a)):
                is_arterial = frozenset((a, b)) in arterial_pairs
                is_gateway = frozenset((a, b)) in gateway_pairs
                if is_gateway:
                    speed = 12.0
                    lanes = 2
                else:
                    speed = 14.0 if is_arterial else 11.0
                    lanes = 3 if is_arterial else 2
                rid = f"r_{start}_{end}"
                points = [
                    {"x": round(coords[start][0], 3), "y": round(coords[start][1], 3)},
                    {"x": round(coords[end][0], 3), "y": round(coords[end][1], 3)},
                ]
                record = RoadRecord(
                    id=rid,
                    start_intersection=start,
                    end_intersection=end,
                    length=euclidean(coords[start], coords[end]),
                    speed_limit=speed,
                    num_lanes=lanes,
                    points=points,
                    is_arterial=is_arterial,
                )
                roads[rid] = record
                if is_arterial:
                    arterial_ids.add(rid)
                if is_gateway:
                    gateway_ids.add(rid)
        return roads, arterial_ids, gateway_ids

    def _build_roadnet(
        self,
        coords: dict[str, tuple[float, float]],
        adjacency: dict[str, set[str]],
        directed_roads: dict[str, RoadRecord],
    ) -> dict[str, list[dict[str, object]]]:
        in_roads_by_node: dict[str, list[RoadRecord]] = defaultdict(list)
        out_roads_by_node: dict[str, list[RoadRecord]] = defaultdict(list)
        for road in directed_roads.values():
            out_roads_by_node[road.start_intersection].append(road)
            in_roads_by_node[road.end_intersection].append(road)

        min_x = min(x for x, _ in coords.values())
        max_x = max(x for x, _ in coords.values())
        min_y = min(y for _, y in coords.values())
        max_y = max(y for _, y in coords.values())
        border_eps = 3.0

        intersections: list[dict[str, object]] = []
        for nid in sorted(coords):
            x, y = coords[nid]
            degree = len(adjacency[nid])
            is_border = (
                abs(x - min_x) < border_eps
                or abs(x - max_x) < border_eps
                or abs(y - min_y) < border_eps
                or abs(y - max_y) < border_eps
            )
            # Keep boundary intersections non-virtual when they are part of the street grid.
            # Mark only true stubs/dead-ends as virtual.
            virtual = degree <= 1

            road_links: list[dict[str, object]] = []
            incoming = sorted(in_roads_by_node[nid], key=lambda r: r.id)
            outgoing = sorted(out_roads_by_node[nid], key=lambda r: r.id)
            for in_road in incoming:
                for out_road in outgoing:
                    if out_road.end_intersection == in_road.start_intersection:
                        continue
                    lane_links = []
                    lane_count = min(in_road.num_lanes, out_road.num_lanes)
                    for lane_idx in range(lane_count):
                        lane_links.append(
                            {
                                "startLaneIndex": lane_idx,
                                "endLaneIndex": lane_idx,
                                "points": [
                                    dict(in_road.points[-1]),
                                    dict(out_road.points[0]),
                                ],
                            }
                        )
                    road_links.append(
                        {
                            "type": self._movement_type(
                                coords[in_road.start_intersection],
                                coords[nid],
                                coords[out_road.end_intersection],
                            ),
                            "startRoad": in_road.id,
                            "endRoad": out_road.id,
                            "laneLinks": lane_links,
                        }
                    )

            lightphases = self._light_phases(nid, coords, incoming, road_links)
            connected_roads = sorted(
                {road.id for road in incoming + outgoing}
            )
            intersections.append(
                {
                    "id": nid,
                    "point": {"x": round(x, 3), "y": round(y, 3)},
                    "width": 0,
                    "roads": connected_roads,
                    "virtual": virtual,
                    "roadLinks": road_links,
                    "trafficLight": {
                        "roadLinkIndices": list(range(len(road_links))),
                        "lightphases": lightphases,
                    },
                }
            )

        roads: list[dict[str, object]] = []
        for rid in sorted(directed_roads):
            road = directed_roads[rid]
            roads.append(
                {
                    "id": road.id,
                    "startIntersection": road.start_intersection,
                    "endIntersection": road.end_intersection,
                    "points": road.points,
                    "lanes": [
                        {"maxSpeed": road.speed_limit, "width": 3.2}
                        for _ in range(road.num_lanes)
                    ],
                }
            )
        return {"intersections": intersections, "roads": roads}

    def _movement_type(
        self,
        in_start: tuple[float, float],
        center: tuple[float, float],
        out_end: tuple[float, float],
    ) -> str:
        v1 = (center[0] - in_start[0], center[1] - in_start[1])
        v2 = (out_end[0] - center[0], out_end[1] - center[1])
        cross = (v1[0] * v2[1]) - (v1[1] * v2[0])
        dot = (v1[0] * v2[0]) + (v1[1] * v2[1])
        angle = math.atan2(cross, dot)
        if abs(angle) < (math.pi / 4):
            return "go_straight"
        if angle > 0:
            return "turn_left"
        return "turn_right"

    def _light_phases(
        self,
        node_id: str,
        coords: dict[str, tuple[float, float]],
        incoming: list[RoadRecord],
        road_links: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if not road_links:
            return [{"time": 30, "availableRoadLinks": []}]

        vertical_incoming: set[str] = set()
        horizontal_incoming: set[str] = set()
        center = coords[node_id]
        for road in incoming:
            source = coords[road.start_intersection]
            vx = center[0] - source[0]
            vy = center[1] - source[1]
            if abs(vy) >= abs(vx):
                vertical_incoming.add(road.id)
            else:
                horizontal_incoming.add(road.id)

        vertical_links: list[int] = []
        horizontal_links: list[int] = []
        for idx, link in enumerate(road_links):
            start_road = str(link["startRoad"])
            if start_road in vertical_incoming:
                vertical_links.append(idx)
            else:
                horizontal_links.append(idx)

        if not vertical_links or not horizontal_links:
            return [{"time": 35, "availableRoadLinks": list(range(len(road_links)))}]

        return [
            {"time": 30, "availableRoadLinks": vertical_links},
            {"time": 5, "availableRoadLinks": []},
            {"time": 30, "availableRoadLinks": horizontal_links},
            {"time": 5, "availableRoadLinks": []},
        ]
