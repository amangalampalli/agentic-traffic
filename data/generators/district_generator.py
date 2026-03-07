"""District partitioning and district-level metadata generation."""

from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

from .schemas import CityGraph, DistrictData, DistrictRecord, DistrictType
from .utils import connected_components, euclidean


class DistrictGenerator:
    """Generate contiguous district partitions over the city intersection graph."""

    DISTRICT_TYPE_WEIGHTS: dict[DistrictType, float] = {
        "residential": 0.35,
        "commercial": 0.25,
        "industrial": 0.20,
        "mixed": 0.20,
    }

    def generate(
        self,
        city_graph: CityGraph,
        num_districts: int,
        seed: int,
    ) -> DistrictData:
        rng = random.Random(seed)
        node_ids = sorted(
            n
            for n in city_graph.intersections.keys()
            if n not in city_graph.gateway_intersections
        )
        if len(node_ids) < 2:
            raise ValueError("Insufficient non-gateway intersections for districting.")
        if num_districts >= len(node_ids):
            num_districts = max(2, len(node_ids) // 2)

        local_coords = {nid: city_graph.intersections[nid] for nid in node_ids}
        seeds = self._farthest_seeds(local_coords, num_districts, rng)
        assignment = self._grow_contiguous_regions(
            local_coords=local_coords,
            adjacency=city_graph.adjacency,
            seeds=seeds,
            rng=rng,
        )
        assignment = self._enforce_contiguity(
            assignment=assignment,
            adjacency=city_graph.adjacency,
            coords=local_coords,
        )
        assignment = self._fill_empty_districts(
            assignment=assignment,
            node_ids=node_ids,
            adjacency=city_graph.adjacency,
            district_ids=list(seeds.keys()),
        )

        district_neighbors: dict[str, set[str]] = {
            did: set() for did in seeds.keys()
        }
        boundary: set[str] = set()
        for a, neighbors in city_graph.adjacency.items():
            if a not in assignment:
                continue
            da = assignment[a]
            for b in neighbors:
                if b not in assignment:
                    continue
                db = assignment[b]
                if da != db:
                    district_neighbors[da].add(db)
                    boundary.add(a)

        entry_roads: dict[str, list[str]] = defaultdict(list)
        exit_roads: dict[str, list[str]] = defaultdict(list)
        inter_district_roads: set[str] = set()
        for road in city_graph.directed_roads.values():
            if (
                road.start_intersection not in assignment
                or road.end_intersection not in assignment
            ):
                continue
            ds = assignment[road.start_intersection]
            de = assignment[road.end_intersection]
            if ds != de:
                inter_district_roads.add(road.id)
                exit_roads[ds].append(road.id)
                entry_roads[de].append(road.id)
        city_graph.inter_district_roads = inter_district_roads

        district_records: dict[str, DistrictRecord] = {}
        type_values = list(self.DISTRICT_TYPE_WEIGHTS.keys())
        type_weights = list(self.DISTRICT_TYPE_WEIGHTS.values())
        for district_id in seeds:
            members = sorted([n for n, d in assignment.items() if d == district_id])
            d_boundary = sorted([n for n in members if n in boundary])
            district_type = rng.choices(type_values, weights=type_weights, k=1)[0]
            district_records[district_id] = DistrictRecord(
                id=district_id,
                district_type=district_type,
                intersections=members,
                neighbors=sorted(district_neighbors[district_id]),
                boundary_intersections=d_boundary,
                entry_roads=sorted(set(entry_roads[district_id])),
                exit_roads=sorted(set(exit_roads[district_id])),
            )

        return DistrictData(
            intersection_to_district=assignment,
            districts=district_records,
            district_neighbors={
                k: sorted(v) for k, v in district_neighbors.items()
            },
            boundary_intersections=sorted(boundary),
            inter_district_roads=sorted(inter_district_roads),
        )

    def _farthest_seeds(
        self,
        coords: dict[str, tuple[float, float]],
        num_districts: int,
        rng: random.Random,
    ) -> dict[str, str]:
        nodes = sorted(coords.keys())
        first = rng.choice(nodes)
        selected = [first]

        while len(selected) < num_districts:
            best_node = None
            best_dist = -1.0
            for node in nodes:
                if node in selected:
                    continue
                nearest = min(
                    euclidean(coords[node], coords[s]) for s in selected
                )
                if nearest > best_dist:
                    best_dist = nearest
                    best_node = node
            if best_node is None:
                break
            selected.append(best_node)
        return {f"d_{idx:02d}": node for idx, node in enumerate(selected)}

    def _assign_nearest(
        self,
        coords: dict[str, tuple[float, float]],
        seeds: dict[str, str],
    ) -> dict[str, str]:
        assignment: dict[str, str] = {}
        for node, point in coords.items():
            district = min(
                seeds.keys(),
                key=lambda did: euclidean(point, coords[seeds[did]]),
            )
            assignment[node] = district
        return assignment

    def _grow_contiguous_regions(
        self,
        local_coords: dict[str, tuple[float, float]],
        adjacency: dict[str, set[str]],
        seeds: dict[str, str],
        rng: random.Random,
    ) -> dict[str, str]:
        districts = list(seeds.keys())
        district_sizes = {district_id: 1 for district_id in districts}
        assignment: dict[str, str] = {seed_node: district_id for district_id, seed_node in seeds.items()}
        frontiers: dict[str, deque[str]] = {
            district_id: deque([seed_node]) for district_id, seed_node in seeds.items()
        }
        remaining = set(local_coords.keys()) - set(assignment.keys())

        if not remaining:
            return assignment

        target_avg = max(1, len(local_coords) // len(districts))
        target_limits = {
            district_id: target_avg + 2 for district_id in districts
        }
        overcap = 0

        # Expand from multiple district frontiers to ensure contiguity by construction.
        frontier_order = deque(districts)
        while remaining:
            if not frontier_order:
                break
            district_id = frontier_order.popleft()
            current_frontier = frontiers[district_id]
            if not current_frontier:
                continue

            source = current_frontier.popleft()
            neighbors = [n for n in adjacency.get(source, set()) if n in remaining]
            rng.shuffle(neighbors)

            expanded = False
            for neighbor in neighbors:
                if neighbor not in remaining:
                    continue
                can_expand = (
                    overcap > 3
                    or district_sizes[district_id] < target_limits[district_id]
                )
                if not can_expand and overcap <= 3:
                    continue
                assignment[neighbor] = district_id
                remaining.remove(neighbor)
                current_frontier.append(neighbor)
                district_sizes[district_id] += 1
                frontier_order.append(district_id)
                frontier_order.append(district_id)
                expanded = True
                break

            if expanded:
                continue

            # If all districts reached targets, allow unrestricted growth to consume leftovers.
            overcap += 1
            for fallback_neighbor in neighbors:
                if fallback_neighbor not in remaining:
                    continue
                assignment[fallback_neighbor] = district_id
                remaining.remove(fallback_neighbor)
                current_frontier.append(fallback_neighbor)
                district_sizes[district_id] += 1
                frontier_order.append(district_id)
                break

            if not expanded and not current_frontier:
                # keep exploring this district only if it can still absorb nodes.
                if all(size >= target_limits[d] for d, size in district_sizes.items()):
                    continue

            frontier_order.append(district_id)

            if overcap > 10_000:
                # Safety break for unexpected stalling.
                break

        # If anything remains because of local disconnectedness in the non-gateway subgraph,
        # assign by nearest-seed fallback and rely on contiguity enforcement later.
        if remaining:
            fallback = self._assign_nearest(local_coords, seeds)
            for node in remaining:
                assignment[node] = fallback[node]
        return assignment

    def _enforce_contiguity(
        self,
        assignment: dict[str, str],
        adjacency: dict[str, set[str]],
        coords: dict[str, tuple[float, float]],
    ) -> dict[str, str]:
        district_ids = sorted(set(assignment.values()))
        changed = True
        while changed:
            changed = False
            for district_id in district_ids:
                nodes = [n for n, d in assignment.items() if d == district_id]
                if len(nodes) <= 1:
                    continue
                comps = connected_components(nodes, adjacency)
                if len(comps) <= 1:
                    continue
                comps.sort(key=len, reverse=True)
                keep = comps[0]
                for comp in comps[1:]:
                    for node in comp:
                        reassigned = self._best_neighbor_district(
                            node=node,
                            assignment=assignment,
                            adjacency=adjacency,
                            coords=coords,
                        )
                        if reassigned != district_id:
                            assignment[node] = reassigned
                            changed = True
        return assignment

    def _best_neighbor_district(
        self,
        node: str,
        assignment: dict[str, str],
        adjacency: dict[str, set[str]],
        coords: dict[str, tuple[float, float]],
    ) -> str:
        neighbors = [n for n in adjacency[node] if n in assignment]
        if not neighbors:
            return assignment[node]
        counts = Counter(assignment[n] for n in neighbors)
        best = counts.most_common(1)[0][0]
        if len(counts) == 1:
            return best
        best_score = float("inf")
        best_district = best
        for district in counts.keys():
            district_nodes = [n for n, d in assignment.items() if d == district]
            if not district_nodes:
                continue
            centroid_x = sum(coords[n][0] for n in district_nodes) / len(district_nodes)
            centroid_y = sum(coords[n][1] for n in district_nodes) / len(district_nodes)
            dist = euclidean(coords[node], (centroid_x, centroid_y))
            if dist < best_score:
                best_score = dist
                best_district = district
        return best_district

    def _fill_empty_districts(
        self,
        assignment: dict[str, str],
        node_ids: list[str],
        adjacency: dict[str, set[str]],
        district_ids: list[str],
    ) -> dict[str, str]:
        # This method now mainly keeps a lower bound on singleton-heavy districts.
        # Keep existing behavior if an empty district somehow appears.
        counts = Counter(assignment.values())
        empty = [d for d in district_ids if counts[d] == 0]
        if not empty:
            return assignment

        for empty_id in empty:
            largest = max(district_ids, key=lambda d: counts[d])
            donor_candidates = [n for n in node_ids if assignment[n] == largest]
            if not donor_candidates:
                continue
            pivot = donor_candidates[0]
            for candidate in donor_candidates:
                if any(
                    assignment[n] != largest and assignment[n] != empty_id
                    for n in adjacency.get(candidate, set())
                    if n in assignment
                ):
                    pivot = candidate
                    break
            assignment[pivot] = empty_id
            counts[largest] -= 1
            counts[empty_id] += 1

        # For any missing district, steal a boundary node from the largest district.
        for district in district_ids:
            if counts[district] > 1:
                continue
            root = next((n for n in node_ids if assignment[n] == district), None)
            if root is None:
                continue
            queue = deque([root])
            while queue and counts[district] < 2:
                current = queue.popleft()
                for candidate in node_ids:
                    if assignment[candidate] == district:
                        continue
                    if candidate not in adjacency.get(current, set()):
                        continue
                    old = assignment[candidate]
                    if counts[old] <= 2:
                        continue
                    assignment[candidate] = district
                    counts[old] -= 1
                    counts[district] += 1
                    queue.append(candidate)
                    if counts[district] >= 2:
                        break
        return assignment
