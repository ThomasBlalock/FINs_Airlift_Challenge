from collections import OrderedDict
import math
from typing import Dict, Iterable, List, Optional, Tuple
from airlift.envs.airlift_env import ObservationHelper
from solution.common import (
    CargoEdge,
    CargoEdges,
    Leg,
    PathCache,
    Plane,
    PlaneTypeMap,
    Planning,
)


class Model:
    def __init__(self) -> None:
        pass

    def create_planning(self, obs) -> Planning:
        global_state = next(iter(obs.values()))["globalstate"]
        self.processing_time = global_state["scenario_info"][0].processing_time
        graph = ObservationHelper.get_multidigraph(global_state)
        self.paths = PathCache(graph)
        self.plane_type_map = PlaneTypeMap(global_state["route_map"])
        self.cargo_edges = self._create_cargo_edges(obs)
        self.planes = self._create_assignments(obs)
        # print_cargo_edges(self.cargo_edges)
        # print_planes(self.planes.values())
        return Planning(self.cargo_edges, self.planes)

    def update_planning(self, obs) -> Optional[Planning]:
        # TODO: we should probably put the planning in the model so it's more straightforward
        # Update the palnning for new cargo
        global_state = next(iter(obs.values()))["globalstate"]
        new_cargos = global_state["event_new_cargo"]
        if len(new_cargos) > 0:
            self.cargo_edges = self._add_cargo_edges_from_cargos(
                self.cargo_edges, new_cargos
            )
            self.planes = self._create_assignments(obs)
            return Planning(self.cargo_edges, self.planes)
        # TODO: Do something to return a modified planning if malfunctions happen...

    def _create_cargo_edges(self, obs) -> CargoEdges:
        global_state = next(iter(obs.values()))["globalstate"]
        cargo_edges = self._add_cargo_edges_from_cargos(
            CargoEdges(), global_state["active_cargo"]
        )
        return cargo_edges

    def _add_cargo_edges_from_cargos(
        self, cargo_edges: CargoEdges, cargos
    ) -> CargoEdges:
        for cargo in cargos:
            shortest_path = self.paths.get_path(cargo.location, cargo.destination)

            earliest_pickup = cargo.earliest_pickup_time

            # travel forward
            for orig, dest in zip(shortest_path[1:], shortest_path[:-1]):
                travel_time = self.paths.get_travel_time(orig, dest)
                earliest_pickup += (
                    self.processing_time + travel_time + self.processing_time
                )

            latest_pickup = cargo.soft_deadline
            sequence = len(shortest_path) - 1
            # travel backward
            for orig, dest in zip(shortest_path[-2::-1], shortest_path[::-1]):
                travel_time = self.paths.get_travel_time(orig, dest)
                earliest_pickup -= (
                    self.processing_time + travel_time + self.processing_time
                )
                latest_pickup -= (
                    self.processing_time + travel_time + self.processing_time
                )
                cargo_edges.add(
                    CargoEdge(
                        cargo.id,
                        orig,
                        dest,
                        travel_time + self.processing_time,
                        sequence,
                        earliest_pickup,
                        latest_pickup,
                        cargo.weight,
                        self.plane_type_map.get_allowable_plane_types(orig, dest),
                    )
                )
                sequence -= 1
        return cargo_edges

    def _create_assignments(self, obs) -> Dict[str, Plane]:
        ce_plane_map: Dict[Tuple[int, int], str] = dict()
        planes: Dict[int, Plane] = dict()
        for a_id, agent in obs.items():
            planes[a_id] = Plane(
                a_id,
                agent["current_airport"],
                agent["current_airport"],
                agent["plane_type"],
                agent["max_weight"],
            )

        for ce in sorted(
            self.cargo_edges.cargo_edges,
            key=lambda ce: (math.floor(ce.ep / 30), ce.sequence),
        ):
            sorted_planes = sorted(
                [p for p in planes.values() if p.type in ce.allowed_plane_types],
                key=lambda p: p.matches(ce, self.paths),
            )
            found = False
            for plane in sorted_planes:
                if plane.can_service(ce, self.paths, self.plane_type_map):
                    (ep_diff_ce, lp_diff_ce, ep_diff_leg, lp_diff_leg) = (
                        plane.add_cargo_edge(ce, self.paths)
                    )
                    ce_plane_map[(ce.cargo_id, ce.sequence)] = plane.id
                    leg = plane.legs[-1]
                    self.update_ep_lp(
                        (ep_diff_ce, lp_diff_ce, ep_diff_leg, lp_diff_leg),
                        ce,
                        leg,
                        planes,
                        ce_plane_map,
                    )
                    found = True
                    break
            if not found:
                print(f"No plane found for ce {ce}")

        return planes

    def update_ep_lp(
        self,
        changes: Tuple[int, int, int, int],
        cur_ce: CargoEdge,
        cur_leg: Leg,
        planes: List[Plane],
        ce_plane_map: Dict[Tuple[int, int], str],
    ) -> None:
        (ep_diff_ce, lp_diff_ce, ep_diff_leg, lp_diff_leg) = changes

        orig_leg_ep = cur_leg.ep - ep_diff_leg
        orig_leg_lp = cur_leg.lp + lp_diff_leg

        if ep_diff_ce > 0:
            for ce in self.cargo_edges.cargo_edges:
                if ce.cargo_id == cur_ce.cargo_id and ce.sequence > cur_ce.sequence:
                    ce.ep += ep_diff_ce

        if ep_diff_leg > 0:
            ce_seq_to_propagate: OrderedDict[Tuple[int, int], int] = OrderedDict()
            for ce in cur_leg.cargo_edges:
                if ce.cargo_id != cur_ce.cargo_id:
                    already_added = max(0, orig_leg_ep - ce.ep)
                    to_add = max(0, ep_diff_leg - already_added)
                    if to_add > 0:
                        ce_seq_to_propagate[ce.cargo_id, ce.sequence + 1] = to_add
            for ce_seq, to_add in ce_seq_to_propagate.items():
                if ce_seq not in ce_plane_map:
                    for ce in self.cargo_edges.cargo_edges:
                        if ce.cargo_id == ce_seq[0] and ce.sequence >= ce_seq[1]:
                            ce.ep += to_add

            while len(ce_seq_to_propagate) > 0:
                ce_seq, to_add = ce_seq_to_propagate.popitem(last=False)
                if ce_seq in ce_plane_map:
                    plane = planes[ce_plane_map[ce_seq]]
                    leg = plane.find_leg(ce_seq)
                    leg.ep += ep_diff_leg
                    for ce in leg.cargo_edges:
                        cur_ce_seq = (ce.cargo_id, ce.sequence)
                        if cur_ce_seq in ce_seq_to_propagate:
                            del ce_seq_to_propagate[cur_ce_seq]
                        ce_seq_to_propagate[ce.cargo_id, ce.sequence + 1] = ep_diff_leg

        if lp_diff_ce > 0:
            ce_seq_to_propagate: List[Tuple[int, int, int]] = [
                (cur_ce.cargo_id, cur_ce.sequence - 1, lp_diff_ce)
            ]
            while len(ce_seq_to_propagate) > 0:
                ce_seq = ce_seq_to_propagate.pop()
                if (ce_seq[0], ce_seq[1]) in ce_plane_map:
                    plane = planes[ce_plane_map[ce_seq[0], ce_seq[1]]]
                    leg = plane.find_leg(ce_seq)

                    for ce in leg.cargo_edges:
                        if ce.corresponds(ce_seq):
                            already_subtracted = max(0, ce.lp - leg.lp)
                            to_subtract = max(0, ce_seq[2] - already_subtracted)
                            leg.lp -= to_subtract
                            break
                    if to_subtract > 0:
                        for ce in leg.cargo_edges:
                            ce_seq_to_propagate.append(
                                (ce.cargo_id, ce.sequence - 1, to_subtract)
                            )

        if lp_diff_leg > 0:
            ce_seq_to_propagate: OrderedDict[Tuple[int, int], int] = OrderedDict()
            for ce in cur_leg.cargo_edges:
                if ce.cargo_id != cur_ce.cargo_id:
                    already_subtracted = max(0, ce.lp - orig_leg_lp)
                    to_subtract = max(0, lp_diff_leg - already_subtracted)
                    if to_subtract > 0:
                        ce_seq_to_propagate[ce.cargo_id, ce.sequence - 1] = to_subtract

            while len(ce_seq_to_propagate) > 0:
                ce_seq, to_subtract = ce_seq_to_propagate.popitem(last=False)
                if ce_seq in ce_plane_map:
                    plane = planes[ce_plane_map[ce_seq]]
                    leg = plane.find_leg(ce_seq)
                    for ce in leg.cargo_edges:
                        if ce.corresponds(ce_seq):
                            already_subtracted = max(0, ce.lp - leg.lp)
                            to_subtract_cur = max(0, lp_diff_leg - already_subtracted)
                            leg.lp -= to_subtract
                            break
                    if to_subtract_cur > 0:
                        for ce in leg.cargo_edges:
                            ce_seq_to_propagate[ce.cargo_id, ce.sequence - 1] = (
                                to_subtract_cur
                            )
                            cur_ce_seq = (ce.cargo_id, ce.sequence)
                            if cur_ce_seq in ce_seq_to_propagate:
                                del ce_seq_to_propagate[cur_ce_seq]


def print_cargo_edges(cargo_edges: CargoEdges) -> None:
    for ce in cargo_edges.cargo_edges:
        print(
            f"{ce.cargo_id};{ce.origin};{ce.destination};{ce.duration};{ce.sequence};{ce.ep};{ce.lp};{ce.weight};{ce.allowed_plane_types}"
        )


def print_planes(planes: Iterable[Plane]) -> None:
    cnt = 0
    for plane in planes:
        for leg in plane.legs:
            print("-")
            for ce in leg.cargo_edges:
                cnt += 1
                print(
                    f"{plane.id};{ce.cargo_id};{ce.origin};{ce.destination};{ce.ep};{ce.lp}"
                )
    print(f"Planned {cnt} cargo edges")
