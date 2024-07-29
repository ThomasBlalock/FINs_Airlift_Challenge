import math
from airlift.solutions import Solution
from airlift.envs import ActionHelper
from airlift.envs.airport import NOAIRPORT_ID
from airlift.envs.agents import PlaneState
from airlift.envs.airlift_env import ObservationHelper

from typing import Optional, Tuple, Dict, List

import networkx as nx
from solution.common import Leg

from solution.strategic import Model


class MySolution(Solution):
    """
    Utilizing this class for your solution is required for your submission. The primary solution algorithm will go inside the
    policy function.
    """

    def __init__(self):
        super().__init__()

    def reset(self, obs, observation_spaces=None, action_spaces=None, seed=None):
        # Currently, the evaluator will NOT pass in an observation space or action space (they will be set to None)
        super().reset(obs, observation_spaces, action_spaces, seed)

        self.current_time = 0

        self.model = Model()
        self.planning = self.model.create_planning(obs)

        global_state = next(iter(obs.values()))["globalstate"]

        self.nr_agents = len(obs)
        self.latest_deadline = max(
            c.hard_deadline for c in global_state["active_cargo"]
        )

        self.path_matrices = dict()
        for plane_type, route_map in global_state["route_map"].items():
            self.path_matrices[plane_type] = PathMatrix(route_map)

    def calculate_priority(self, next_deadline: Optional[int]) -> int:
        if next_deadline is None:
            return self.nr_agents

        time_left = next_deadline - self.current_time
        total_time_left = self.latest_deadline - next_deadline
        priority = math.floor(time_left / total_time_left * self.nr_agents)
        priority = max(min(priority, self.nr_agents), 1)
        return priority

    def policies(self, obs, dones, infos):
        # Use the action helper to generate an action
        actions = {}

        global_state = next(iter(obs.values()))["globalstate"]
        new_planning = self.model.update_planning(obs)
        if new_planning is not None:
            self.planning = new_planning

        for a, agent in obs.items():
            plane = self.planning.planes[a]
            plane_state = agent["state"]
            plane_type = agent["plane_type"]
            current_airport = agent["current_airport"]
            max_weight = agent["max_weight"]
            cur_weight = agent["current_weight"]
            available_destinations = agent["available_routes"]
            cargo_at_current_airport = agent["cargo_at_current_airport"]
            cargo_onboard = agent["cargo_onboard"]
            destination = agent["destination"]

            if plane_state in (PlaneState.WAITING, PlaneState.READY_FOR_TAKEOFF):
                cargo_to_unload = []
                cargo_to_load = []
                # unload
                for cargo_id in cargo_onboard:
                    # Dump missed
                    cargo = ObservationHelper.get_active_cargo_info(
                        global_state, cargo_id
                    )
                    if cargo is None:
                        # Missed cargo
                        # print(
                        #     f"Dumping missed cargo {cargo_id} from {a} at {current_airport}"
                        # )
                        cargo_to_unload.append(cargo_id)
                        for plane in self.planning.planes.values():
                            # TODO This is slow
                            new_legs = []
                            for legs in plane.legs:
                                filtered_ce = [
                                    ce
                                    for ce in legs.cargo_edges
                                    if ce.cargo_id != cargo_id
                                ]
                                if len(filtered_ce) > 0:
                                    new_legs.append(Leg.construct(filtered_ce))
                            plane.legs = new_legs
                        continue

                    # Unload cargo where the next CargoEdge is not an available destination
                    # TODO: this will happen if there is mal, not ideal

                    if plane.has_legs():
                        if all(
                            ce.cargo_id != cargo_id for ce in plane.legs[0].cargo_edges
                        ):
                            cargo_to_unload.append(cargo_id)
                            cur_weight -= cargo.weight
                    else:
                        cargo_to_unload.append(cargo_id)
                        cur_weight -= cargo.weight

                # load
                for cargo in (
                    ObservationHelper.get_active_cargo_info(
                        global_state, cargo_at_current_airport
                    )
                    or []
                ):
                    if plane.has_legs():
                        for ce in plane.legs[0].cargo_edges:
                            if (
                                ce.origin == current_airport
                                and ce.cargo_id == cargo.id
                                and cargo.weight <= max_weight - cur_weight
                            ):
                                # If there is a CargoEdge from this airport for this cargo, load
                                cargo_to_load.append(cargo.id)
                                cur_weight += cargo.weight

                # load all cargo at once
                if plane.has_legs():
                    cargo_missing = len(cargo_to_load) < len(plane.legs[0].cargo_edges)
                    can_wait = all(
                        self.current_time < ce.lp
                        for ce in plane.legs[0].cargo_edges
                        if ce.cargo_id in cargo_to_load
                    )
                    if cargo_missing and can_wait:
                        cargo_to_load = []

                priority = self.calculate_priority(plane.get_next_deadline())
                if len(cargo_to_load) > 0:
                    print(
                        f"[{self.current_time}] Loading {cargo_to_load} on {a} at {current_airport} lp {plane.legs[0].lp}"
                    )

                if len(cargo_to_unload) > 0:
                    ce_to_unload = [
                        ce
                        for ce in self.planning.cargo_edges.cargo_edges
                        if ce.cargo_id in cargo_to_unload
                        and ce.origin == current_airport
                    ]
                    next_cargo_deadline = min(
                        (ce.lp for ce in ce_to_unload), default=None
                    )

                    priority = min(
                        priority, self.calculate_priority(next_cargo_deadline)
                    )
                    print(
                        f"[{self.current_time}] Unloading {cargo_to_unload} from {a} at {current_airport} lp {next_cargo_deadline}"
                    )

                actions[a] = {
                    "priority": priority,
                    "cargo_to_load": cargo_to_load,
                    "cargo_to_unload": cargo_to_unload,
                    "destination": NOAIRPORT_ID,
                }

            if (
                plane_state == PlaneState.READY_FOR_TAKEOFF
                and len(cargo_to_load) + len(cargo_to_unload) == 0
            ):

                ce_onboard = []
                should_depart = False
                if plane.has_legs():
                    ce_onboard = [
                        ce
                        for ce in plane.legs[0].cargo_edges
                        if ce.cargo_id in cargo_onboard
                    ]
                    # If it is time to dispatch any CE on board, then dispatch and break out
                    should_depart = len(ce_onboard) == len(
                        plane.legs[0].cargo_edges
                    ) or any(ce for ce in ce_onboard if self.current_time >= ce.lp + 20)

                # destination from CargoEdge on board
                destination = NOAIRPORT_ID
                if should_depart:
                    ce = plane.legs[0].cargo_edges[0]
                    destination = ce.destination
                    actions[a] = {
                        "priority": self.calculate_priority(plane.get_next_deadline()),
                        "cargo_to_load": [],
                        "cargo_to_unload": [],
                        "destination": ce.destination,
                    }
                    print(
                        f"[{self.current_time}] Sending {a} to {destination} for {ce}"
                    )
                    for ce in ce_onboard:
                        if ce.destination == destination:
                            # Remove them from legs as they are being executed
                            plane.legs[0].remove(ce)
                        else:
                            print(
                                f"WARNING: plane being dispatched to {destination} with {ce} onboard"
                            )
                    if len(plane.legs[0].cargo_edges) == 0:
                        plane.legs.pop(0)

                # destination to first CargoEdge origin assigned
                if len(ce_onboard) == 0 and plane.has_legs():
                    # If plane is empty find the first CargoEdge assigned to the plane and go that way
                    for ce in plane.legs[0].cargo_edges:
                        if ce.origin == current_airport:
                            # Stay here until the cargo is loaded
                            destination = NOAIRPORT_ID
                            actions[a] = {
                                "priority": self.calculate_priority(
                                    plane.get_next_deadline()
                                ),
                                "cargo_to_load": [],
                                "cargo_to_unload": [],
                                "destination": NOAIRPORT_ID,
                            }
                            # print(f"{a} waiting at {current_airport} for {ce}")
                            break

                        path = self.path_matrices[plane_type].get_path(
                            current_airport, ce.origin
                        )
                        if path[1] in available_destinations:
                            # Head to it
                            destination = path[1]
                            actions[a] = {
                                "priority": self.calculate_priority(
                                    plane.get_next_deadline()
                                ),
                                "cargo_to_load": [],
                                "cargo_to_unload": [],
                                "destination": destination,
                            }
                            print(
                                f"[{self.current_time}] Sending {a} on {path} to pickup {ce}"
                            )
                            break
                        else:
                            print(f"WARNING: {a} cannot go on {path} to pickup {ce}")
            if a not in actions:
                noop_action = ActionHelper.noop_action()
                noop_action["priority"] = self.calculate_priority(
                    plane.get_next_deadline()
                )
                actions[a] = noop_action
        self.current_time += 1
        return actions


class PathMatrix:
    def __init__(self, graph: nx.graph) -> None:
        self.graph: nx.graph = graph
        self._matrix: Dict[Tuple[int, int], List[int]] = {}

    def _compute_shortest_path(self, orig: int, dest: int, from_to: Tuple[int, int]):
        path = nx.shortest_path(self.graph, orig, dest, weight="cost")
        self._matrix[from_to] = path
        for sub_ind in range(1, len(path) - 1):
            sub_from_to = (path[sub_ind], dest)
            self._matrix[sub_from_to] = path[sub_ind : len(path)]

    def get_path(self, orig: int, dest: int):
        from_to = (orig, dest)
        if from_to not in self._matrix:
            self._compute_shortest_path(orig, dest, from_to)
        return self._matrix[from_to]
