from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary
import numpy as np
from common.distance import distance_pt
from scheduling.depot_policy import DepotPolicy
from common.temporal_idx import TemporalIdx
from datetime import datetime, timedelta


class ILPDTMPolicy(DepotPolicy):
    def __init__(self, k, predict_model, start_day, end_day, start_hour, end_hour,
                 time_interval, radius, cost_limit, depot, dist_func):
        super().__init__(depot)
        self.nb_ts_per_day = 1440 // time_interval
        self.t_idx = TemporalIdx(start_day, end_day, time_interval)
        self.predict_model = predict_model
        self.dist_func = dist_func
        self.k = k
        self.radius = radius
        self.cost_limit = cost_limit
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.time_interval = time_interval
        self.cur_day = None
        self.allocation_strategy = None

    def next_locations(self, cur_ts, cur_locations, already_spent_costs):
        cur_day = datetime.strptime(self.t_idx.ts_to_datetime(cur_ts).strftime('%Y-%m-%d'), '%Y-%m-%d')
        if cur_day != self.cur_day:
            self.cur_day = cur_day
            end_hour_ts = self.t_idx.datetime_to_ts(cur_day + timedelta(hours=self.end_hour))
            pred_num = end_hour_ts - cur_ts - 1
            pred_heat_maps = self.predict_model.predict(cur_ts, pred_num)
            self.allocation_strategy = planning_ilp(pred_heat_maps, [self.depot] * self.k, [0] * self.k,
                                                    self.cost_limit, self.radius, self.depot, self.dist_func)
        start_hour_ts = self.t_idx.datetime_to_ts(cur_day + timedelta(hours=self.start_hour))
        allocation_idx = cur_ts + 1 - start_hour_ts
        return self.allocation_strategy[allocation_idx]


def planning_ilp(pred_heat_maps, agents, already_spent_costs, energy_limitation, radius, depot, dist_func):
    # pred_heat_maps TXHXW
    T, R, C = pred_heat_maps.shape
    crowd_flows = np.zeros((R*C, T))
    for r in range(R):
        for c in range(C):
            crowd_flows[r*C+c, :] = pred_heat_maps[:, r, c]
    cost_matrix = generate_cost_matrix(R, C)
    depot_r, depot_c = depot
    depot_loc = depot_r * C + depot_c
    planning_decision = solve_ilp(crowd_flows, cost_matrix, energy_limitation, depot_loc, R*C, len(agents), T, C)
    return planning_decision


def solve_ilp(crowd_flows, cost_matrix, energy_limitation, depot, nb_locations, nb_agents, nb_time_steps, nb_cols):
    x = LpVariable.dicts('edge', [(i, j, k, t) for i in range(nb_locations) for j in range(nb_locations)
                                  for k in range(nb_time_steps) for t in range(1, nb_time_steps + 2)], 0, 1, LpBinary)
    u = LpVariable.dicts('action', [(i, k, t) for i in range(nb_locations)
                                    for k in range(nb_agents)
                                    for t in range(1, nb_time_steps + 1)], 0, 1, LpBinary)
    objective = lpSum([lpSum(lpSum(u[(i, k, t)] * crowd_flows[i, t - 1] for k in range(nb_agents))
                             for i in range(nb_locations)) for t in range(1, nb_time_steps + 1)])

    # this is quicker than 1b
    constraints = [lpSum(lpSum(x[(depot, j, k, 1)] for k in range(nb_agents)) for j in range(nb_locations)) == nb_agents,
                   lpSum(lpSum(x[(i, depot, k, nb_time_steps + 1)] for k in range(nb_agents)) for i in range(nb_locations)) == nb_agents]
    # init some variables
    for i in range(nb_locations):
        if i == depot:
            continue
        for j in range(nb_locations):
            for k in range(nb_agents):
                constraints.append(x[(i, j, k, 1)] == 0)

    # structure constraints (1c)
    for i in range(nb_locations):
        for t in range(1, nb_time_steps + 1):
            for k in range(nb_agents):
                constraints.append(lpSum(x[(h, i, k, t)] for h in range(nb_locations)) == u[(i, k, t)])
                constraints.append(lpSum(x[(i, j, k, t + 1)] for j in range(nb_locations)) == u[(i, k, t)])

    # structure constraints (1d)
    for i in range(nb_locations):
        for t in range(1, nb_time_steps + 1):
            constraints.append(lpSum(u[(i, k, t)] for k in range(nb_agents)) <= 1)

    # energy constraints (1f)
    for k in range(nb_agents):
        constraints.append(
            lpSum(lpSum(cost_matrix[i, j] * x[(i, j, k, t)] for j in range(nb_locations) for i in range(nb_locations))
                  for t in range(1, nb_time_steps + 2)) <= energy_limitation)

    prob = LpProblem('MALMCS', LpMaximize)
    prob += objective
    for constraint in constraints:
        prob += constraint
    prob.solve()

    planning_decision = []
    for t in range(1, nb_time_steps + 1):
        time_step_decision = []
        for k in range(nb_agents):
            loc = [u[(i, k, t)].varValue for i in range(nb_locations)].index(1)
            pos = get_grid_idx(loc, nb_cols)
            time_step_decision.append(pos)
        planning_decision.append(time_step_decision)
    print(lpSum(lpSum(cost_matrix[i, j] * x[(i, j, 0, t)] for j in range(nb_locations) for i in range(nb_locations))
                  for t in range(1, nb_time_steps + 2)).value())
    print(lpSum(lpSum(cost_matrix[i, j] * x[(i, j, 1, t)] for j in range(nb_locations) for i in range(nb_locations))
                  for t in range(1, nb_time_steps + 2)).value())
    return planning_decision


def generate_cost_matrix(nb_rows, nb_cols):
    nb_locations = nb_rows * nb_cols
    cost_matrix = np.zeros((nb_locations, nb_locations))
    for i in range(nb_locations):
        for j in range(nb_locations):
            loc1 = get_grid_idx(i, nb_cols)
            loc2 = get_grid_idx(j, nb_cols)
            cost_matrix[i, j] = distance_pt(loc1, loc2)
    return cost_matrix


def get_grid_idx(loc_id, nb_cols):
    col_idx = int(loc_id % nb_cols)
    row_idx = int((loc_id - col_idx) / nb_cols)
    return row_idx, col_idx