from datetime import timedelta
from common.temporal_idx import TemporalIdx
from evaluation.coverage_evaluation import cal_covered_users


class Evaluator:
    """
    this evaluator is common, it can test any policy
    """
    def __init__(self, data, start_day, end_day, eval_start_day, time_interval,
                 start_hour, end_hour, radius, depot, dist_func):
        self.data = data
        self.t_idx = TemporalIdx(start_day, end_day, time_interval)
        self.end_day = end_day
        self.eval_start_day = eval_start_day
        self.time_interval = time_interval
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.radius = radius
        self.depot = depot
        self.dist_func = dist_func

    def evaluate(self, policy, k):
        acc_coverage = 0
        day = self.eval_start_day
        start_hour_offset = int(self.start_hour * (60 / self.time_interval))
        eval_ts_num = int((self.end_hour - self.start_hour) * (60 / self.time_interval))
        end_day = self.end_day
        while day < end_day:
            print(day)
            already_spent_costs = [0] * k
            day_ts = self.t_idx.datetime_to_ts(day)
            day_coverage = 0
            # before service time
            positions = [self.depot] * k
            whole_day_routine = [[init_loc] for init_loc in positions]
            # decide first location
            next_positions = policy.next_locations(day_ts + start_hour_offset - 1, positions, already_spent_costs)
            for j in range(k):
                dis = self.dist_func(positions[j], next_positions[j])
                already_spent_costs[j] += dis
                whole_day_routine[j].append(next_positions[j])
            positions = next_positions
            for i in range(eval_ts_num):
                cur_true_heat_map = self.data[day_ts + start_hour_offset + i, :, :]
                day_coverage += cal_covered_users(positions, cur_true_heat_map, self.radius)
                if i < eval_ts_num - 1:
                    next_positions = policy.next_locations(day_ts + start_hour_offset + i, positions, already_spent_costs)
                    for j in range(k):
                        dis = self.dist_func(positions[j], next_positions[j])
                        already_spent_costs[j] += dis
                        whole_day_routine[j].append(next_positions[j])
                    positions = next_positions
            print("energy consumption: ")
            for j in range(k):
                # add return to depot cost
                already_spent_costs[j] += self.dist_func(positions[j], self.depot)
                print("agent {0}: {1}".format(j, already_spent_costs[j]))
            print('whole day routine:')
            for j in range(k):
                whole_day_routine[j].append(self.depot)
                print(whole_day_routine[j])
            print('day coverage:{}'.format(day_coverage))
            acc_coverage += day_coverage
            day += timedelta(days=1)
        return acc_coverage
