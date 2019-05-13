from common.distance import distance, distance_pt


class Cluster:
    def __init__(self, agent, already_spent_cost, seq_targets):
        self.agent = agent
        self.already_spent_cost = already_spent_cost
        self.seq_targets = seq_targets
        cost = distance(self.agent[1], self.agent[0], self.seq_targets[0][1], self.seq_targets[0][0])
        for i in range(len(self.seq_targets) - 1):
            cost += distance(self.seq_targets[i][1], self.seq_targets[i][0],
                             self.seq_targets[i + 1][1], self.seq_targets[i + 1][0])
        self.cost = cost + self.already_spent_cost

    def re_cal_cost(self):
        cost = distance_pt(self.agent, self.seq_targets[0])
        for i in range(len(self.seq_targets) - 1):
            cost += distance_pt(self.seq_targets[i], self.seq_targets[i + 1])
        self.cost = cost + self.already_spent_cost
