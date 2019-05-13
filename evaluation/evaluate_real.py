import timeit
from datetime import datetime
from common.distance import distance_pt
from evaluation.evaluator import Evaluator
from prediction.model_helper import Seq2seqModelHelper
from scheduling.eads_mpc_policy import EADSMPCPolicy
import numpy as np
from common.data_mask import generate_masked_heat_map

if __name__ == '__main__':
    k = 20
    r = 1
    c = 50
    start_day = datetime(2018, 1, 1)
    end_day = datetime(2018, 11, 1)
    eval_start_day = datetime(2018, 10, 1)
    time_interval = 60
    start_hour = 10
    end_hour = 22
    depot = (32, 66)
    nb_ts_per_day = 1440 // time_interval
    start_time_str = start_day.strftime('%Y%m%d')
    end_time_str = end_day.strftime('%Y%m%d')
    data_path = '../data/gt/frames_{}_{}_{}.npy'.format(start_time_str, end_time_str, nb_ts_per_day)
    data = generate_masked_heat_map(np.load(data_path))
    model_path = '../data/pred/pred_all_{}.pkl'.format(nb_ts_per_day)
    dist_func = distance_pt
    prediction_model = Seq2seqModelHelper(model_path, start_day, end_day, nb_ts_per_day, eval_start_day, start_hour)
    evaluator = Evaluator(data, start_day, end_day, eval_start_day, time_interval,
                          start_hour, end_hour, r, depot, dist_func)
    policy = EADSMPCPolicy(k, prediction_model, start_day, end_day, start_hour, end_hour,
                           time_interval, r, c, depot, dist_func)
    start = timeit.default_timer()
    tot_coverage = evaluator.evaluate(policy, k)
    stop = timeit.default_timer()
    print('Total Coverage: {}'.format(tot_coverage))
    print('Time: {} s'.format(stop - start))
