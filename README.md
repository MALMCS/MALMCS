# Dynamic Public Resource Allocation based on Human Mobility Prediction

## Data Description

### Folder `gt`

`frames_20180101_20181101_24.npy`: hourly crowd flows (Tx51x108)

`masked_agg_frames_5_10_20180101_20181101_24.npy`: aggregated hourly crowd flows (Tx5x10)

`ext/holiday_20180101_20181101_24.npy`: holiday features

`ext/price_20180101_20181101_24.npy`: promotion features

`ext/tod_20180101_20181101_24.npy`: time of day features

### Folder `pred`

`pred_all_24.pkl` and `pred_all_agg_24.pkl`

prediction results from the seq2seq+ext model for evaluation acceleration.

## Prediction Model

`model.py`: seq2seq+ext model for crowd flow prediction

## Scheduling Algorithms

### Energy Adaptive Scheduling (EADS)

`eads_policy.py`

### Integer Linear Programming (ILP) 

`ilp_policy.py`

## Evaluation

`evaluate_real.py`: evaluate the scheduling algorithm based on the real crowd flows.

`evaluate_small.py`: evaluate the scheduling algorithm based on the aggregated crowd flows.
