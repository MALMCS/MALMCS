import numpy as np


def select_each_frame_best_locations(k, radius, predicted_heat_maps):
    """
    init future positions to the maximal k coverage positions in each time stamp
    :param k: agent number
    :param radius: agent influence radius
    :param predicted_heat_maps: predicted heat maps (next timeslot to the end of the day)
    :return: T x k future positions
    """
    T = predicted_heat_maps.shape[0]
    future_positions = []
    for i in range(T):
        best_positions = select_top_k_positions(k, predicted_heat_maps[i, :, :], radius)
        future_positions.append(best_positions)
    return future_positions


def select_top_k_positions(k, heat_map, radius):
    top_k_positions = []
    updated_heat_map = heat_map.copy()
    aux_map = np.zeros(updated_heat_map.shape, dtype=int)
    row_num, col_num = updated_heat_map.shape
    for i in range(row_num):
        for j in range(col_num):
            aux_map[i, j] = cal_influence_count(updated_heat_map, j, i, radius)
    while k > 0:
        best_pos = np.unravel_index(np.argmax(aux_map, axis=None), aux_map.shape)
        top_k_positions.append((best_pos[0], best_pos[1]))
        aux_map = update_aux_map(aux_map, updated_heat_map, best_pos, radius)
        k -= 1
    return top_k_positions


def cal_influence_count(heat_map, pos_x, pos_y, radius):
    cnt = 0
    row_num, col_num = heat_map.shape
    max_x = pos_x + radius if pos_x + radius < col_num else col_num - 1
    min_x = pos_x - radius if pos_x - radius >= 0 else 0
    max_y = pos_y + radius if pos_y + radius < row_num else row_num - 1
    min_y = pos_y - radius if pos_y - radius >= 0 else 0
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            cnt += heat_map[y, x]
    return cnt


def update_aux_map(aux_map, heat_map, best_pos, radius):
    row_num, col_num = aux_map.shape
    center_y = best_pos[0]
    center_x = best_pos[1]
    max_x = center_x + radius if center_x + radius < col_num else col_num - 1
    min_x = center_x - radius if center_x - radius >= 0 else 0
    max_y = center_y + radius if center_y + radius < row_num else row_num - 1
    min_y = center_y - radius if center_y - radius >= 0 else 0
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            heat_map[y, x] = 0
    # re-calculate aux for grids influenced by influenced grids
    max_x_recal = max_x + radius if max_x + radius < col_num else col_num - 1
    min_x_recal = min_x - radius if min_x - radius >= 0 else 0
    max_y_recal = max_y + radius if max_y + radius < row_num else row_num - 1
    min_y_recal = min_y - radius if min_y - radius >= 0 else 0
    for i in range(min_y_recal, max_y_recal + 1):
        for j in range(min_x_recal, max_x_recal + 1):
            aux_map[i, j] = cal_influence_count(heat_map, j, i, radius)
    return aux_map


def select_best_location(candidates, heat_map, radius):
    """
    select maximal coverage locations among candidates
    :param candidates:
    :param heat_map:
    :param radius:
    :return:
    """
    best_candidate_with_count = max(
        map(lambda candidate: (candidate, cal_influence_count(heat_map, candidate[1], candidate[0], radius)),
            candidates), key=lambda x: x[1])
    return best_candidate_with_count[0]


def select_best_location_with_score(candidates, heat_map, radius):
    best_candidate_with_count = max(
        map(lambda candidate: (candidate, cal_influence_count(heat_map, candidate[1], candidate[0], radius)),
            candidates), key=lambda x: x[1])
    return best_candidate_with_count


def cal_remaining_heat_map(heat_map, positions, radius):
    """
    calculate heat map after area served by positions with radius
    :param heat_map: the original heat map
    :param positions: the agent positions !!!(y, x)!!!
    :param radius: the service radius
    :return: the remaining heat map
    """
    remaining_heat_map = heat_map.copy()
    for position in positions:
        update_heat_map(remaining_heat_map, position, radius)
    return remaining_heat_map


def update_heat_map(heat_map, best_pos, radius):
    row_num, col_num = heat_map.shape
    center_y = best_pos[0]
    center_x = best_pos[1]
    max_x = center_x + radius if center_x + radius < col_num else col_num - 1
    min_x = center_x - radius if center_x - radius >= 0 else 0
    max_y = center_y + radius if center_y + radius < row_num else row_num - 1
    min_y = center_y - radius if center_y - radius >= 0 else 0
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            heat_map[y, x] = 0


def cal_influence_count(heat_map, pos_x, pos_y, radius):
    cnt = 0
    row_num, col_num = heat_map.shape
    max_x = pos_x + radius if pos_x + radius < col_num else col_num - 1
    min_x = pos_x - radius if pos_x - radius >= 0 else 0
    max_y = pos_y + radius if pos_y + radius < row_num else row_num - 1
    min_y = pos_y - radius if pos_y - radius >= 0 else 0
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            cnt += heat_map[y, x]
    return cnt