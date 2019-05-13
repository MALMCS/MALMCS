import numpy as np


def cal_covered_users(positions, heat_map, radius):
    """
    :param positions: $k$ positions array of !!!(y, x)!!!
    :param heat_map: grid data with count
    :param radius: 0(1 grid), 1(8 grids), 2(25 grids)
    :return: coverage score
    """
    row_num, col_num = heat_map.shape
    mask = np.zeros(heat_map.shape, dtype=int)
    for position in positions:
        center_x = position[1]
        center_y = position[0]
        max_x = center_x + radius if center_x + radius < col_num else col_num - 1
        min_x = center_x - radius if center_x - radius >= 0 else 0
        max_y = center_y + radius if center_y + radius < row_num else row_num - 1
        min_y = center_y - radius if center_y - radius >= 0 else 0
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                mask[y, x] = 1
    return np.sum(np.multiply(mask, heat_map))
