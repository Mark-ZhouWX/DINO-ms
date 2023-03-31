from typing import List


def get_prod_shape(level_shapes: List[List]):
    prod = []
    for one_level in level_shapes:
        h, w = one_level
        prod.append(h*w)
    return prod


def get_list_cum_sum(input_list, return_type=0):
    """
    get cum sum of input list, one-dimensional
    Args:
        input_list (List): input list
        return_type (int): return value type, 0, start, 1, end, 2, both
    """
    num_ele = len(input_list)
    res = [0]
    for e in input_list:
        res.append(e + res[-1])
    if return_type == 0:
        return res[:num_ele]
    elif return_type == 1:
        return res[1:]
    else:
        return res[:num_ele], res[1:]


def get_index_from_shapes(level_shapes: List[List], index_type=0):
    """
    get cum sum of input level_shapes
    Args:
        level_shapes (List): input level shapes
        index_type (int): return index type, 0, start, 1, end, 2, both
    """
    prod_shapes = get_prod_shape(level_shapes)
    res = get_list_cum_sum(prod_shapes, index_type)
    return res