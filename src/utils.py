import numpy as np

def welford_next(arr:np.ndarray, new_point:np.float32) -> np.ndarray:
    '''
    Implementation of the Welford's online update algorithm.
    
    Parameters:
    arr: Numpy Array containing 4 values: [previous mean, previous standard deviation, previous s, number of points before current]
    new_point: Float value denoting the new point which is to be used to calculate new mean, standard deviation and s

    Return:
    Numpy Array similar to arr, except with updated values using new_point
    '''

    old_mean, _, old_s, num_points = arr

    num_points += 1
    new_mean = old_mean + (new_point - old_mean) / num_points
    new_s = old_s + (new_point - old_mean) * (new_point - new_mean)

    return [new_mean, np.sqrt(new_s / num_points) if num_points > 1 else new_s, new_s, num_points]