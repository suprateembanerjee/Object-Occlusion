import numpy as np

def welford_next(arr:np.ndarray, new_point:np.float32) -> np.ndarray:
    '''
    Implementation of the Welford's online update algorithm.
    
    Parameters:
    arr: Numpy array containing 4 values: [previous mean, previous standard deviation, previous s, number of points before current]
    new_point: Float value denoting the new point which is to be used to calculate new mean, standard deviation and s
    Return:
    Numpy Array similar to arr, except with updated values using new_point
    '''

    old_mean, _, old_s, num_points = arr

    num_points += 1
    new_mean = old_mean + (new_point - old_mean) / num_points
    new_s = old_s + (new_point - old_mean) * (new_point - new_mean)

    return [new_mean, np.sqrt(new_s / num_points) if num_points > 1 else new_s, new_s, num_points]

def welford(history:np.ndarray, frame:np.ndarray) -> np.ndarray:
    '''
    This implementation of Welford Algorithm takes past image data in a history array, and a new frame
    from a video sequence to compute an updated history array based on the values from the frame.

    Parameters:
    history: Numpy array of shape [h, w, c, x] 
             where x = [old mean, old std, old s, number of frames encountered before]
    frame: Numpy array of shape [h, w, c]

    Return:
    Numpy array of shape [h, w, c, x] containing updated mean, std, s and num_points values
    '''

    old_mean, _, old_s, num_points = np.transpose(history, [3,0,1,2])

    num_points += 1.
    new_mean = old_mean + (frame - old_mean) / num_points
    new_s = old_s + (frame - old_mean) * (frame - new_mean)

    # mask = num_points > 1
    # new_std = np.sqrt(new_s / num_points) * mask + new_s * np.logical_not(mask)

    # This is an optimization that is based on the prior that all pixels have equal
    # length of history (frames seen before), and saves compute by ~ 6 to 10%
    new_std = np.sqrt(new_s / num_points) if num_points[0][0][0] > 1 else new_s

    return np.transpose(np.array([new_mean, new_std, new_s, num_points]), [1,2,3,0])