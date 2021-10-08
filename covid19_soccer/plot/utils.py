import numpy as np

def get_from_trace(var,trace):
    """ Reshapes and returns an numpy array from an arviz trace
    """
    var = np.array(trace.posterior[var])
    var = var.reshape((var.shape[0]*var.shape[1],) + var.shape[2:])
    return var