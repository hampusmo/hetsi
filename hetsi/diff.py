# Functions for differentiation

import numpy as np

def fd(data: np.ndarray, spacing: np.ndarray, axes: tuple[int], **kwargs) -> np.ndarray:
    """Uses Numpy finite differences to estimate gradients. 
    data: ndarray - Data to be differentiated.
    spacing: ndarray - Spacing of samples along specified axes.
    axes: tuple - Axes of array to operate along.
    
    Returns ndarray of shape data.ndim + len(axes)."""

    spacing = [spacing[i] for i in axes]
    data_diff = np.gradient(data, *spacing, axis = axes, **kwargs)

    if len(axes) == 1:
        return data_diff
    
    else:
        return np.stack(data_diff, axis = -1)


def diff_to_curl(diff_data: np.ndarray, dir_axis = -2, diff_axis = -1):
    """Calculates the curl of an array of estimated gradients with shape (..., dirs, dx_i)."""

    czy = diff_data[...,2,1] - diff_data[...,1,2]
    czx = diff_data[...,0,2] - diff_data[...,2,0]
    cyx = diff_data[...,1,0] - diff_data[...,0,1]
    
    return np.stack([czy, czx, cyx], axis = -1)


def diff_to_div(diff_data: np.ndarray, dir_axis = -2, diff_axis = -1):
    """Calculates the divergence of an array of estimated gradients with shape (..., dirs, dx_i)."""

    return np.trace(diff_data, axis1 = dir_axis, axis2 = diff_axis)


def diff_to_st(diff_data: np.ndarray, dir_axis = -2, diff_axis = -1):
    """Calculates the infinitesimal strain tensor of an array of estimated gradients with shape (..., dirs, dx_i)."""
    
    return (diff_data + np.moveaxis(diff_data, dir_axis, diff_axis)) / 2


def diff_to_lapl(diff2_data: np.ndarray, diff_axis1 = -2, diff_axis2 = -1):
    """Calculates the Laplacian of an array of estimated gradients with shape (..., dirs, dx_i1, dx_i2)."""
    
    return np.trace(diff2_data, axis1 = diff_axis1, axis2 = diff_axis2)

if __name__ == "__main__":
    pass