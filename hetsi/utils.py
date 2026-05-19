# General utilities
import numpy as np
import torch
import hetsi.diff as diff
from os import mkdir
from os.path import isdir
import json

def gen_coord_grid(dataset, spacing, dtype = np.float64, strip_excess = False, normalize = False):
    idx = np.indices(dataset.shape, dtype = dtype)
    idx = np.moveaxis(idx, 0, -1)
    si = [1 for i in range(dataset.ndim)] + [idx.shape[-1]]
    idx *= spacing.reshape(si)

    if normalize:
        idx = idx / np.amax(idx) # 0-1 Scaling, preserving inter-feature consistency
        
    return idx


def autograd_grad(p, x, higher_order_grad = True):
    """Calculates the gradient of parameters using the PyTorch autograd framework. Note that x must have requires_grad = True. Higher order grad indicates that gradients of the gradient may be calculated, else it is treated as a constant in the optimizer."""
    
    if p.dtype in (torch.complex32, torch.complex64, torch.complex128, torch.complex):
        dgr = torch.autograd.grad(p.real, 
                                x, 
                                grad_outputs = torch.ones_like(p.real), 
                                retain_graph=True, 
                                create_graph = higher_order_grad)[0]
        
        dgi = torch.autograd.grad(p.imag, 
                                x, 
                                grad_outputs = torch.ones_like(p.imag), 
                                retain_graph=True, 
                                create_graph = higher_order_grad)[0] 
        
        dg = torch.complex(dgr, dgi)
    
    else:
        dg = torch.autograd.grad(p, 
                                x, 
                                grad_outputs = torch.ones_like(p), 
                                retain_graph=True, 
                                create_graph = higher_order_grad)[0]
    
    return dg
    

def autograd_grad_vmap(p, x, higher_order_grad = True, chunk_size = 2048):
    """Vectorized grad calculation for very large inputs. """
    
    dg = torch.vmap(lambda d, y: autograd_grad(d, y, higher_order_grad = higher_order_grad),
                    in_dims=0,
                    out_dims=0,
                    chunk_size=chunk_size)(p, x)
    
    return dg

def autograd_div(p: torch.Tensor, x: torch.Tensor, higher_order_grad = True):
    dg = autograd_grad(p, x, higher_order_grad)
    sel = torch.tensor([0,1,2], dtype = torch.int64)
    return torch.sum(dg[..., sel, sel], dim = -1)


def reshape_data(data: torch.Tensor, reshape_dims = None, fixed_dims = (0,4)):
    """Reshapes input data to match the form sought by the algorithm. If reshape_dims is provided, fixed_dims are ignored.
    Returns the data in new shape with all reshape dims collapsed into the first dimension, and all others maintain their order.
    Fixed dims specify dims which survive instead.
    
    Example: Array: (A,B,C,D), reshape_data(Array, reshape_dims = (0,2) -> return (A*C, B, D)).
    Example 2: Array: (A,B,C,D), reshape_data(Array, fixed_dims = (0,2) -> return (B * D, A, C))"""

    dims = list(range(data.ndim))

    if reshape_dims is not None:
        fix = tuple(i for i in dims if i not in reshape_dims)
        move = tuple(i for i in dims if i in reshape_dims)

    else:
        fix = tuple(i for i in dims if i in fixed_dims)
        move = tuple(i for i in dims if i not in fixed_dims)

    return data.permute(move + fix).flatten(0, len(move)-1)

def _dist_check(x0, x, r):
    """Broadcastable distance check, returns bool array of Euclidean distance between x and x_0 if < r."""
    
    output = []

    for curr_x0, curr_r in zip(x0, r):
        d = np.sum((x - curr_x0) ** 2, axis = -1)
        output.append(d <= (curr_r ** 2))
    
    return output

def gen_4tp_mask(w_ratio = 1., dist_corr = 5e-4):
    """Generate an insert mask for the 4-Target FEM phantom from the BIOQIC group.
    returns: mask - ndarray[Bool], based on visual inspection of input data."""

    # Phantom parameters
    pshape = (100,80)
    pdims = (1,2) # Dimensions in data corresponding, for creating singleton dimensions
    pspacing = np.array([1,1], dtype = np.float64) * 1e-3 
    x_map = np.indices(pshape, dtype = np.float64) * pspacing[:, None, None]
    x_map = np.moveaxis(x_map, 0, -1)

    # Phantom assumed to be symmetric in z, only check in x,y
    # Center aligned with middle in dim 1
    # Dim 0 set by visual inspection.
    
    x0_list = [np.array([x_map[17,0,0], x_map[-1,-1, 1]/2 + 0.5e-3]),
               np.array([x_map[31,0,0], x_map[-1,-1, 1]/2 + 0.5e-3]),
               np.array([x_map[49,0,0], x_map[-1,-1, 1]/2 + 0.5e-3]),
               np.array([x_map[74,0,0], x_map[-1,-1, 1]/2 + 0.5e-3])]
    
    r_list = (np.array([0.002, 0.004, 0.01, 0.020]) / 2) * w_ratio + dist_corr # r-values from Barnhill publication

    masks = _dist_check(x0_list, x_map, r_list)
    
    # Background mask should not rely on 
    masks_for_matrix = _dist_check(x0_list, x_map, r_list / w_ratio)
    bmask = np.logical_not(np.sum(np.stack(masks_for_matrix, axis = 0), axis = 0))
    masks.append(torch.tensor(bmask, dtype = torch.float64))
    
    # Expand for broadcasting
    pshape = (1,100,80)
    nshape = tuple(1 if i not in pdims else pshape[i] for i in range(6))
    out = []

    for m in masks:
        out.append(m.reshape(nshape))

    return masks

def fft_butter_radial(data, order = 4, freqs = 100, mode = "lp", dims = (1,2,3), spacing = np.array([1,1,1]) * 1e-3):

    assert mode.lower() in ("lp", "hp", "bp"), "Filter modes can only be lp, hp or bp."
    assert len(dims) == len(spacing), "Length of spacing must match length of dims."

    dfft = np.fft.fftn(data, axes = dims)

    k_shape = np.ones(dfft.ndim, dtype = int)

    ks = [np.fft.fftfreq(dfft.shape[dims[i]], spacing[i]) for i in range(len(dims))]
    kg = np.meshgrid(*ks, indexing="ij")
    kg = np.stack(kg, axis = -1)
    k = np.linalg.norm(kg, axis = -1)

    for i in range(len(dims)):
        k_shape[dims[i]] = k.shape[i]

    k = k.reshape(k_shape)

    if mode == "bp":
        f_lp = 1. / (np.sqrt(1 + (1j * k / freqs[0]) ** (2 * order[0])))
        f_hp = 1. / (np.sqrt(1 + (1j * k / freqs[1]) ** (2 * order[1])))
        f_hp = 1. - f_hp
        f_win = f_lp * f_hp

    if mode == "lp":
        f_lp = 1. / (np.sqrt(1 + (1j * k / freqs) ** (2 * order)))
        f_win = f_lp
    
    if mode == "hp":
        f_hp = 1. / (np.sqrt(1 + (1j * k / freqs) ** (2 * order)))
        f_hp = 1. - f_hp
        f_win = f_hp
    
    dfft = dfft * f_win

    return np.fft.ifftn(dfft, axes = dims)

def fftFreqFilter(dataset, t_dim = -2, freq_idxs = [1,]):
    """Filters the dataset such that only the frequencies at "freq_idxs" are maintained."""

    data_fft = np.fft.rfft(dataset, axis = t_dim)
    data_fft = np.moveaxis(data_fft, t_dim, -1)
    data_save = data_fft[..., freq_idxs].copy()

    data_fft[...] = 0.
    data_fft[...,freq_idxs] = data_save
    data_fft = np.moveaxis(data_fft, -1, t_dim)

    return np.fft.irfft(data_fft, axis = t_dim)

def modified_gini_coefficient(data):
    # Basis axis is final axes

    data_i = data[..., None]
    data_j = data_i.transpose(-1,-2)
    delta = torch.sum(torch.abs(data_i.abs() - data_j.abs()), dim = (-1,-2))
    g = delta / (2 * data.shape[-1] * torch.sum(data.abs(), dim = -1))
    return data.shape[-1] / (data.shape[-1] - 1) * g # Including bias correction

def save_results(target_path, model_state = None, optimizer_state = None, prediction = None, bases = None, targets = None, inputs = None, results = None, parameters = None):
    if not target_path.endswith("/"):
        target_path += "/"

    if not isdir(target_path):
        mkdir(target_path)

    if model_state is not None:
        torch.save(model_state, target_path + "model.pth")
    if optimizer_state is not None:
        torch.save(optimizer_state, target_path + "opt.pth")
    if prediction is not None:
        torch.save(prediction, target_path + "pred.pth")
    if bases is not None:
        torch.save(bases, target_path + "base.pth")
    if targets is not None:
        torch.save(targets, target_path + "target.pth")
    if inputs is not None:
        torch.save(inputs, target_path + "input.pth")
    
    if results is not None:
        with open(target_path + "losses.json", "w") as file:
            json.dump(results, file)

    # Parmeters to JSON
    ser_params = {}
    
    if parameters is not None:
        for key, value in parameters.items():
            if hasattr(value, "to_json"):
                ser_params[key] = value
            else:
                ser_params[key] = str(value)

        with open(target_path + "config.json", "w") as file:
            strrep = json.dumps(ser_params, separators=(",", ":"))
            file.write(strrep)

if __name__ == "__main__":
    pass