import os, sys

sys.path.append("..") # Add top level dir

import hetsi as hsi
import numpy as np
import torch
import scipy.io as spio
import time

def main():
    tstart = time.time()

    fpath = "..." # Input data path
    tpath = "../outputs/brainfem/" # Target data path (outputs)

    raw = spio.loadmat(fpath + "/BrainSimDisplacement.mat")
    y_data = raw["phase"]
    params = raw["info"]
    frequencies = params[0,0]["frequencies_Hz"].squeeze()

    dx, dy, dz = params[0,0]["dx_m"], params[0,0]["dy_m"], params[0,0]["dz_m"]
    y_spacing = np.array([1,dx[0,0],dy[0,0],dz[0,0],1]).squeeze()
    y_data = np.moveaxis(y_data, -1, 0) # Freq axis leading
    y_data = np.moveaxis(y_data, 2, 1) # X-axis leading y and z
    y_data = y_data[..., [1,0,2]] # Rearrange feature vector
    
    # General dataset parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tload = time.time() - tstart

    # Preprocessing
    y_filtered = y_data
   
    # Differentiate
    d_dims = (1,2,3) # Dimensions to differentiate along

    ydx = hsi.diff.fd(y_filtered, y_spacing, axes = d_dims)
    yddx = hsi.diff.fd(ydx, y_spacing, axes = d_dims)
    dx_split = [ydx[..., i] for i in range(ydx.shape[-1])] # Split up each derivative component

    curl = hsi.diff.diff_to_curl(ydx)
    div = hsi.diff.diff_to_div(ydx)
    strain_tensor = hsi.diff.diff_to_st(ydx)
    
    lapl = hsi.diff.diff_to_lapl(yddx)
    gdiv = hsi.diff.fd(div, y_spacing, axes = d_dims)

    div = np.ones_like(y_data, dtype = div.dtype) * div[..., None] # Divergence is scalar, expand to all dims
    bias = np.ones_like(y_data, dtype = y_data.dtype)

    # Library
    #library = [*dx_split, curl, div, lapl, gdiv, bias] # Static
    library = [*dx_split, curl, div, bias] # Dynamic

    # Target
    freq_dim = [0]
    f_reshape = [frequencies.shape[0] if i in freq_dim else 1 for i in range(y_data.ndim)]
    target = -((frequencies * 2 * np.pi) ** 2).reshape(*f_reshape) * y_filtered

    # Dynamics
    tstd = np.abs(target).std()
    strain_tensor = torch.tensor(strain_tensor, device=device, dtype = torch.complex64)
    strain_tensor = hsi.utils.reshape_data(strain_tensor, reshape_dims = (1,2,3))
    t_lapl = torch.tensor(lapl, dtype = torch.complex64, device = device)
    t_lapl = hsi.utils.reshape_data(t_lapl, reshape_dims = (1,2,3))
    t_div = torch.tensor(div, device = device, dtype = torch.complex64)
    t_div = hsi.utils.reshape_data(t_div, reshape_dims=(1,2,3))
    t_gdiv = torch.tensor(gdiv, device = device, dtype = torch.complex64)
    t_gdiv = hsi.utils.reshape_data(t_gdiv, reshape_dims=(1,2,3))

    dyn_eps = 1e-3
    dynamic = [hsi.model.HLaplBasis(strain_tensor / tstd, t_lapl / tstd, eps = dyn_eps), hsi.model.HGDivBasis(t_gdiv / tstd, t_div / tstd, eps = dyn_eps)] # Dynamic
    #dynamic = [] # Static

    tpreproc = time.time() - tstart

    # Predictor
    rff_scale = 150
    n_width = 512
    pred = hsi.networks.RFFMLP([3, n_width, n_width, n_width, n_width, n_width, n_width, len(library) + len(dynamic)],
                              activation = torch.nn.GELU(),
                              dtype = torch.complex64,
                              scale = rff_scale).to(device)
    
    opt = torch.optim.Adam(pred.parameters(), lr = 1e-3)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma = 0.98)
    #sch = None

    l1_reg = 1e-1
    reg = hsi.loss.mL1Reg(l1_reg)
    
    # Regression
    params = {"epochs": 10, #300, 
              "batch_size": 2048, 
              "regularization": reg,
              "l1_strength": l1_reg,
              "predictor": pred,
              "cg": 100,
              "loss_fn": hsi.loss.CMSE(),
              "optimizer": opt,
              "scheduler": sch,
              "scheduler_gamma": sch.gamma,
              "scale": rff_scale}
    
    params["spec"] = ["dx", "dy", "dz", "curl", "div", "lapl", "gdiv", "bias"] # Static
    #params["spec"] = ["dx", "dy", "dz", "curl", "div", "bias", "hlapl", "hgdiv"] # Dynamic
    
    x_data = hsi.utils.gen_coord_grid(y_data, y_spacing)

    # Strip fixed x-dim features
    xdel = [i for i in range(x_data.shape[-1]) if i in params.get("fixed_dims", (0,4)) ]
    x_data = np.delete(x_data, xdel, axis = -1)
    
    model = hsi.model.HetSI(y_data=target, 
                            x_data=x_data,
                            b_data=library,
                            fixed_dims=params.get("fixed_dims", (0,4)),
                            dtype=params.get("dtype", torch.complex64),
                            self_grad=params.get("self_grad", True),
                            dynamic=dynamic
                            )

    predictor = params.get("predictor")
    regularization = params.get("regularization", hsi.loss.mL1Reg(params.get("l1", 1e-1)))
    scheduler = params.get("scheduler", None)
    clip_grad = params.get("cg", None)
    loss_fn = params.get("loss_fn", hsi.loss.CMSE())
    preproc = params.get("preproc", None)
    optimizer = params.get("optimizer", torch.optim.Adam(predictor.parameters(), lr=params.get("learning_rate", 1e-3)))
    batch_size = params.get("batch_size", 2048)

    model.setup(predictor,
                optimizer,
                loss_fn,
                regularization,
                scheduler,
                preproc,
                clip_grad,
                batch_size = batch_size,
                )
    
    tsetup = time.time() - tstart

    results = model.regress(epochs=params.get("epochs", 1000))

    tregress = time.time() - tstart

    # Generate results
    _, xs, ys, pred, bases = model.full_pred(use_batched_limit=True, batch_size = 1024)
    
    # Save results

    params["times"] = [tstart, tload, tpreproc, tsetup, tregress]

    hsi.utils.save_results(tpath, model.state_dict(), optimizer.state_dict(), pred, bases, ys, xs, results, params)

if __name__ == "__main__":
    main()
    pass
    