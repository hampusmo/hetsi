import os, sys

sys.path.append("..") # Add top level dir

import hetsi as hsi
import numpy as np
import torch
import time

def main():
    tstart = time.time()

    fpath = "..." # Input data path
    tpath = "../outputs/boxfem/" # Target data path (outputs)
    
    y_data = hsi.data.loadBioqicFEM(fpath)
    
    # General dataset parameters
    y_spacing = np.array([1 ,0.001, 0.001, 0.001, 1])
    frequencies = np.array([50, 60, 70, 80, 90, 100])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tload = time.time() - tstart

    # Preprocessing
    y_filtered = y_data # Already in Frequency domain

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

    div = np.ones_like(y_filtered, dtype = div.dtype) * div[..., None] # Divergence is scalar, expand to all dims
    bias = np.ones_like(y_filtered, dtype = y_filtered.dtype)

    # Static library components
    library = [*dx_split, curl, div, lapl, gdiv, bias] # Static
    #library = [*dx_split, curl, div, bias] # Lapl and gdiv in dynamic basis


    # Target
    freq_dim = [0]
    f_reshape = [frequencies.shape[0] if i in freq_dim else 1 for i in range(y_filtered.ndim)]
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

    dynamic = [hsi.model.HLaplBasis(strain_tensor / tstd, t_lapl / tstd), hsi.model.HGDivBasis(t_gdiv / tstd, t_div / tstd)] # Dynamic
    #dynamic = [] # Static

    # Predictor
    rff_scale = 200 # Roughly 5 voxels @ 1 mm -> approx 200
    pred = hsi.networks.RFFMLP([3, 512, 512, 512, 512, len(library) + len(dynamic)],
                              activation = torch.nn.GELU(),
                              dtype = torch.complex64,
                              scale = rff_scale).to(device)
    
    opt = torch.optim.Adam(pred.parameters(), lr = 1e-3)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma = 0.99)
    l1_reg = 1e-1
    reg = hsi.loss.mL1Reg(l1_reg)
    
    tpreproc = time.time() - tstart

    # Regression
    params = {"epochs": 500,
              "batch_size": 2048, 
              "regularization": reg,
              "l1": l1_reg,
              "predictor": pred,
              "cg": 100,
              "loss_fn": hsi.loss.CMSE(),
              "optimizer": opt,
              "scheduler": sch,
              "scheduler_gamma": sch.gamma,
              "scale": rff_scale}
    
    params["spec"] = ["dx", "dy", "dz", "curl", "div", "lapl", "gdiv", "bias", "hlapl", "hgdiv"]
    
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

    # Fetch parameters / Defaults
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
    _, xs, ys, pred, bases = model.full_pred()
    
    # Save results
    params["times"] = [tstart, tload, tpreproc, tsetup, tregress]

    hsi.utils.save_results(tpath, model.state_dict(), optimizer.state_dict(), pred, bases, ys, xs, results, params)

if __name__ == "__main__":
    main()
    pass
    