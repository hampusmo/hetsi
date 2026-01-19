import os, sys

sys.path.append("..") # Add top level dir

import hetsi as hsi
import numpy as np
import torch
import scipy.ndimage as spi
import json
import time

def main():
    times = []
    t_start = time.time()
    times.append(t_start)

    fpath = "..." # Input data path
    tpath = "../outputs/boxmre/" # Target data path (outputs)
    y_data = hsi.data.loadBioqic(fpath)

    t_load = time.time() - t_start
    print("Loading time: ", t_load)
    times.append(t_load)

    # General dataset parameters
    y_spacing = np.array([1,0.0015, 0.0015, 0.0015, 1])
    frequencies = np.array([30, 40, 50, 60, 70, 80, 90, 100])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocessing
    f_sigma = 0.8 # Standard deviation of Gaussian filter
    f_dims = (1,2,3) # Dimensions to filter along

    y_data = y_data[:,5:-5, 5:-5, ...] # Crop edges

    y_data = np.fft.rfft(y_data, axis = -2)[..., 1, :] # Extract dominant frequency
    y_filtered = spi.gaussian_filter(y_data, f_sigma, axes = f_dims)
    
    # Differentiate
    d_dims = (1,2,3) # Dimensions to differentiate along

    ydx = hsi.diff.fd(y_filtered, y_spacing, axes = d_dims)
    yddx = hsi.diff.fd(ydx, y_spacing, axes = d_dims)

    curl = hsi.diff.diff_to_curl(ydx)
    div = hsi.diff.diff_to_div(ydx)
    strain_tensor = hsi.diff.diff_to_st(ydx)
    
    lapl = hsi.diff.diff_to_lapl(yddx)
    gdiv = hsi.diff.fd(div, y_spacing, axes = d_dims)

    div = np.ones_like(y_data, dtype = div.dtype) * div[..., None] # Divergence is scalar, expand to all dims
    bias = np.ones_like(y_data, dtype = y_data.dtype)

    # Library
    dx_split = [ydx[..., i] for i in range(ydx.shape[-1])] # Split up each derivative component

    #library = [*dx_split, curl, div, lapl, gdiv, bias] # Static
    library = [*dx_split, curl, div, bias] # Dynamic

    # Target
    freq_dim = [0]
    f_reshape = [frequencies.shape[0] if i in freq_dim else 1 for i in range(y_data.ndim)]
    target = -((frequencies * 2 * np.pi) ** 2).reshape(*f_reshape) * y_filtered

    # Dynamic components
    tstd = np.abs(target).std()
    strain_tensor = torch.tensor(strain_tensor, device=device, dtype = torch.complex64)
    strain_tensor = hsi.utils.reshape_data(strain_tensor, reshape_dims = (1,2,3))
    t_lapl = torch.tensor(lapl, dtype = torch.complex64, device = device)
    t_lapl = hsi.utils.reshape_data(t_lapl, reshape_dims = (1,2,3))
    t_div = torch.tensor(div, device = device, dtype = torch.complex64)
    t_div = hsi.utils.reshape_data(t_div, reshape_dims=(1,2,3))
    t_gdiv = torch.tensor(gdiv, device = device, dtype = torch.complex64)
    t_gdiv = hsi.utils.reshape_data(t_gdiv, reshape_dims=(1,2,3))

    dynamic = [hsi.model.HLaplBasis(strain_tensor / tstd, t_lapl / tstd), hsi.model.HGDivBasis(t_gdiv / tstd, t_div / tstd)]
    #dynamic = []

    t_pre = time.time() - t_start
    times.append(t_pre)
    print("Preprocessing time: ", t_pre)

    # Predictor
    rff_scale = 100. # Approx. 6 voxel features @ 1.5mm -> 100
    pred = hsi.networks.RFFMLP([3, 512, 512, 512, 512, len(library) + len(dynamic)],
                              activation = torch.nn.GELU(),
                              dtype = torch.complex64,
                              scale = rff_scale).to(device)
    
    opt = torch.optim.Adam(pred.parameters(), lr = 1e-3)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma = 0.99)
    #sch = None
    
    # Regression
    params = {"epochs": 500, 
              "batch_size": 2048, 
              "regularization": hsi.loss.mL1Reg(1e-1),
              "predictor": pred,
              "cg": 100,
              "loss_fn": hsi.loss.CMSE(),
              "optimizer": opt,
              "scheduler": sch,
              "scheduler_gamma": sch.gamma,
              "fixed_dims": (0,4),
              "spec": ["dx", "dy", "dz", "curl", "div", "bias", "hlapl", "hgdiv"], 
              #"spec": ["dx", "dy", "dz", "curl", "div", "lapl", "gdiv", "bias"],
              "scale": rff_scale}
    
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

    t_setup = time.time() - t_start
    times.append(t_setup)
    print("Setup time: ", t_setup)

    results = model.regress(epochs=params.get("epochs", 1000))

    t_regress = time.time() - t_start
    times.append(t_regress)
    #times.append(sum(times))
    #print("Training time: ", t_regress - t_setup)
    #print("Total time: ", t_regress - t_start)

    params["times"] = times

    _, xs, ys, pred, bases = model.full_pred(use_batched_limit = True, )
    
    # Save results
    if not tpath.endswith("/"):
        tpath += "/"

    if not os.path.isdir(tpath):
        os.mkdir(tpath)

    torch.save(model.state_dict(), tpath + "model.pth")
    torch.save(optimizer.state_dict(), tpath + "opt.pth")
    torch.save(pred.detach(), tpath + "pred.pth")
    torch.save(bases, tpath + "base.pth")
    torch.save(ys, tpath + "target.pth")
    torch.save(xs, tpath + "input.pth")
    
    with open(tpath + "losses.json", "w") as file:
        json.dump(results, file)

    # Parmeters to JSON
    ser_params = {}
    
    for key, value in params.items():
        if hasattr(value, "to_json"):
            ser_params[key] = value
        else:
            ser_params[key] = str(value)

    with open(tpath + "config.json", "w") as file:
        strrep = json.dumps(ser_params, separators=(",", ":"))
        file.write(strrep)

    with open(tpath + "times.json", "w") as file:
        json.dump(times, file)

    t_save = time.time()

if __name__ == "__main__":
    main()
    pass
    