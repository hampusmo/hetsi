import os, sys

sys.path.append("..") # Add top level dir

import hetsi as hsi
import numpy as np
import torch
import os

def main():

    fpath = "../notebooks/data/" # Input data path
    tpath = "../outputs/boxfem_noisy/" # Target data path (outputs)
    full_data = np.load(fpath + "bfem_noisy.npy")

    # General dataset parameters
    y_spacing = np.array([1 ,0.001, 0.001, 0.001, 1])
    frequencies = np.array([50, 60, 70, 80, 90, 100])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocessing
    
    pfull = []
    bfull = []
    yfull = []
    snrfull = [None,60,40,20,0]

    for i in range(full_data.shape[0]):
        y_data = full_data[i, ...]
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
        strain_tensor = torch.tensor(strain_tensor, device=device, dtype = torch.complex64)
        strain_tensor = hsi.utils.reshape_data(strain_tensor, reshape_dims = (1,2,3))
        t_lapl = torch.tensor(lapl, dtype = torch.complex64, device = device)
        t_lapl = hsi.utils.reshape_data(t_lapl, reshape_dims = (1,2,3))
        t_div = torch.tensor(div, device = device, dtype = torch.complex64)
        t_div = hsi.utils.reshape_data(t_div, reshape_dims=(1,2,3))
        t_gdiv = torch.tensor(gdiv, device = device, dtype = torch.complex64)
        t_gdiv = hsi.utils.reshape_data(t_gdiv, reshape_dims=(1,2,3))

        #dynamic = [] # Static case
        dynamic = [hsi.model.HLaplBasis(strain_tensor / target.std(), t_lapl / target.std()), hsi.model.HGDivBasis(t_gdiv / target.std(), t_div / target.std())] # Dynamic case
      

        # Predictor
        rff_scale = 200
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
                "spec": ["dx", "dy", "dz", "curl", "div", "lapl", "gdiv" "bias"],
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
        
        results = model.regress(epochs=params.get("epochs", 1000))

        # Generate results
        _, xs, ys, pred, bases = model.full_pred()

        yfull.append(ys.cpu())
        pfull.append(pred.detach().cpu())
        #bfull.append(bases)
   

    # Stack em
    yfull = torch.stack(yfull, dim = -1)
    pfull = torch.stack(pfull, dim = -1)
    #bfull = torch.stack(bfull, dim = -1)

    hsi.utils.save_results(tpath, None, None, pfull, None, yfull, None, None, params)

if __name__ == "__main__":
    main()
    pass
    