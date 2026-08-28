## Main module

import torch
import torch.nn as nn
import torch.utils.data as tud
import numpy as np
from tqdm.auto import tqdm
import hetsi.utils as hsu

# Global data indexing

class idxDset(tud.Dataset):
    """Dataset of indices, for synchronizing sampling from bases and data."""

    def __init__(self, l_data):
        self.idxs = torch.arange(0, l_data, dtype = torch.int64)
        pass
    
    def __len__(self):
        return self.idxs.shape[0]
    
    def __getitem__(self, idx):
        return self.idxs[idx]
    
    def __getitems__(self, idxs):
        return [self.idxs[idx] for idx in idxs]

    @staticmethod
    def collate(b):
        return (*b,)

class stdData(tud.Dataset):
    """Dataset for input or output data."""

    def __init__(self, basis: torch.Tensor, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.data = basis.to(device)
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, ...]

    def __getitems__(self, idxs):
        return self.data[idxs, ...]

# Dataset for single basis

class stdBasis(tud.Dataset):
    """Dataset representing a single static basis."""

    def __init__(self, basis: torch.tensor, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.b = basis.to(device)
    
    def __len__(self):
        return self.b.shape[0]

    def __getitem__(self, idx):
        return self.b[idx, ...]

    def __getitems__(self, idxs):
        return self.b[idxs, ...]
    
    def __call__(self, idx, *args):
        b = self.__getitems__(idx)
        return b
    
# Dynamic updating basis

class HGDivBasis(tud.Dataset):
    """Spatially heterogeneous longitudinal wave basis."""

    def __init__(self, gdiv: torch.Tensor, div: torch.Tensor, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), eps = 1e-6):
        """
        
        :param gdiv: Tensor with gradient of divergence.
        :type gdiv: torch.Tensor
        :param div: Tensor with divergence.
        :type div: torch.Tensor
        :param device: Target device to use.
        :param eps: Division correction factor.
        """
        self.b1 = gdiv[..., None].to(device)
        self.b2 = div[..., None].to(device)
        self.b = 1. # Placeholder
        self.eps = eps
    
    def __len__(self):
        return self.b1.shape[0]
    
    def __getitem__(self, idx):
        return self.b1[idx, ...]

    def __getitems__(self, idx):
        return self.b1[idx, ...]

    def __call__(self, idx, x, p, i):
        dg = hsu.autograd_grad(p[..., i:i+1], x)
        gdiv = self.b1[idx, ...]
        div = self.b2[idx, ...]

        return gdiv + div * dg.transpose(-1, -2) * hsu.complex_inverse(p[..., i:i+1], self.eps)


class HLaplBasis(tud.Dataset):
    """Spatially heterogeneous shear wave basis."""
    def __init__(self, stens: torch.Tensor, lapl: torch.Tensor, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), eps = 1e-6):
        """
        
        :param stens: Strain tensor.
        :type stens: torch.Tensor
        :param lapl: Tensor with Laplacian.
        :type lapl: torch.Tensor
        :param device: Target device to use.
        :param eps: Division correction factor.
        """
        self.b1 = stens.to(device)
        self.b2 = lapl[..., None].to(device)
        self.b = 1. # Placeholder
        self.eps = eps
    
    def __len__(self):
        return self.b2.shape[0]
    
    def __getitem__(self, idx):
        return self.b2[idx, ...]

    def __getitems__(self, idx):
        return self.b2[idx, ...]

    def __call__(self, idx, x, p, i):
        dg = hsu.autograd_grad(p[..., i:i+1], x)
        lapl = self.b2[idx, ...]
        stens = self.b1[idx, ...]

        return lapl + torch.sum(hsu.complex_inverse(p[..., i:i+1], self.eps) * dg * 2 * stens, dim = -1, keepdim=True)

# Main class

class HetSI(nn.Module):

    """Sparse regression module implementing the HetSI algorithm."""

    def __init__(self, y_data: np.ndarray, x_data: np.ndarray, b_data: list[np.ndarray], scale_data = True, fixed_dims = (0,), dynamic = None, **kwargs):
        """
        :param y_data: Array of target quantity. Should have N dimensions corresponding to each axis of variation.
        :type y_data: numpy.ndarray
        :param x_data: Array of predictor inputs (typically coordinates or indices). Should have N+1 dimensions.
        :type x_data: numpy.ndarray
        :param b_data: List of arrays of standard basis functions to use. Each entry should have N dimensions.
        :type b_data: list[numpy.ndarray]
        :param scale_data: Whether target and basis data should be normalized before regression.
        :type scale_data: bool
        :param fixed_dims: Dimensions which are considered fixed for the problem.
        :type fixed_dims: tuple(int)
        :param dynamic: List with preprocessed basis classes.
        :type dynamic: list[basis] or None
        
        :param kwargs: dtype: Torch datatype to use for basis and target data. Default: torch.float32. | x_dtype: Torch datatype of the input data. Default: torch.float32. | self_grad: bool, if predictor outputs should have _requires_grad. Necessary for derivatives of predicted variables. Default: False"""

        super(HetSI, self).__init__()

        self.fd = fixed_dims
        self.dtype = kwargs.get("dtype", torch.float32)
        self.self_grad = kwargs.get("self_grad", False)

        xd, yd = torch.tensor(x_data, dtype = kwargs.get("x_dtype", torch.float32)), torch.tensor(y_data, dtype = self.dtype)[..., None] # Add trailing for consistency
        bd = torch.tensor(np.stack(b_data, axis = -1), dtype = self.dtype)

        fd = sorted(fixed_dims, key = lambda x: x)
        xd = self.strip_x_dims(xd, fd) # Remove superfluous x-dims
        
        # Rearrange fixed dims
        self.dshape = y_data.shape # Store orignal shape
        self.pshape = tuple(y_data.shape[i] if i not in fixed_dims else 1 for i in range(y_data.ndim))

        xd = self.rearrange_data(xd, fd, ignore_lagging=True)
        yd = self.rearrange_data(yd, fd, ignore_lagging=True)
        bd = self.rearrange_data(bd, fd, ignore_lagging=True)

        assert xd.shape[0] == yd.shape[0]
        assert bd.shape[0] == xd.shape[0]

        # Storage
        self.x = stdData(xd)
        self.y = stdData(yd)
        self.b = [stdBasis(b) for b in bd.split(1, dim = -1)]
        self.idx = idxDset(len(self.x))

        # Dynamic components
        if dynamic is not None:
            self.b.extend(dynamic) # Add dynamic components

        # Normalize
        if scale_data:
            self.yscale = yd.abs().std()
            self.y.data /= self.yscale
            
            for b in self.b:
                b.b /= self.yscale

        # Training storage
        self.batch_losses = []
        self.batch_reg = []
        self.batch_mse = []

    def setup(self, predictor, optimizer, loss_fn, regularization = None, scheduler = None, preproc = None, clip_grad = None, **kwargs):
        """
        Sets up necessary components for regression.
        
        :param predictor: Torch module performing prediction.
        :param optimizer: Torch optimizer, set to optimize predictor and any learnable preprocessors.
        :param loss_fn: Callable taking predicted and reference field and calculating a loss.
        :param regularization: Callable taking in parameter field and calculating a regularization.
        :param scheduler: Torch scheduler for the optimizer.
        :param preproc: Torch module implementing a preprocessing scheme.
        :param clip_grad: Gradient clipping limit, utilizing clip_grad_value.
        kwargs:
        :param kwargs: batch_size -- Number of points to train with per step. Default: 1024. | shuffle -- Bool if batches should be shuffled. Default: True
        """

        self.pred = predictor
        self.opt = optimizer
        self.lfn = loss_fn
        self.reg = regularization
        self.sched = scheduler
        self.preproc = preproc
        self.clip_grad = clip_grad

        self.dlidx = tud.DataLoader(self.idx, 
                                    batch_size = kwargs.get("batch_size", 1024),
                                    shuffle = kwargs.get("shuffle", True))

        self.rdy = True
    
    def postpend_basis(self, basis):
        if isinstance(basis, list):
            self.b.extend(basis)
        else:
            self.b.append(basis)

    def regress(self, epochs = 1e3):
        """Run HetSI.
        
        :param epochs: Number of epochs to complete.
        :returns: Lists of training loss, regularization loss and data loss.
        :rtype: tuple(list, list, list)"""

        assert self.rdy, "Complete model setup by running the setup method."

        self.train()
        
        tqdm_progress = tqdm(range(epochs), unit = "iter")

        for epoch in tqdm_progress:
            for bidx in self.dlidx:
                
                self.opt.zero_grad()

                x = self.x[bidx]
                y = self.y[bidx]

                if self.self_grad:
                    x = x.detach().requires_grad_()

                p = self.pred(x)

                if self.preproc is not None:
                    p = self.preproc(p) # Pre-calculation correction, e.g. thresholding, absolute values, etc.

                b = torch.cat([bx(bidx, x, p, i) for (i, bx) in enumerate(self.b)], dim = -1)
                bp = b * p[..., :b.shape[-1]] # Conditional for pretraining
                
                res = torch.sum(bp, dim = -1, keepdim=True) # Target prediction
                loss = self.lfn(res, y)
                self.batch_mse.append(loss.item())

                if self.reg is not None:
                    bc = self.feature_normalize_l1(b) # Scale correction per feature
                    reg = self.reg(bc * p[..., :bc.shape[-1]]) # Scale corrected regularization
                    self.batch_reg.append(reg.item())
                    loss += reg

                self.batch_losses.append(loss.item())
                tqdm_progress.set_description_str(f"Err: {loss.item():{2}.{4}}, Reg: {reg.item():{2}.{4}}")
                loss.backward()

                if self.clip_grad is not None:
                    nn.utils.clip_grad_value_(self.pred.parameters(), self.clip_grad)

                self.opt.step()

            if self.sched is not None:
                self.sched.step()
        
        return self.batch_losses, self.batch_reg, self.batch_mse

    def full_pred(self, use_batched_limit = False, batch_size = 2048):
        """
        Generate predictions for all points, and bases. All non-fixed dims are flattened, reshape back to original shapes to compare with ordinary data.
        
        :param use_batched_limit: If regressor should split the calculation into multiple parts. Useful if dataset is large.
        :param batch_size: Number of points per batched split.
        :returns: Returns a tuple with tensors of: indices, input points, target points, predictions at indices, bases at indicies.
        """

        self.eval()
        
        if use_batched_limit:
            idxs = self.idx[:]

            lidx = torch.split(idxs, batch_size, dim = 0)

            pc = []
            bc = []

            for i, idx in enumerate(lidx):

                xc = self.x[idx]
                yc = self.y[idx]

                if self.self_grad:
                    xc.requires_grad_()

                p = self.pred(xc)

                if self.preproc is not None:
                    p = self.preproc(p)

                bc.append(torch.cat([base(idx, xc, p, i).detach().cpu() for i, base in enumerate(self.b)], dim=-1))
                
                pc.append(p.detach().cpu())
                xc = xc.detach()

                del xc, yc # Clean up for limiting memory use
                
            clidx = torch.cat(lidx, dim = 0) # Ensure proper ordering

            p = torch.cat(pc, dim = 0).cpu()
            x = self.x[clidx].cpu()
            y = self.y[clidx].cpu()
            b = torch.cat(bc, dim = 0).cpu()


        else:
            idxs = self.idx[:]
            y = self.y[idxs]
            x = self.x[idxs]

            if self.self_grad:
                x.requires_grad_()

            p = self.pred(x)
            b = torch.cat([base(idxs, x, p, i) for i, base in enumerate(self.b)], dim=-1)

        return idxs, x, y, p, b

    @staticmethod
    def feature_normalize(x, eps = 1e-7):
        return torch.sqrt(torch.sum((x * x.conj()) + eps, dim = tuple(range(1, x.ndim-1)), keepdim=True))
    
    @staticmethod
    def feature_normalize_l1(x, eps = 1e-7):
        return torch.sum(torch.abs(x), dim = tuple(range(1, x.ndim-1)), keepdim=True)

    @staticmethod
    def strip_x_dims(x, fd):
        xsel = tuple(slice(0,1) if i in fd else slice(None) for i in range(x.ndim-1))
        return x[xsel]

    @staticmethod
    def rearrange_data(x:torch.Tensor, fd: tuple[int], ignore_lagging = False):
        """Rearranges data to dynamic dimensions followed by fixed dimensions and flattens the dynamic ones. Such that 
        output.shape = (prod(J), *I) where J = [dim is not a fixed_dim], I = [dim is a fixed_dim]
        
        Example: Fixed dims: (1,2), Input: Tensor.shape = (a, b, c, d, e) -> Output.shape = (a x d, b, c, e).
        Example 2: Fixed dims: (0,), Input: Tensor.shape = (a, b, c, d) -> Output.shape = (b x c, a, d)."""

        if ignore_lagging:
            dims = list(range(x.ndim-1))
            
        else:
            dims = list(range(x.ndim))

        fix = tuple(i for i in dims if i in fd)

        if ignore_lagging:
            fix += (x.ndim - 1,)

        move = tuple(i for i in dims if i not in fd)
        x = x.permute(move + fix)

        return x.flatten(0, len(move)-1)
    
    
class freq_kelvin(torch.nn.Module):
    """
    Kelvin-Voigt style preprocessing module, modifying the predicted variables along the frequency axis according to fixed relationship.
    """

    def __init__(self, freqs = [30,50,70]):
        super().__init__()
        self.register_buffer("omega", torch.pi * 2 * torch.tensor(freqs, dtype = torch.complex64))
        
    def forward(self, p):
        p = p * torch.ones((1, self.omega.shape[0],1,p.shape[-1]), device=p.device, dtype = p.dtype)

        c11 = p[..., -2:-1].real.to(p.dtype)
        c12 = p[..., -2:-1].imag.to(p.dtype)

        c21 = p[..., -1:].real.to(p.dtype)
        c22 = p[..., -1:].imag.to(p.dtype)

        c1 = c11 + 1j * c12 * self.omega[None, :, None, None]
        c2 = c21 + 1j * c22 * self.omega[None, :, None, None]

        return torch.cat([p[..., :-2], c1, c2], dim = -1)

if __name__ == "__main__":
    pass