import pickle
from math import floor, ceil
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .homography_layers import CuboidLayerGlobal

def make1DGaussian(size, fwhm=3, center=None):
    """ Make a 1D gaussian kernel.

    size is the length of the kernel,
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(0, size, 1, dtype=float)

    if center is None:
        center = size // 2

    return np.exp(-4*np.log(2) * (x-center)**2 / fwhm**2)


def make2DGaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

class RecasensSaliencyToGridMixin(object):
    """Grid generator based on 'Learning to Zoom: a Saliency-Based Sampling \
    Layer for Neural Networks' [https://arxiv.org/pdf/1809.03355.pdf]."""

    def __init__(self, output_shape, grid_shape=(31, 51), separable=True,
                 attraction_fwhm=13, anti_crop=True, **kwargs):
        super(RecasensSaliencyToGridMixin, self).__init__()
        self.output_shape = output_shape
        self.output_height, self.output_width = output_shape
        self.grid_shape = grid_shape
        self.padding_size = min(self.grid_shape)-1
        self.total_shape = tuple(
            dim+2*self.padding_size
            for dim in self.grid_shape
        )
        self.padding_mode = 'reflect' if anti_crop else 'replicate'
        self.separable = separable

        if self.separable:
            self.filter = make1DGaussian(
                2*self.padding_size+1, fwhm=attraction_fwhm)
            self.filter = torch.FloatTensor(self.filter).unsqueeze(0) \
                                                        .unsqueeze(0).cuda()

            self.P_basis_x = torch.zeros(self.total_shape[1])
            for i in range(self.total_shape[1]):
                self.P_basis_x[i] = \
                    (i-self.padding_size)/(self.grid_shape[1]-1.0)
            self.P_basis_y = torch.zeros(self.total_shape[0])
            for i in range(self.total_shape[0]):
                self.P_basis_y[i] = \
                    (i-self.padding_size)/(self.grid_shape[0]-1.0)
        else:
            self.filter = make2DGaussian(
                2*self.padding_size+1, fwhm=attraction_fwhm)
            self.filter = torch.FloatTensor(self.filter) \
                               .unsqueeze(0).unsqueeze(0).cuda()

            self.P_basis = torch.zeros(2, *self.total_shape)
            for k in range(2):
                for i in range(self.total_shape[0]):
                    for j in range(self.total_shape[1]):
                        self.P_basis[k, i, j] = k*(i-self.padding_size)/(self.grid_shape[0]-1.0)+(1.0-k)*(j-self.padding_size)/(self.grid_shape[1]-1.0)  # noqa: E501

    def separable_saliency_to_grid(self, imgs, x_saliency,
                                   y_saliency, device):
        assert self.separable
        x_saliency = F.pad(x_saliency, (self.padding_size, self.padding_size),
                           mode=self.padding_mode)
        y_saliency = F.pad(y_saliency, (self.padding_size, self.padding_size),
                           mode=self.padding_mode)

        N = imgs.shape[0]
        P_x = torch.zeros(1, 1, self.total_shape[1], device=device)
        P_x[0, 0, :] = self.P_basis_x
        P_x = P_x.expand(N, 1, self.total_shape[1])
        P_y = torch.zeros(1, 1, self.total_shape[0], device=device)
        P_y[0, 0, :] = self.P_basis_y
        P_y = P_y.expand(N, 1, self.total_shape[0])

        weights = F.conv1d(x_saliency, self.filter)
        weighted_offsets = torch.mul(P_x, x_saliency)
        weighted_offsets = F.conv1d(weighted_offsets, self.filter)
        xgrid = weighted_offsets/weights
        xgrid = torch.clamp(xgrid*2-1, min=-1, max=1)
        xgrid = xgrid.view(-1, 1, 1, self.grid_shape[1])
        xgrid = xgrid.expand(-1, 1, *self.grid_shape)

        weights = F.conv1d(y_saliency, self.filter)
        weighted_offsets = F.conv1d(torch.mul(P_y, y_saliency), self.filter)
        ygrid = weighted_offsets/weights
        ygrid = torch.clamp(ygrid*2-1, min=-1, max=1)
        ygrid = ygrid.view(-1, 1, self.grid_shape[0], 1)
        ygrid = ygrid.expand(-1, 1, *self.grid_shape)

        grid = torch.cat((xgrid, ygrid), 1)
        grid = F.interpolate(grid, size=self.output_shape, mode='bilinear',
                             align_corners=True)
        return grid.permute(0, 2, 3, 1)

    def nonseparable_saliency_to_grid(self, imgs, saliency, device):
        assert not self.separable
        p = self.padding_size
        saliency = F.pad(saliency, (p, p, p, p), mode=self.padding_mode)

        N = imgs.shape[0]
        P = torch.zeros(1, 2, *self.total_shape, device=device)
        P[0, :, :, :] = self.P_basis
        P = P.expand(N, 2, *self.total_shape)

        saliency_cat = torch.cat((saliency, saliency), 1)
        weights = F.conv2d(saliency, self.filter)
        weighted_offsets = torch.mul(P, saliency_cat) \
                                .view(-1, 1, *self.total_shape)
        weighted_offsets = F.conv2d(weighted_offsets, self.filter) \
                            .view(-1, 2, *self.grid_shape)

        weighted_offsets_x = weighted_offsets[:, 0, :, :] \
            .contiguous().view(-1, 1, *self.grid_shape)
        xgrid = weighted_offsets_x/weights
        xgrid = torch.clamp(xgrid*2-1, min=-1, max=1)
        xgrid = xgrid.view(-1, 1, *self.grid_shape)

        weighted_offsets_y = weighted_offsets[:, 1, :, :] \
            .contiguous().view(-1, 1, *self.grid_shape)
        ygrid = weighted_offsets_y/weights
        ygrid = torch.clamp(ygrid*2-1, min=-1, max=1)
        ygrid = ygrid.view(-1, 1, *self.grid_shape)

        grid = torch.cat((xgrid, ygrid), 1)
        grid = F.interpolate(grid, size=self.output_shape, mode='bilinear',
                             align_corners=True)
        return grid.permute(0, 2, 3, 1)

class PlainKDEGrid(nn.Module, RecasensSaliencyToGridMixin):
    """Image adaptive grid generator with fixed hyperparameters -- KDE SI"""

    def __init__(
        self,
        bandwidth_scale=1,
        amplitude_scale=1,
        **kwargs
    ):
        super(PlainKDEGrid, self).__init__()
        RecasensSaliencyToGridMixin.__init__(self, **kwargs)
        self.bandwidth_scale = bandwidth_scale
        self.amplitude_scale = amplitude_scale
        self.input_shape = kwargs.get('input_shape', (1080, 1920))

    def bbox2sal(self, batch_bboxes, imgs, jitter=None):
        device = batch_bboxes[0].device
        h_out, w_out = self.grid_shape
        sals = []
        B = imgs.shape[0]
        for i in range(B):
            h, w = self.input_shape
            bboxes = batch_bboxes[i]
            if len(bboxes) == 0:  # zero detections case
                sal = torch.ones(h_out, w_out, device=device).unsqueeze(0)
                sal /= sal.sum()
                sals.append(sal)
                continue
            bboxes[:, 2:] -= bboxes[:, :2]  # ltrb -> ltwh
            cxy = bboxes[:, :2] + 0.5*bboxes[:, 2:]
            if jitter is not None:
                cxy += 2*jitter*(torch.randn(cxy.shape, device=device)-0.5)

            widths = (bboxes[:, 2] * self.bandwidth_scale).unsqueeze(1)
            heights = (bboxes[:, 3] * self.bandwidth_scale).unsqueeze(1)

            X, Y = torch.meshgrid(
                torch.linspace(0, w, w_out, dtype=torch.float, device=device),
                torch.linspace(0, h, h_out, dtype=torch.float, device=device),
            )
            grids = torch.stack((X.flatten(), Y.flatten()), dim=1).t()

            m, n = cxy.shape[0], grids.shape[1]

            norm1 = (cxy[:, 0:1]**2/widths + cxy[:, 1:2]**2/heights) \
                .expand(m, n)
            norm2 = grids[0:1, :]**2/widths + grids[1:2, :]**2/heights
            norms = norm1 + norm2

            cxy_norm = cxy
            cxy_norm[:, 0:1] /= widths
            cxy_norm[:, 1:2] /= heights

            distances = norms - 2*cxy_norm.mm(grids)

            sal = (-0.5 * distances).exp()
            sal = self.amplitude_scale * (sal / (0.00001+sal.sum(dim=1, keepdim=True)))  # noqa: E501, normalize each distribution
            sal += 1/((2*self.padding_size+1)**2)
            sal = sal.sum(dim=0)
            sal /= sal.sum()
            sal = sal.reshape(w_out, h_out).t().unsqueeze(0)  # noqa: E501, add channel dimension
            sals.append(sal)
        return torch.stack(sals)

    def forward(self, imgs, gt_bboxes, jitter=False, **kwargs):
        vis_options = kwargs.get('vis_options', {})

        if isinstance(gt_bboxes, torch.Tensor):
            batch_bboxes = gt_bboxes
        else:
            if len(gt_bboxes[0].shape) == 3:
                batch_bboxes = gt_bboxes[0].clone()  # noqa: E501, removing the augmentation dimension
            else:
                batch_bboxes = [bboxes.clone() for bboxes in gt_bboxes]
        device = batch_bboxes[0].device
        saliency = self.bbox2sal(batch_bboxes, imgs, jitter)

        if self.separable:
            x_saliency = saliency.sum(dim=2)
            y_saliency = saliency.sum(dim=3)
            grid = self.separable_saliency_to_grid(imgs, x_saliency,
                                                   y_saliency, device)
        else:
            grid = self.nonseparable_saliency_to_grid(imgs,
                                                      saliency, device)

        return grid, saliency
        # return grid

class FixedKDEGrid(nn.Module, RecasensSaliencyToGridMixin):
    """Grid generator that uses a fixed saliency map -- KDE SD"""

    def __init__(self, saliency_file, **kwargs):
        super(FixedKDEGrid, self).__init__()
        RecasensSaliencyToGridMixin.__init__(self, **kwargs)
        self.saliency = pickle.load(open(saliency_file, 'rb'))

    def forward(self, imgs, **kwargs):
        vis_options = kwargs.get('vis_options', {})
        device = imgs.device
        self.saliency = self.saliency.to(device)

        if self.separable:
            x_saliency = self.saliency.sum(dim=2)
            y_saliency = self.saliency.sum(dim=3)
            grid = self.separable_saliency_to_grid(imgs, x_saliency,
                                                   y_saliency, device)
        else:
            grid = self.nonseparable_saliency_to_grid(imgs,
                                                      self.saliency, device)

        return grid


class SaliencyKDEGrid(nn.Module, RecasensSaliencyToGridMixin):
    """
    Same grid generator, but takes a dense saliency map directly
    instead of bboxes.
    """
    def __init__(self, **kwargs):
        super().__init__()
        RecasensSaliencyToGridMixin.__init__(self, **kwargs)
        self.input_shape = kwargs.get("input_shape", (1080, 1920))

    def forward(self, imgs, saliency, **kwargs):
        """
        imgs: [B, C, H, W]
        saliency: [B, 1, Hs, Ws] OR [B, Hs, Ws]
                  (we will resize to grid_shape internally)
        returns: grid compatible with warp()
        """
        device = imgs.device
        B, _, H, W = imgs.shape
        h_out, w_out = self.grid_shape  # target saliency resolution used by mixin

        # normalize shape to [B,1,*,*]
        if saliency.dim() == 3:
            saliency = saliency.unsqueeze(1)
        saliency = saliency.to(device).float()

        # resize saliency to (h_out,w_out) expected by saliency_to_grid
        saliency = torch.nn.functional.interpolate(
            saliency, size=(h_out, w_out), mode="bilinear", align_corners=False
        )

        # make it a proper distribution like bbox2sal does (sum to 1)
        saliency = saliency.clamp_min(0)
        saliency = saliency / (saliency.sum(dim=(2, 3), keepdim=True) + 1e-8)

        if self.separable:
            x_saliency = saliency.sum(dim=2)  # [B,1,w_out]
            y_saliency = saliency.sum(dim=3)  # [B,1,h_out]
            grid = self.separable_saliency_to_grid(imgs, x_saliency, y_saliency, device)
        else:
            grid = self.nonseparable_saliency_to_grid(imgs, saliency, device)

        return grid


class CuboidGlobalKDEGrid(nn.Module, RecasensSaliencyToGridMixin):
    """
        Grid generator that uses a two-plane based saliency map 
        which has a fixed parameter set we learn.
    """

    def __init__(self, input_shape, output_shape, **kwargs):
        super(CuboidGlobalKDEGrid, self).__init__()
        RecasensSaliencyToGridMixin.__init__(self, output_shape, **kwargs)
        self.im_shape = input_shape
        self.homo = CuboidLayerGlobal(self.im_shape)
        
    def forward(self, imgs, v_pts,
                **kwargs):
        device = imgs.device

        self.saliency = self.homo.forward(imgs, v_pts)

        self.saliency = F.interpolate(self.saliency, (31, 51))

        if self.separable:
            x_saliency = self.saliency.sum(dim=2)
            y_saliency = self.saliency.sum(dim=3)
            grid = self.separable_saliency_to_grid(imgs, x_saliency,
                                                   y_saliency, device)
        else:
            grid = self.nonseparable_saliency_to_grid(imgs,
                                                      self.saliency, device)
        return grid
    
def invert_grid(grid, input_shape, separable=False):
    f = invert_separable_grid if separable else invert_nonseparable_grid
    return f(grid, list(input_shape))


## ANURAG: This makes it fast but doesn't work on all torch versions
# @torch.jit.script
def invert_separable_grid(grid, input_shape: List[int]):
    grid = grid.clone()
    device = grid.device
    H: int = input_shape[2]
    W: int = input_shape[3]
    B, grid_H, grid_W, _ = grid.shape
    assert B == input_shape[0]

    eps = 1e-8
    grid[:, :, :, 0] = (grid[:, :, :, 0] + 1) / 2 * (W - 1)
    grid[:, :, :, 1] = (grid[:, :, :, 1] + 1) / 2 * (H - 1)
    # grid now ranges from 0 to ([H or W] - 1)
    # TODO: implement batch operations
    inverse_grid = 2 * max(H, W) * torch.ones(
        [B, H, W, 2], dtype=torch.float32, device=device)
    for b in range(B):
        # each of these is ((grid_H - 1)*(grid_W - 1)) x 2
        p00 = grid[b,  :-1,   :-1, :].contiguous().view(-1, 2)  # noqa: 203
        p10 = grid[b, 1:  ,   :-1, :].contiguous().view(-1, 2)  # noqa: 203
        p01 = grid[b,  :-1,  1:  , :].contiguous().view(-1, 2)  # noqa: 203

        ref = torch.floor(p00).to(torch.int)
        v00 = p00 - ref
        v10 = p10 - ref
        v01 = p01 - ref
        vx = p01[:, 0] - p00[:, 0]
        vy = p10[:, 1] - p00[:, 1]

        min_x = int(floor(v00[:, 0].min() - eps))
        max_x = int(ceil(v01[:, 0].max() + eps))
        min_y = int(floor(v00[:, 1].min() - eps))
        max_y = int(ceil(v10[:, 1].max() + eps))

        pts = torch.cartesian_prod(
            torch.arange(min_x, max_x + 1, device=device),
            torch.arange(min_y, max_y + 1, device=device),
        ).T  # 2 x (x_range*y_range)

        unwarped_x = (pts[0].unsqueeze(0) - v00[:, 0].unsqueeze(1)) / vx.unsqueeze(1)  # noqa: E501
        unwarped_y = (pts[1].unsqueeze(0) - v00[:, 1].unsqueeze(1)) / vy.unsqueeze(1)  # noqa: E501
        unwarped_pts = torch.stack((unwarped_y, unwarped_x), dim=0)  # noqa: E501, has shape2 x ((grid_H - 1)*(grid_W - 1)) x (x_range*y_range)

        good_indices = torch.logical_and(
            torch.logical_and(-eps <= unwarped_pts[0],
                              unwarped_pts[0] <= 1+eps),
            torch.logical_and(-eps <= unwarped_pts[1],
                              unwarped_pts[1] <= 1+eps),
        )  # ((grid_H - 1)*(grid_W - 1)) x (x_range*y_range)
        nonzero_good_indices = good_indices.nonzero()
        inverse_j = pts[0, nonzero_good_indices[:, 1]] + ref[nonzero_good_indices[:, 0], 0]  # noqa: E501
        inverse_i = pts[1, nonzero_good_indices[:, 1]] + ref[nonzero_good_indices[:, 0], 1]  # noqa: E501
        # TODO: is replacing this with reshape operations on good_indices faster? # noqa: E501
        j = nonzero_good_indices[:, 0] % (grid_W - 1)
        i = nonzero_good_indices[:, 0] // (grid_W - 1)
        grid_mappings = torch.stack(
            (j + unwarped_pts[1, good_indices], i + unwarped_pts[0, good_indices]),  # noqa: E501
            dim=1
        )
        in_bounds = torch.logical_and(
            torch.logical_and(0 <= inverse_i, inverse_i < H),
            torch.logical_and(0 <= inverse_j, inverse_j < W),
        )
        inverse_grid[b, inverse_i[in_bounds], inverse_j[in_bounds], :] = grid_mappings[in_bounds, :]  # noqa: E501

    inverse_grid[..., 0] = (inverse_grid[..., 0]) / (grid_W - 1) * 2.0 - 1.0  # noqa: E501
    inverse_grid[..., 1] = (inverse_grid[..., 1]) / (grid_H - 1) * 2.0 - 1.0  # noqa: E501
    return inverse_grid


def invert_nonseparable_grid(grid, input_shape):
    grid = grid.clone()
    device = grid.device
    _, _, H, W = input_shape
    B, grid_H, grid_W, _ = grid.shape
    assert B == input_shape[0]

    eps = 1e-8
    grid[:, :, :, 0] = (grid[:, :, :, 0] + 1) / 2 * (W - 1)
    grid[:, :, :, 1] = (grid[:, :, :, 1] + 1) / 2 * (H - 1)
    # grid now ranges from 0 to ([H or W] - 1)
    # TODO: implement batch operations
    inverse_grid = 2 * max(H, W) * torch.ones(
        (B, H, W, 2), dtype=torch.float32, device=device)
    for b in range(B):
        # each of these is ((grid_H - 1)*(grid_W - 1)) x 2
        p00 = grid[b,  :-1,   :-1, :].contiguous().view(-1, 2)  # noqa: 203
        p10 = grid[b, 1:  ,   :-1, :].contiguous().view(-1, 2)  # noqa: 203
        p01 = grid[b,  :-1,  1:  , :].contiguous().view(-1, 2)  # noqa: 203
        p11 = grid[b, 1:  ,  1:  , :].contiguous().view(-1, 2)  # noqa: 203

        ref = torch.floor(p00).type(torch.int)
        v00 = p00 - ref
        v10 = p10 - ref
        v01 = p01 - ref
        v11 = p11 - ref

        min_x = int(floor(min(v00[:, 0].min(), v10[:, 0].min()) - eps))
        max_x = int(ceil(max(v01[:, 0].max(), v11[:, 0].max()) + eps))
        min_y = int(floor(min(v00[:, 1].min(), v01[:, 1].min()) - eps))
        max_y = int(ceil(max(v10[:, 1].max(), v11[:, 1].max()) + eps))

        pts = torch.cartesian_prod(
            torch.arange(min_x, max_x + 1, device=device),
            torch.arange(min_y, max_y + 1, device=device),
        ).T

        # each of these is  ((grid_H - 1)*(grid_W - 1)) x 2
        vb = v10 - v00
        vc = v01 - v00
        vd = v00 - v10 - v01 + v11

        vx = pts.permute(1, 0).unsqueeze(0)  # 1 x (x_range*y_range) x 2
        Ma = v00.unsqueeze(1) - vx  # noqa: E501, ((grid_H - 1)*(grid_W - 1)) x (x_range*y_range) x 2

        vc_cross_vd = (vc[:, 0] * vd[:, 1] - vc[:, 1] * vd[:, 0]).unsqueeze(1)  # noqa: E501, ((grid_H - 1)*(grid_W - 1)) x 1
        vc_cross_vb = (vc[:, 0] * vb[:, 1] - vc[:, 1] * vb[:, 0]).unsqueeze(1)  # noqa: E501, ((grid_H - 1)*(grid_W - 1)) x 1
        Ma_cross_vd = (Ma[:, :, 0] * vd[:, 1].unsqueeze(1) - Ma[:, :, 1] * vd[:, 0].unsqueeze(1))  # noqa: E501, ((grid_H - 1)*(grid_W - 1)) x (x_range*y_range)
        Ma_cross_vb = (Ma[:, :, 0] * vb[:, 1].unsqueeze(1) - Ma[:, :, 1] * vb[:, 0].unsqueeze(1))  # noqa: E501, ((grid_H - 1)*(grid_W - 1)) x (x_range*y_range)

        qf_a = vc_cross_vd.expand(*Ma_cross_vd.shape)
        qf_b = vc_cross_vb + Ma_cross_vd
        qf_c = Ma_cross_vb

        mu_neg = -1 * torch.ones_like(Ma_cross_vd)
        mu_pos = -1 * torch.ones_like(Ma_cross_vd)
        mu_linear = -1 * torch.ones_like(Ma_cross_vd)

        nzie = (qf_a.abs() > 1e-10).expand(*Ma_cross_vd.shape)

        disc = (qf_b[nzie]**2 - 4 * qf_a[nzie] * qf_c[nzie]) ** 0.5
        mu_pos[nzie] = (-qf_b[nzie] + disc) / (2 * qf_a[nzie])
        mu_neg[nzie] = (-qf_b[nzie] - disc) / (2 * qf_a[nzie])
        mu_linear[~nzie] = qf_c[~nzie] / qf_b[~nzie]

        mu_pos_valid = torch.logical_and(mu_pos >= 0, mu_pos <= 1)
        mu_neg_valid = torch.logical_and(mu_neg >= 0, mu_neg <= 1)
        mu_linear_valid = torch.logical_and(mu_linear >= 0, mu_linear <= 1)

        mu = -1 * torch.ones_like(Ma_cross_vd)
        mu[mu_pos_valid] = mu_pos[mu_pos_valid]
        mu[mu_neg_valid] = mu_neg[mu_neg_valid]
        mu[mu_linear_valid] = mu_linear[mu_linear_valid]

        lmbda = -1 * (Ma[:, :, 1] + mu * vc[:, 1:2]) / (vb[:, 1:2] + vd[:, 1:2] * mu)  # noqa: E501

        unwarped_pts = torch.stack((lmbda, mu), dim=0)

        good_indices = torch.logical_and(
            torch.logical_and(-eps <= unwarped_pts[0],
                              unwarped_pts[0] <= 1+eps),
            torch.logical_and(-eps <= unwarped_pts[1],
                              unwarped_pts[1] <= 1+eps),
        )  # ((grid_H - 1)*(grid_W - 1)) x (x_range*y_range)
        nonzero_good_indices = good_indices.nonzero()
        inverse_j = pts[0, nonzero_good_indices[:, 1]] + ref[nonzero_good_indices[:, 0], 0]  # noqa: E501
        inverse_i = pts[1, nonzero_good_indices[:, 1]] + ref[nonzero_good_indices[:, 0], 1]  # noqa: E501
        # TODO: is replacing this with reshape operations on good_indices faster? # noqa: E501
        j = nonzero_good_indices[:, 0] % (grid_W - 1)
        i = nonzero_good_indices[:, 0] // (grid_W - 1)
        grid_mappings = torch.stack(
            (j + unwarped_pts[1, good_indices], i + unwarped_pts[0, good_indices]),  # noqa: E501
            dim=1
        )
        in_bounds = torch.logical_and(
            torch.logical_and(0 <= inverse_i, inverse_i < H),
            torch.logical_and(0 <= inverse_j, inverse_j < W),
        )
        inverse_grid[b, inverse_i[in_bounds], inverse_j[in_bounds], :] = grid_mappings[in_bounds, :]  # noqa: E501

    inverse_grid[..., 0] = (inverse_grid[..., 0]) / (grid_W - 1) * 2.0 - 1.0  # noqa: E501
    inverse_grid[..., 1] = (inverse_grid[..., 1]) / (grid_H - 1) * 2.0 - 1.0  # noqa: E501
    return inverse_grid

def warp(grid, feats):
    warped_feats = F.grid_sample(feats, grid, align_corners=True)
    return warped_feats

def unwarp(inverse_grid, feats):
    unwarped_feats = F.grid_sample(feats, inverse_grid, align_corners=True)
    return unwarped_feats

def unwarp_bboxes(bboxes, grid, output_shape):
    """Unwarps a tensor of bboxes of shape (n, 4) or (n, 5) according to the grid \
    of shape (h, w, 2) used to warp the corresponding image and the \
    output_shape (H, W, ...)."""
    bboxes = bboxes.clone()
    # image map of unwarped (x,y) coordinates
    img = grid.permute(2, 0, 1).unsqueeze(0)

    warped_height, warped_width = grid.shape[0:2]
    xgrid = 2 * (bboxes[:, 0:4:2] / warped_width) - 1
    ygrid = 2 * (bboxes[:, 1:4:2] / warped_height) - 1
    grid = torch.stack((xgrid, ygrid), dim=2).unsqueeze(0)

    # warped_bboxes has shape (2, num_bboxes, 2)
    warped_bboxes = F.grid_sample(
        img, grid, align_corners=True, padding_mode="border").squeeze(0)
    bboxes[:, 0:4:2] = (warped_bboxes[0] + 1) / 2 * output_shape[1]
    bboxes[:, 1:4:2] = (warped_bboxes[1] + 1) / 2 * output_shape[0]

    return bboxes    

def unwarp_masks(masks, grid, output_shape):
    pass

import PIL
from PIL import Image
import cv2

def load_img_warp(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    # w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    # image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    # return image
    return 2.*image - 1.

def save_img_warp(img_tensor, out_path):
    img_tensor = img_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img_tensor = img_tensor[:, :, ::-1]
    img_tensor = (img_tensor + 1.)/2.0
    img_tensor = (img_tensor * 255).astype(np.uint8)
    cv2.imwrite(out_path, img_tensor)
    return img_tensor


if __name__ == "__main__":

    img_path = "dataset/images/denver_1d442d363b3f48f09c34a481a7d56310_000001_02190.jpg"
    img_cv2 = cv2.imread(img_path)
    img = load_img_warp(img_path).to('cuda')

    
    import json
    ann_path = "dataset/annotations/instances_train_gps_split.json"
    with open(ann_path, "r") as f:
        anns = json.load(f)
    
    im_name = img_path.split("/")[-1]
    image_id = [i["id"] for i in anns["images"] if i["file_name"] == im_name][0]
    image_anns = [i for i in anns["annotations"] if i["image_id"] == image_id]
    bboxes = [i["bbox"] for i in image_anns]
    bboxes = torch.tensor(bboxes).to('cuda')
    ## xyxy format 
    bboxes[:, 2:] += bboxes[:, :2]
    bboxes = bboxes.unsqueeze(0)

    ## kornia == 0.6.12
    ## torch == 1.13.1+cu117
    ## uncomment the @torch.jit.script decorator to make it faster


    ## Saliency Warping paper: 
    ### (ECCV 2018) Learning to Zoom: a Saliency-Based Sampling Layer for Neural Networks
    ##
    ## Fovea ICCV 2021 --> They applied zooming to detection
    ## This saliency for the entire dataset
    ## How to compute? Send the entire dataset bboxes to generate the grid
    ## You might need to use PlaneKDEGrid at that point
    ## And then save the grid to a file
    ## And the next time use FIxedKDEGrid to load the grid from the file
    ## This is from the older paper
    # grid_net = FixedKDEGrid(saliency_file="dataset_saliency.pkl", output_shape=(1080, 1920))
    # grid = grid_net(img).to('cuda')

    ## Two-Plane Prior, CVPR 2023
    ## This paper says, if you are using saliency, you should use geometry to define it
    ## This saliency/Vanishing point is for this image only
    ## This is what we use to get VPs: https://github.com/zhou13/neurvps
    ## use pretrained model to get the vanishing points
    ## model trained on TMM17 dataset
    ## Alternate Method to get VPs:
    ## https://github.com/yanconglin/VanishingPoint_HoughTransform_GaussianSphere
    # grid_net = CuboidGlobalKDEGrid(input_shape=(1080, 1920), output_shape=(1080, 1920), separable=True).to('cuda')
    # vpts = torch.tensor([[1044.0, 681.0]]).to('cuda') 
    # grid = grid_net(img, vpts).to('cuda')

    ## Shen/Anurag ECCV submission 
    ## Use GT boxes to define the saliency for domain adaptation
    ##
    ## Definitions of amplitude and bandwidth in this paper:
    ## Saliency Scale in shen's paper is reciprocal of bandwidth
    # https://openaccess.thecvf.com/content/ICCV2021/papers/Thavamani_FOVEA_Foveated_Image_Magnification_for_Autonomous_Navigation_ICCV_2021_paper.pdf
    # FOVEA ICCV 2021
    # 31x51 grid
    # Equation 8 a and b are the amplitude and bandwidth respectively
    ## this is per image and not per dataset saliency
    grid_net = PlainKDEGrid(input_shape=(1080, 1920), output_shape=(1080, 1920), separable=True, bandwidth_scale=32, amplitude_scale=1.0).to('cuda')
    grid = grid_net(img, gt_bboxes=bboxes).to('cuda')

    save_img_warp(img, "original_img.png")

    warped_img = warp(grid, img)

    save_img_warp(warped_img, "warped_img.png")

    inverse_grid = invert_grid(grid, (1, 3, 1080, 1920), separable=True)
    ## We sent it the warped image, instead we send it the warped features.
    ## features = network(img)
    ### boxes = head(features)
    ###
    ### How to use this?
    ### warped_img = warp(grid, img)
    ### warped_features = network(warped_img) ## imagine this to be backbone
    ### unwarped_features = unwarp(inverse_grid, warped_features)
    ### boxes = head(unwarped_features)
    ###
    ### Unwarp is from LZU paper (Learning to Zoom and Unzoom), CVPR 2023
    unwarped_img = unwarp(inverse_grid, warped_img)

    unwarped_img_cv2 = save_img_warp(unwarped_img, "unwarped_img.png")

    difference = img_cv2 - unwarped_img_cv2
    cv2.imwrite("difference.png", difference)

