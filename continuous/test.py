import torch
import torch.nn as nn
import numpy as np


class CDF_quadratic(nn.Module):
    """
    Non-linear CDF layers
    Here, the domain means the range of input variables
    n_bins: number of bins for discreting the domain or range
    input_dim: input var's dimension
    r: for generating the mesh for discreting domain
    bound: bound of domain
    """

    def __init__(self, n_bins, input_dim, r=1.2, bound=50.0, **kwargs):
        super(CDF_quadratic, self).__init__(**kwargs)

        assert n_bins % 2 == 0

        self.n_bins = n_bins
        self.input_dim = input_dim
        # generate a nonuniform mesh symmetric to zero,
        # and increasing by ratio r away from zero.
        self.bound = bound
        self.r = r

        m = n_bins / 2  #
        x1L = bound * (r - 1.0) / (np.power(r, m) - 1.0)

        index = torch.reshape(torch.arange(0, self.n_bins + 1, dtype=torch.float32), (-1, 1))
        index -= m
        xr = torch.where(index >= 0, (1. - torch.pow(r, index)) / (1. - r),
                         (1. - torch.pow(r, torch.abs(index))) / (1. - r))  # xr先设置成一个对称的向量
        xr = torch.where(index >= 0, x1L * xr, -x1L * xr)  # torch.where(condition, x, y), if condition xr[i] = x else y
        # x1L is a scaled factor, now xr is a symmetric non-uniform vector in [-bound, bound]
        xr = torch.reshape(xr, (-1, 1))
        xr = (xr + bound) / 2.0 / bound  # xr:[-bound, bound] --> [0, 2*bound] --> [0, 1]

        self.x1L = x1L / 2.0 / bound
        mesh = torch.cat([torch.reshape(torch.tensor([0.0]), (-1, 1)), torch.reshape(xr[1:-1, 0], (-1, 1)),
                          torch.reshape(torch.tensor([1.0]), (-1, 1))], 0)  # replace head and tail of original xr with 0 and 1
        self.register_buffer('mesh', mesh)
        elmt_size = torch.reshape(self.mesh[1:] - self.mesh[:-1], (-1, 1))  # the interval of adjacent element of mesh.
        self.register_buffer('elmt_size', elmt_size)
        self.p = nn.Parameter(torch.zeros(self.n_bins - 1, input_dim))

    def forward(self, x, t=None):
        self._pdf_normalize()
        # rescale, such points in [-bound, bound] will be mapped to [0,1]
        x = (x + self.bound) / 2.0 / self.bound

        # cdf mapping
        x, logdet = self._cdf(x)

        # maps [0,1] back to [-bound, bound]
        x = x * 2.0 * self.bound - self.bound
        return x, logdet

    def inverse(self, z, t=None):
        self._pdf_normalize()
        # rescale such points in [-bound, bound] will be mapped to [0,1]
        x = (z + self.bound) / 2.0 / self.bound

        # cdf mapping
        x, logdet = self._cdf_inv(x)

        # maps [0,1] back to [-bound, bound]
        x = x * 2.0 * self.bound - self.bound
        return x, logdet

    # normalize the piecewise representation of pdf
    def _pdf_normalize(self):
        # peicewise pdf
        p0 = torch.ones((1, self.input_dim), dtype=torch.float32, device=self.mesh.device)
        self.pdf = p0
        # px 计算过程: self.p is a trainable parameter
        px = torch.exp(self.p) * (self.elmt_size[:-1] + self.elmt_size[1:]) / 2.0  # p(si), i !=0 or m+1 in paper
        px = (1 - self.elmt_size[0]) / torch.sum(px, 0, keepdim=True)
        px = px * torch.exp(self.p)
        self.pdf = torch.concat([self.pdf, px], 0)
        self.pdf = torch.concat([self.pdf, p0], 0)  # self.pdf: p(si) in paper,

        # probability in each element
        cell = (self.pdf[:-1, :] + self.pdf[1:, :]) / 2.0 * self.elmt_size
        # CDF - contribution from previous elements.
        r_zeros = torch.zeros((1, self.input_dim), dtype=torch.float32, device=self.mesh.device)
        self.F_ref = r_zeros
        for i in range(1, self.n_bins):
            tp = torch.sum(cell[:i, :], 0, keepdim=True)
            self.F_ref = torch.concat([self.F_ref, tp], 0)

    # the cdf is a piecewise quadratic function.
    def _cdf(self, x):
        x_sign = torch.sign(x - 0.5)
        m = torch.floor(torch.log(torch.abs(x - 0.5) * (self.r - 1) / self.x1L + 1.0) / np.log(self.r))
        k_ind = torch.where(x_sign >= 0, self.n_bins / 2 + m, self.n_bins / 2 - m - 1)
        k_ind = k_ind.to(dtype=torch.int64)
        cover = torch.where(k_ind * (k_ind - self.n_bins + 1) <= 0, 1.0, 0.0)

        # print('k_ind', k_ind)
        k_ind = torch.where(k_ind < 0, 0 * k_ind, k_ind)
        k_ind = torch.where(k_ind > (self.n_bins - 1), (self.n_bins - 1) * torch.ones_like(k_ind), k_ind)

        # print(self.pdf[:,0].shape)

        # print(k_ind[:,0])
        v1 = torch.reshape(torch.gather(self.pdf[:, 0], 0, k_ind[:, 0]), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(torch.gather(self.pdf[:, i], 0, k_ind[:, i]), (-1, 1))
            v1 = torch.concat([v1, tp], 1)

        v2 = torch.reshape(torch.gather(self.pdf[:, 0], 0, k_ind[:, 0] + 1), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(torch.gather(self.pdf[:, i], 0, k_ind[:, i] + 1), (-1, 1))
            v2 = torch.concat([v2, tp], 1)

        xmodi = torch.reshape(x[:, 0] - torch.gather(self.mesh[:, 0], 0, k_ind[:, 0]), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(x[:, i] - torch.gather(self.mesh[:, 0], 0, k_ind[:, i]), (-1, 1))
            xmodi = torch.concat([xmodi, tp], 1)

        h_list = torch.reshape(torch.gather(self.elmt_size[:, 0], 0, k_ind[:, 0]), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(torch.gather(self.elmt_size[:, 0], 0, k_ind[:, i]), (-1, 1))
            h_list = torch.concat([h_list, tp], 1)

        F_pre = torch.reshape(torch.gather(self.F_ref[:, 0], 0, k_ind[:, 0]), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(torch.gather(self.F_ref[:, i], 0, k_ind[:, i]), (-1, 1))
            F_pre = torch.concat([F_pre, tp], 1)

        y = torch.where(cover > 0, F_pre + xmodi ** 2 / 2.0 * (v2 - v1) / h_list + xmodi * v1, x)

        dlogdet = torch.where(cover > 0, xmodi * (v2 - v1) / h_list + v1, torch.ones_like(cover))
        dlogdet = torch.sum(torch.log(dlogdet), dim=[1])

        return y, dlogdet

    # inverse of the cdf
    def _cdf_inv(self, y):
        xr = torch.broadcast_to(self.mesh, [self.n_bins + 1, self.input_dim])
        yr1, _ = self._cdf(xr)

        p0 = torch.zeros((1, self.input_dim), device=self.mesh.device, dtype=torch.float32)
        p1 = torch.ones((1, self.input_dim), device=self.mesh.device, dtype=torch.float32)
        yr = torch.concat([p0, yr1[1:-1, :], p1], 0)

        k_ind = torch.searchsorted((yr.T).contiguous(), (y.T).contiguous(), right=True)
        k_ind = torch.transpose(k_ind, 0, 1)
        k_ind = k_ind.to(dtype=torch.int64)
        k_ind -= 1

        cover = torch.where(k_ind * (k_ind - self.n_bins + 1) <= 0, 1.0, 0.0)

        k_ind = torch.where(k_ind < 0, 0, k_ind)
        k_ind = torch.where(k_ind > (self.n_bins - 1), self.n_bins - 1, k_ind)

        c_cover = torch.reshape(cover[:, 0], (-1, 1))

        v1 = torch.where(c_cover > 0, torch.reshape(torch.gather(self.pdf[:, 0], 0, k_ind[:, 0]), (-1, 1)),
                         -1. * torch.ones_like(c_cover))
        for i in range(1, self.input_dim):
            c_cover = torch.reshape(cover[:, i], (-1, 1))
            tp = torch.where(c_cover > 0, torch.reshape(torch.gather(self.pdf[:, i], 0, k_ind[:, i]), (-1, 1)),
                             -1.0 * torch.ones_like(c_cover))
            v1 = torch.concat([v1, tp], 1)

        c_cover = torch.reshape(cover[:, 0], (-1, 1))
        v2 = torch.where(c_cover > 0, torch.reshape(torch.gather(self.pdf[:, 0], 0, k_ind[:, 0] + 1), (-1, 1)),
                         -2.0 * torch.ones_like(c_cover))
        for i in range(1, self.input_dim):
            c_cover = torch.reshape(cover[:, i], (-1, 1))
            tp = torch.where(c_cover > 0, torch.reshape(torch.gather(self.pdf[:, i], 0, k_ind[:, i] + 1), (-1, 1)),
                             -2.0 * torch.ones_like(c_cover))
            v2 = torch.concat([v2, tp], 1)

        ys = torch.reshape(y[:, 0] - torch.gather(yr[:, 0], 0, k_ind[:, 0]), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(y[:, i] - torch.gather(yr[:, i], 0, k_ind[:, i]), (-1, 1))
            ys = torch.concat([ys, tp], 1)

        xs = torch.reshape(torch.gather(xr[:, 0], 0, k_ind[:, 0]), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(torch.gather(xr[:, i], 0, k_ind[:, i]), (-1, 1))
            xs = torch.concat([xs, tp], 1)

        h_list = torch.reshape(torch.gather(self.elmt_size[:, 0], 0, k_ind[:, 0]), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(torch.gather(self.elmt_size[:, 0], 0, k_ind[:, i]), (-1, 1))
            h_list = torch.concat([h_list, tp], 1)

        tp = 2.0 * ys * h_list * (v2 - v1)
        tp += v1 * v1 * h_list * h_list
        tp = torch.sqrt(tp) - v1 * h_list
        tp = torch.where(torch.abs(v1 - v2) < 1.0e-6, ys / v1, tp / (v2 - v1))
        tp += xs

        x = torch.where(cover > 0, tp, y)

        tp = 2.0 * ys * h_list * (v2 - v1)
        tp += v1 * v1 * h_list * h_list
        tp = h_list / torch.sqrt(tp)

        dlogdet = torch.where(cover > 0, tp, torch.ones_like(cover))
        dlogdet = torch.sum(torch.log(dlogdet), dim=[1])

        return x, dlogdet


def check_devices():
    gpu_count = torch.cuda.device_count()
    devices = [torch.device(f'cuda:{i}') for i in range(gpu_count)]
    print('gpu num is: {}'.format(gpu_count))
    return devices


devices = check_devices()
# 定义 CDF_quadratic 层
n_bins = 10
input_dim = 3
cdf_layer = CDF_quadratic(n_bins, input_dim)

# 示例输入数据
x = torch.randn(5, input_dim)

# 正向传播
y, log_det = cdf_layer(x)
print("Transformed Output:", y)
print("Log-Abs-Det-Jacobian:", log_det)

# 逆向传播
x_recovered, inv_log_det = cdf_layer.inverse(y)
print("Recovered Output:", x_recovered)
print("Inverse Log-Abs-Det-Jacobian:", inv_log_det)


