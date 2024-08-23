This file is mainly use to introduce how the fuctions and classes are used in our code.

# Multi-dimensional gaussian distribution

The aim of KRnet is mapping the pinn-loss disribution to a multi-dimensional standard normal distribution, in the code we construct a class to reperesent it.

```python
 class DiagGaussian():
    """
    Multivariate Gaussian distribution with diagonal covariance matrix
    """

    def __init__(self, mu, cov):
        """Constructor
        Args:
          shape: Tuple with shape of data, if int shape has one dimension
          trainable: Flag whether to use trainable or fixed parameters
        """
        # super().__init__()
        self.shape = mu.shape
        self.d = np.prod(self.shape)  # dim

        self.loc = mu
        self.scale = torch.sqrt(torch.diagonal(cov)).view(-1, )  # diagonal elements std: \sigma_1, \sigma_2  
        self.log_scale = torch.log(self.scale)  # torch.zeros(1, *self.shape) # log sigma

    def forward(self, num_samples=1):
        eps = torch.randn(
            (num_samples,) + self.shape, dtype=self.loc.dtype, device=self.loc.device
        )  # generate samples satisfy N(0,1)
        log_scale = self.log_scale  # log \sigma_i

        z = self.loc + self.scale * eps  # d-dimension variable obey multivariate Gaussian distribution
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow(eps, 2), -1
        )  # -0.5 denote log sqrt(2*pi)^{-1} = - 0.5 log 2*pi , log_p : log probability density function
        return z, log_p

    def sample(self, sample_shape=torch.Size()):
        z, log_p = self.forward(sample_shape[0])
        return z

    def log_prob(self, z):
        log_scale = self.log_scale
        # qurad_form = torch.pow((z - self.loc) / torch.exp(log_scale), 2)
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow((z - self.loc) / self.scale, 2),
            dim=-1)
        return log_p
```

this is a standard multivariate gaussian distribution with 4 inner function.

- `.__init__(mu, cov)` takes 2 vector mu, cov as initial parameters, generally mu is a 1-dim vector, (e.g. $[\mu_1, \mu_2]$); cov is covariance (e.g. $cov = \begin{bmatrix} \sigma_{11}, \sigma_{12} \\ \sigma_{21},\sigma_{22} \end{bmatrix}$), simply we only focus on Multivariate Gaussian distribution with diagonal covariance matrix. therefore, $\sigma_{ij}=0, i \neq j$, e.g. cov = [[1,0],[0,1]]
- `.forward(num_samples=1)` takes 1 parameter: `num_samples` to generate `num_samples` data which satisfies the initialized multivariate Gaussian distribution. It firstly generate n samples satisfy multivariate standard gaussian distribution. then $Z = \sigma X + \mu, X \sim (0, 1), Z \sim (\mu, \sigma^2)$, then return Z and log probability density.
- `.sample(self, sample_shape=torch.Size())` take a tuple or list parameter(e.g. (n_sample,) ), using `.forward()`function to generate samples.
- `.log_prob(self, z)` using sample data `z` to compute log Joint probability density. Since Z satisfies iid components multivariate gaussian distribution,

$$
f(x_1,\cdots, x_k) = \prod_{i=1}^{k} f(x_i) =\prod_{i=1}^k \dfrac{1}{\sqrt{2\pi}\sigma_i} e^{-\frac{(x_i -\mu)^2}{2\sigma_i^2}} = (2\pi)^{-\frac{k}{2}} \prod_{i=1}^k \sigma_i^{-1} e^{-\frac{(x_i -\mu)^2}{2\sigma_i^2}} \\
\log f(x_1,\cdots, x_k) = -\frac{k}{2} * \log (2\pi) - \sum_i (\log \sigma_i +\frac{(x_i -\mu)^2}{2\sigma_i^2} )
$$

# AffineCoupling layer

This layer is an important layer in real NVP, this layer is used to map X to Z, a single affinecoupling layer can be seen as $f_{[i]}$ as below.

$$
z=f(x)=f_{[L]}\circ\ldots\circ f_{[1]}(x)\quad\mathrm{and}\quad x=f^{-1}(z)=f_{[1]}^{-1}\circ\ldots\circ f_{[L]}^{-1}(z),
$$

```python

class AffineCoupling(nn.Module):
    """ Affine Coupling Layers 
    Args:
        input_size: input var dimension, size of input tensor: such as X in R^D , then input_size = D,
        split_size: split size of input var : X = [X[0:d] , X[d:D]], split_size=D-d
        hidden_size: width of hidden layers
        n_hidden: depth of hidden layers
        cond_label_size: condition variable size
    """
    def __init__(self, input_size, split_size, hidden_size, n_hidden, cond_label_size=None):
        super().__init__()

        self.log_beta = nn.Parameter(torch.zeros(input_size-split_size, dtype=torch.float32))
        self.input_size = input_size
        self.split_size = split_size
        net = [nn.Linear(split_size + (cond_label_size if cond_label_size is not None else 0), hidden_size+split_size + (cond_label_size if cond_label_size is not None else 0)), 
                nn.ReLU(), 
                nn.Linear(hidden_size + split_size + (cond_label_size if cond_label_size is not None else 0), hidden_size)]
        for _ in range(n_hidden):
            net += [nn.ReLU(),nn.Linear(hidden_size, hidden_size)]# ResNet_block(hidden_size, hidden_size)
        net += [nn.ReLU(), nn.Linear(hidden_size, 2*(input_size-split_size))]
        # 2*(input_size-split_size), half for s_i, half for t_i
        self.net = nn.Sequential(*net)
        self.alpha = 0.6

    def forward(self, x, y=None):
        """
        x: torch tensor, (N, d)
        y: torch tensor, (N, cond_dim), condition var
        """
        x1 = (x[:,:self.split_size]).view(-1,self.split_size)
        x2 = (x[:,self.split_size:]).view(-1,self.input_size-self.split_size)
  
        h = self.net(x1 if y is None else torch.cat([x1,y], dim=1))
        s = h[:, :self.input_size-self.split_size]
        s = s.view(-1, self.input_size-self.split_size)
        t = h[:,self.input_size-self.split_size:] 
        t = t.view(-1, self.input_size-self.split_size)

        u2 = x2 + (self.alpha*x2*torch.tanh(s) + torch.exp(torch.clip(self.log_beta, -5.0, 5.0)) * torch.tanh(t))

        log_abs_det_jacobian = torch.log(1+self.alpha*torch.tanh(s))  # 比论文里多写一个log
        log_abs_det_jacobian = log_abs_det_jacobian.sum(dim=1)
        return torch.cat([x1, u2], dim=-1), log_abs_det_jacobian

    def inverse(self, u, y=None):
        u1 = (u[:,:self.split_size]).view(-1,self.split_size)
        u2 = (u[:,self.split_size:]).view(-1,self.input_size-self.split_size)

        h = self.net(u1 if y is None else torch.cat([u1,y], dim=1))
        s = h[:,:self.input_size-self.split_size]
        s = s.view(-1, self.input_size-self.split_size)
        t = h[:,self.input_size-self.split_size:] 
        t = t.view(-1, self.input_size-self.split_size)

        x2 = (u2 - torch.exp(torch.clip(self.log_beta, -5.0, 5.0))*torch.tanh(t))/(1 + self.alpha*torch.tanh(s))
        log_abs_det_jacobian = -torch.log(1 + self.alpha*torch.tanh(s))  # 原本是倒数， 取对数后变为负数
        log_abs_det_jacobian = log_abs_det_jacobian.sum(dim=1)
        return torch.cat([u1,x2],dim=-1), log_abs_det_jacobian
```

this layer have a special structure, all of its layer are reversible and have explicit derivation.

$$
p_X(x)=p_Z(f(x))|\det\nabla_xf|. \\
|\det\nabla_xf|=\prod_{i=1}^L\left|\det\nabla_{x_{[i-1]}}f_{[i]}\right|,
$$

we indicate $x_{[i-1]}$ as the intermediate variables with $x_{[0]} = input:x; \quad  x_{[L]}=output:z.$ For each layer input $x_{[i]} \in R^m$ and it will divide into 2 parts $x_{[i,1]} \in R^d, x_{[i,2]} \in R^{m-d}.$

we define the affinecoupling layer with 3 function as follow:

$$
\begin{aligned}&x_{[i],1}=x_{[i-1],1}\\&x_{[i],2}=x_{[i-1],2}\odot\left(1+\alpha\tanh(s_{i}(x_{[i-1],1}))\right)+e^{\beta_{i}}\odot\tanh(t_{i}(x_{[i-1],1})), \\&(s_i,t_i)=\text{NN}_{[i]}(x_{[i-1],1}).\end{aligned}
$$

an input vector x go through this affinecoupling layer and return [y1, y2], in above $s_i, t_i$ have the same shape as $x_2$, split_size = len(x1). Since x2 will take Hadamard product with $s_i$ and $t_i$, output size of NN is 2 * len(x2) = 2*(m-d)

cond_label_size=None, and it is useless in DAS experiment.

- `__init__(input_size, split_size, hidden_size, n_hidden, cond_label_size=None)` input_size = len(x) = m, split_size = len(x1) = d, hidden_size: related to neural network hidden size, n_hidden: depth of hidden layers. it has a trainable parameter $\log \beta$ and a neural network with input_size=d, output_size=2*(m-d), len(s)=len(t)=m-d. network flow: x --> NN --> s
- `forward(self, x, y=None)` : it takes x as input and use affinecoupling layer to compute s, t, then use above equation to compute y2. Lastly, return [y1, y2] and log_abs_det_jacobian: $\log\left|\det\nabla_{x}f(x)\right|$. The detail compute progress:

$$
x = \begin{bmatrix}x_1 \\ x_2\end{bmatrix}, f(x) = f\begin{bmatrix} & x_1 \\ & x_2\end{bmatrix} = \begin{bmatrix}x_1 \\ x_{2}=x_{2}\odot\left(1+\alpha\tanh(s(x_{1}))\right)+e^{\beta}\odot\tanh(t_(x_{1}))\end{bmatrix} \\

\nabla_x f(x) = \begin{bmatrix}1 & 0 \\ \nabla_{x_1}f(x_2) & 1 + \alpha \tanh(s(x_1)) \end{bmatrix}, \quad \det \nabla_x f(x) = 1 + \alpha \tanh(s(x_1))
$$

- `inverse(self, u, y=None)` use u = [y1, y2] to compute original [x1, x2] and log_abs_det_jacobian: $\log\left|\det\nabla_{y}f^{-1}(y)\right|$.
  $$
  \begin{bmatrix}x_1 \\ x_2\end{bmatrix} = f^{-1} \begin{bmatrix}y_1 \\ y_2\end{bmatrix} = \begin{bmatrix}y_1 \\ \frac{y_2 -e^{\beta}\odot\tanh(t_(y_{1}))}{1 + \alpha \tanh(s(y_1))} \end{bmatrix} \\

  \det \nabla_y f^{-1}(y) = \det \begin{bmatrix}1 & 0 \\ \nabla_{y_1}f(y_2) & \frac{1}{1 + \alpha \tanh(s(y_1))} \end{bmatrix} = \frac{1}{1 + \alpha \tanh(s(y_1))}
  $$

# squeezing layer

Squeezing layer $L_S$ is used to deactivate some dimensions using a mask

$$
q=[\underbrace{1,\ldots,1,\underbrace{0,\ldots,0}_{d-n}]^{\mathsf{T}},}_{n}
$$

the components $q \odot x$ will keep being updated at the layer or net after, and the rest components $(1 − q)  \odot x$ will be fixed from then on.

```python
class squeezing(nn.Module):
    """ KRnet squeezing layer
    Args:
        input_size: batch_inputs (N, d), N*d
        n_cut: reduce dim, from (N,d)->(N,d-n_cut)
    forward func: input
    """
    def __init__(self, input_size, n_cut=1):
        super().__init__()
        self.data_init = True
        self.input_size = input_size
        self.n_cut = n_cut
        self.x = None  

    def forward(self, x):    # 程序中没有使用这个用法
        # log_det = torch.zeros()
        n_dim = x.shape[-1]
        if n_dim<self.n_cut:
            raise Exception("Input dimension is less than n_cut.")
        if self.input_size==n_dim:  # N*d = d, N=1
            if self.input_size>self.n_cut:  # d>n
                if self.x is not None:
                    raise Exception("x is already set.")
                else:
                    self.x = x[:,(n_dim-self.n_cut):]  # 输入x后n个元素
                    z = x[:,:(n_dim-self.n_cut)]  # 前d-n个
            else:
                self.x = None
        elif n_dim<=self.n_cut:  # N != 1  &  d < n,
            z = torch.cat([x, self.x], dim=-1)
            self.x = None
        else:
            cut = x[:, (n_dim-self.n_cut):]
            self.x = torch.cat([cut,self.x], dim=-1)
            z = x[:,:(n_dim-self.n_cut)]
        return z, 0
    def inverse(self, z):
        n_dim = z.shape[-1]
        if self.input_size == n_dim:
            n_start = self.input_size % self.n_cut
            if n_start == 0:
                n_start+=self.n_cut
            self.x = z[:, n_start:]
            x = z[:,:n_start]  # 输入的前n个元素
        else:
            x_length = self.x.shape[-1]
            if x_length<self.n_cut:
                raise Exception()
  
            cut = self.x[:, :self.n_cut]
            x = torch.cat([z, cut],dim=-1)
            if (x_length-self.n_cut)==0:
                self.x = None
            else:
                self.x = self.x[:, self.n_cut:]
        return x, 0
```

`__init__(input_size, n_cut=1)` takes 2 parameters `input_size, n_cut` to initialize. input_size=len(q), split_size: size of part that will be fixed in x.

`forward(x)` input x , return x[0: n_dim - n_nut] as z, and save x[n_dim - n_cut :] as self.x

`inverse(z)` input z return original x=[z, self.x]

# ActNorm layer

this layer plays a role of batchnormalization

$$
\hat{x}=a\odot x+b,
$$


```python
class ActNorm(nn.Module):
    """ ActNorm layer, scale-bias layer 
    Args:
        input_size: input size of input var
        scale: scale parameter, default 1.0
        logscale_factor: log scale parameter, default 3.0
    """
    def __init__(self, input_size, scale = 1.0, logscale_factor = 3.0):
        super().__init__()
        self.scale = scale
        self.logscale_factor = logscale_factor
        self.data_init = True

        self.b = nn.Parameter(torch.zeros(1,input_size))
        self.register_buffer('b_init', torch.zeros(1,input_size))
        self.logs = nn.Parameter(torch.zeros(1,input_size))
        self.register_buffer('logs_init', torch.zeros(1,input_size))

    def forward(self, x, cond_y=None):
        if not self.data_init:
            x_mean = torch.mean(x, 0, keepdim=True)
            x_var = torch.mean(torch.square(x-x_mean), [0], keepdim=True)

            self.b_init = -x_mean
            self.logs_init = torch.log(self.scale/(torch.sqrt(x_var)+1e-6))/self.logscale_factor

            self.data_init = True
        y = x + self.b + self.b_init
        y = y*torch.exp(torch.clip(self.logs + self.logs_init, -5., 5.))

        log_abs_det_jacobian = torch.clip(self.logs + self.logs_init, -5., 5.)
        return y, log_abs_det_jacobian.expand_as(x).sum(dim=-1)

    def inverse(self, y, cond_y=None):
        x = y * torch.exp(-torch.clip(self.logs + self.logs_init, -5., 5.))
        x = x - (self.b + self.b_init)
        log_abs_det_jacobian = -torch.clip(self.logs + self.logs_init, -5., 5.)
        return x, log_abs_det_jacobian.expand_as(x).sum(dim=-1)
    def reset_data_initialization(self):
        self.data_init = False
```

`forward(x)` process: $y = (x + b + b_{init}) * (s * s_{init}) := a \odot x + b$, simply, s, s_init = 1, b_init = 0. $\nabla_x y = a = (s*s_{init} )$


# Non-linear CDF layer


```python
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

        m = n_bins/2
        x1L = bound*(r-1.0)/(np.power(r, m)-1.0)

        index = torch.reshape(torch.arange(0, self.n_bins+1, dtype=torch.float32),(-1,1))
        index -= m
        xr = torch.where(index>=0, (1.-torch.pow(r, index))/(1.-r),
                      (1.-torch.pow(r,torch.abs(index)))/(1.-r))
        xr = torch.where(index>=0, x1L*xr, -x1L*xr)
        xr = torch.reshape(xr,(-1,1))
        xr = (xr + bound)/2.0/bound

        self.x1L = x1L/2.0/bound
        mesh = torch.cat([torch.reshape(torch.tensor([0.0]),(-1,1)), torch.reshape(xr[1:-1,0],(-1,1)), torch.reshape(torch.tensor([1.0]),(-1,1))],0) 
        self.register_buffer('mesh', mesh)
        elmt_size = torch.reshape(self.mesh[1:] - self.mesh[:-1],(-1,1))
        self.register_buffer('elmt_size', elmt_size)
        self.p = nn.Parameter(torch.zeros(self.n_bins-1, input_dim))


    def forward(self, x, t=None):
        self._pdf_normalize()
        # rescale such points in [-bound, bound] will be mapped to [0,1]
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
        p0 = torch.ones((1,self.input_dim), dtype=torch.float32, device=self.mesh.device)
        self.pdf = p0
        px = torch.exp(self.p)*(self.elmt_size[:-1]+self.elmt_size[1:])/2.0
        px = (1 - self.elmt_size[0])/torch.sum(px, 0, keepdim=True)
        px = px*torch.exp(self.p)
        self.pdf = torch.concat([self.pdf, px], 0)
        self.pdf = torch.concat([self.pdf, p0], 0)

        # probability in each element
        cell = (self.pdf[:-1,:] + self.pdf[1:,:])/2.0*self.elmt_size
        # CDF - contribution from previous elements.
        r_zeros= torch.zeros((1,self.input_dim), dtype=torch.float32, device=self.mesh.device)
        self.F_ref = r_zeros
        for i in range(1, self.n_bins):
            tp  = torch.sum(cell[:i,:], 0, keepdim=True)
            self.F_ref = torch.concat([self.F_ref, tp], 0)

    # the cdf is a piecewise quadratic function.
    def _cdf(self, x):
        x_sign = torch.sign(x-0.5)
        m = torch.floor(torch.log(torch.abs(x-0.5)*(self.r-1)/self.x1L + 1.0)/np.log(self.r))
        k_ind = torch.where(x_sign >= 0, self.n_bins/2 + m, self.n_bins/2 - m - 1)
        k_ind = k_ind.to(dtype=torch.int64)
        cover = torch.where(k_ind*(k_ind-self.n_bins+1)<=0, 1.0, 0.0)

        # print('k_ind', k_ind) 
        k_ind = torch.where(k_ind < 0, 0*k_ind, k_ind)
        k_ind = torch.where(k_ind > (self.n_bins-1), (self.n_bins-1)*torch.ones_like(k_ind), k_ind)

        # print(self.pdf[:,0].shape)
      
        # print(k_ind[:,0])
        v1 = torch.reshape(torch.gather(self.pdf[:,0], 0,k_ind[:,0]),(-1,1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(torch.gather(self.pdf[:,i], 0,k_ind[:,i]),(-1,1))
            v1 = torch.concat([v1, tp], 1)

        v2 = torch.reshape(torch.gather(self.pdf[:,0], 0,k_ind[:,0]+1),(-1,1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(torch.gather(self.pdf[:,i], 0,k_ind[:,i]+1),(-1,1))
            v2 = torch.concat([v2, tp], 1)

        xmodi = torch.reshape(x[:,0] - torch.gather(self.mesh[:,0], 0,k_ind[:, 0]), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(x[:,i] - torch.gather(self.mesh[:,0], 0,k_ind[:, i]), (-1, 1))
            xmodi = torch.concat([xmodi, tp], 1)

        h_list = torch.reshape(torch.gather(self.elmt_size[:,0], 0,k_ind[:,0]),(-1,1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(torch.gather(self.elmt_size[:,0], 0,k_ind[:,i]),(-1,1))
            h_list = torch.concat([h_list, tp], 1)

        F_pre = torch.reshape(torch.gather(self.F_ref[:, 0], 0,k_ind[:, 0]), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(torch.gather(self.F_ref[:, i], 0,k_ind[:, i]), (-1, 1))
            F_pre = torch.concat([F_pre, tp], 1)

        y = torch.where(cover>0, F_pre + xmodi**2/2.0*(v2-v1)/h_list + xmodi*v1, x)

        dlogdet = torch.where(cover > 0, xmodi * (v2 - v1) / h_list + v1, torch.ones_like(cover))
        dlogdet = torch.sum(torch.log(dlogdet), dim=[1])

        return y, dlogdet

    # inverse of the cdf
    def _cdf_inv(self, y):
        xr = torch.broadcast_to(self.mesh, [self.n_bins+1, self.input_dim])
        yr1,_ = self._cdf(xr)

        p0 = torch.zeros((1,self.input_dim), device=self.mesh.device,dtype=torch.float32)
        p1 = torch.ones((1,self.input_dim), device=self.mesh.device,dtype=torch.float32)
        yr = torch.concat([p0, yr1[1:-1,:], p1], 0)

        k_ind = torch.searchsorted((yr.T).contiguous(), (y.T).contiguous(), right=True)
        k_ind = torch.transpose(k_ind,0,1)
        k_ind = k_ind.to(dtype=torch.int64)
        k_ind -= 1

        cover = torch.where(k_ind*(k_ind-self.n_bins+1) <= 0, 1.0, 0.0)

        k_ind = torch.where(k_ind < 0, 0, k_ind)
        k_ind = torch.where(k_ind > (self.n_bins-1), self.n_bins-1, k_ind)

        c_cover = torch.reshape(cover[:,0], (-1,1))

        v1 = torch.where(c_cover > 0, torch.reshape(torch.gather(self.pdf[:,0],0, k_ind[:,0]),(-1,1)), -1.*torch.ones_like(c_cover))
        for i in range(1, self.input_dim):
            c_cover = torch.reshape(cover[:,i], (-1,1))
            tp = torch.where(c_cover > 0, torch.reshape(torch.gather(self.pdf[:,i],0, k_ind[:,i]),(-1,1)), -1.0*torch.ones_like(c_cover))
            v1 = torch.concat([v1, tp], 1)

        c_cover = torch.reshape(cover[:,0], (-1,1))
        v2 = torch.where(c_cover > 0, torch.reshape(torch.gather(self.pdf[:,0],0, k_ind[:,0]+1),(-1,1)), -2.0*torch.ones_like(c_cover))
        for i in range(1, self.input_dim):
            c_cover = torch.reshape(cover[:,i], (-1,1))
            tp = torch.where(c_cover > 0, torch.reshape(torch.gather(self.pdf[:,i],0, k_ind[:,i]+1),(-1,1)), -2.0*torch.ones_like(c_cover))
            v2 = torch.concat([v2, tp], 1)

        ys = torch.reshape(y[:, 0] - torch.gather(yr[:, 0],0, k_ind[:, 0]), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(y[:, i] - torch.gather(yr[:, i],0, k_ind[:, i]), (-1, 1))
            ys = torch.concat([ys, tp], 1)

        xs = torch.reshape(torch.gather(xr[:, 0],0, k_ind[:, 0]), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(torch.gather(xr[:, i],0, k_ind[:, i]), (-1, 1))
            xs = torch.concat([xs, tp], 1)

        h_list = torch.reshape(torch.gather(self.elmt_size[:,0],0, k_ind[:,0]),(-1,1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(torch.gather(self.elmt_size[:,0],0, k_ind[:,i]),(-1,1))
            h_list = torch.concat([h_list, tp], 1)

        tp = 2.0*ys*h_list*(v2-v1)
        tp += v1*v1*h_list*h_list
        tp = torch.sqrt(tp) - v1*h_list
        tp = torch.where(torch.abs(v1-v2)<1.0e-6, ys/v1, tp/(v2-v1))
        tp += xs

        x = torch.where(cover > 0, tp, y)

        tp = 2.0 * ys * h_list * (v2 - v1)
        tp += v1 * v1 * h_list * h_list
        tp = h_list/torch.sqrt(tp)

        dlogdet = torch.where(cover > 0, tp, torch.ones_like(cover))
        dlogdet = torch.sum(torch.log(dlogdet), dim=[1])

        return x, dlogdet
```


Classes and methods explaination:
The `__init__` method
This method initializes the CDF_quadratic class.

Parameters:
n_bins: The number of intervals used for a discrete field or range.
input_dim: Dimensions of the input variable.
r: The grid proportion used to generate discrete domains.
bound: indicates the boundary of the domain.
Code description:
self.n_bins and self.input_dim: Store the number of intervals and input dimensions.
self.bound and self.r: Holds the boundaries and proportions of the domain.
A non-uniform grid symmetric to zero is generated, and the grid distance increases proportionally r as you move away from zero.
self.mesh and self.elmt_size: Registered as buffers, respectively, to hold the size of the mesh and each element.
self.p: A trainable parameter that represents the density function for each interval.

`forward` method
This method performs forward propagation and nonlinear transformation of the input tensor.

Parameters:

x: input tensor.
t: Condition variable (optional).
Code description:

Call the _pdf_normalize method to normalize the probability density function (PDF).
Re-scale the input tensor to the range [0,1].
Call the _cdf method for CDF mapping.
Re-scale the mapping result back to the original range.


`inverse` method
This method performs backward propagation, a reverse transformation of the input tensor to recover the original data.

Parameters:

z: Input tensor (after a forward transformation).
t: Condition variable (optional).
Code description:

Call the _pdf_normalize method to normalize the probability density function (PDF).
Re-scale the input tensor to the range [0,1].
The _cdf_inv method is called for CDF reverse mapping.
Re-scale the mapping result back to the original range.

`_pdf_normalize` method
The method normalizes the piecewise representation of the probability density function (PDF).

Code description:
Calculate the probability density of each interval and normalize it.
Calculate the cumulative density function (CDF) for each interval.

`_cdf` method
This method implements CDF mapping of piecewise quadratic functions.

Code description:
Based on the input tensor, the index of its interval is calculated.
According to the index, the corresponding probability density and cumulative density are calculated.
Calculate the CDF map result and the logarithmic absolute determinant.

`_cdf_inv` method
This method realizes the reverse mapping of CDF.

Code description:
Based on the input tensor, the index of its interval is calculated.
According to the index, the corresponding probability density and cumulative density are calculated.
Calculate the CDF reverse mapping result and the logarithmic absolute determinant.
