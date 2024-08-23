# 问题一

$$
X=(x_1, x_2) \quad Z\sim  N(\mu, \Sigma) \\
X \rightarrow f_{pinn} \rightarrow r(X,\Theta_1) \\
X \rightarrow f_{KRnet} \rightarrow Z
$$

DAS论文中直接将样本点坐标$X= (x,y)$当作输入映射到一组2元对角高斯分布Z，而不是将r(x,y) 作为输入。随着$X$ 传入可以得到一组$Z$，实际Z是一组多元高斯分布的样本，可以根据预先指定的高斯分布计算这组样本发生的似然概率。

$$
D_{\mathrm{KL}}(\hat{r}_{\boldsymbol{X}}(\boldsymbol{x})\|\hat{p}_{\text{KRnet}}(\boldsymbol{x};\Theta_{f}))=\int_{B}\hat{r}_{\boldsymbol{X}}\log\hat{r}_{\boldsymbol{X}}d\boldsymbol{x}-\int_{B}\hat{r}_{\boldsymbol{X}}\log\hat{p}_{\text{KRnet}}d\boldsymbol{x}.
$$

第一项与参数$\Theta$无关

$$
\hat{p}_{\text{KRnet}}(\boldsymbol{x};\Theta_f)=p_Z(f_{\text{KRnet}}(\boldsymbol{x}))\left|\det\nabla_xf_{\text{KRnet}}\right|,
$$

$$
H(\hat{r}_{\boldsymbol{X}},\hat{p}_{\text{KRnet}})=-\int_{B}\hat{r}_{\boldsymbol{X}}\log\hat{p}_{\text{KRnet}}d\boldsymbol{x}.
$$

$\hat{r}_{\boldsymbol{X}}$ 看作一个概率密度函数，客观存在但不知道具体函数方程。所以使用IS 方法采样。 这里已知$Z$的分布函数与概率密度函数具体值。所以

$$
H(\hat{r}_{\boldsymbol{X}},\hat{p}_{\text{KRnet}})=-\int_{B} p_Z(x) \dfrac{\hat{r}_{\boldsymbol{X}}}{p_Z(x)} \log\hat{p}_{\text{KRnet}}d\boldsymbol{x}.
$$

利用IS采样估计均值，先利用先验分布$Z$随机生成一组样本，再逆变换得到$X$, 代入下式得到

$$
H(\hat{r}_{\boldsymbol{X}},\hat{p}_{\text{KRnet}})\approx-\frac{1}{N_{r}}\sum_{i=1}^{N_{r}}\frac{\hat{r}_{\boldsymbol{X}}(\boldsymbol{x}_{B}^{(i)})}{\hat{p}_{\text{KRnet}}(\boldsymbol{x}_{B}^{(i)};\Theta_{f})}\mathrm{log}\hat{p}_{\text{KRnet}}(\boldsymbol{x}_{B}^{(i)};\Theta_{f}),
$$

如何证明$X\rightarrow Z$ 的神经网络是单射，否则两者的概率密度函数单调性不一致。

# 问题二

$$
\min_{\boldsymbol{\theta}}\max_{\int_{\Omega}p_{\boldsymbol{\alpha}}(\boldsymbol{x})d\boldsymbol{x}=1}^{p_{\boldsymbol{\alpha}}>0,}\mathcal{J}(u_{\boldsymbol{\theta}},p_{\boldsymbol{\alpha}})=\int_{\Omega}r^2(\boldsymbol{x};\boldsymbol{\theta})p_{\boldsymbol{\alpha}}(\boldsymbol{x})d\boldsymbol{x}-\beta\int_{\Omega}|\nabla_{\boldsymbol{x}}p_{\boldsymbol{\alpha}}(\boldsymbol{x})|^2d\boldsymbol{x},
$$

这个优化模型的参数$\beta$ 是超参数还是可训练参数，选多大合适？

$$
\mathcal{J}(u_{\boldsymbol{\theta}},p_{\boldsymbol{\alpha}})\approx\frac{1}{m}\sum_{i=1}^{m}\frac{r^{2}\left[u_{\boldsymbol{\theta}}(\boldsymbol{x}_{\boldsymbol{\alpha}^{(i)}})\right]p_{\boldsymbol{\alpha}}(x_{\boldsymbol{\alpha}^{\prime}}^{(i)})}{p_{\boldsymbol{\alpha}^{\prime}}(\boldsymbol{x}_{\boldsymbol{\alpha}^{\prime}}^{(i)})}-\beta\cdot\frac{1}{m}\sum_{i=1}^{m}\frac{|\nabla_{\boldsymbol{x}}p_{\boldsymbol{\alpha}}(\boldsymbol{x}_{\boldsymbol{\alpha}^{\prime}}^{(i)})|^{2}}{p_{\boldsymbol{\alpha}^{\prime}}(\boldsymbol{x}_{\boldsymbol{\alpha}^{\prime}}^{(i)})},
$$

这个损失函数中的$p_{\alpha}, p_{\alpha '} $ 分别指的是新的和旧的模型参数，$p_{\alpha}$未知。

# 问题三

AAS 模型中的生成器和判别器具体指的是KRnet和PINN 中的哪些部分？对抗的部分具体体现在哪两个组件？原本论文里面有一个WGAN里的w距离表示两种分布差异最大，但是最后的损失函数里面没有这部分了，是化简掉了么？

我的感觉是：生成器应该是KRnet这个可逆映射，先在先验分布下生成样本在逆映射回去得到X，再通过PINN得到残差的分布。
