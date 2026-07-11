# CGDR 方法部分

## Correlation-Guided Deformable Registration

异构遥感图像变化检测同时受到跨传感器模态差异、双时相局部错位、SAR 斑点噪声以及真实地物变化的影响。尤其在光学与 SAR 跨模态场景中，两类传感器的成像机理存在根本差异，直接进行像素级差异建模容易将配准误差和散射噪声误判为变化。为此，本文在 AEF-GLCD 框架中引入相关系数引导的可变形配准模块，即 Correlation-Guided Deformable Registration, CGDR。该模块以局部相关系数为数学约束，以 SAR 局部散射响应为物理先验，在变化检测前对待配准图像进行自适应形变对齐。

与传统全局配准或无约束光流估计不同，CGDR 并不对整幅图像施加同等强度的形变，而是根据局部相关性将空间区域划分为高置信匹配区域与低置信匹配区域。对于高相关区域，CGDR 采用硬门控执行细粒度可变形配准；对于低相关区域，仅保留受抑制的粗粒度位移，以避免 SAR 斑点噪声、弱纹理区域或真实变化区域被强制对齐。该设计能够在抑制伪变化响应的同时，尽可能保持真实变化信息。

设参考图像和待配准图像分别为：

$$
\mathbf{I}_f,\mathbf{I}_m \in \mathbb{R}^{B\times C\times H\times W},
$$

其中，$\mathbf{I}_f$ 表示 fixed image，$\mathbf{I}_m$ 表示 moving image，$B$ 为批大小，$C$ 为图像通道数，$H$ 和 $W$ 分别为空间高度与宽度。CGDR 的目标是估计二维可变形位移场：

$$
\mathbf{u}=(u_x,u_y)\in \mathbb{R}^{B\times 2\times H\times W},
$$

其中，$u_x$ 和 $u_y$ 分别表示水平方向和垂直方向的像素位移。给定位移场 $\mathbf{u}$，通过可微双线性采样算子 $\mathcal{W}(\cdot,\cdot)$ 得到配准后的图像：

$$
\mathbf{I}_a=\mathcal{W}(\mathbf{I}_m,\mathbf{u}),
$$

其中，$\mathbf{I}_a$ 表示对齐到参考图像空间后的 moving image。

### Speckle Suppression and Feature Extraction

由于 SAR 图像通常包含乘性斑点噪声，直接计算局部相关性会导致不稳定匹配。为降低噪声对配准估计的影响，CGDR 首先采用 Lee-style 局部统计滤波。对任意输入图像 $\mathbf{I}$，在以像素位置 $\mathbf{p}=(x,y)$ 为中心的局部窗口 $\Omega_k(\mathbf{p})$ 内计算局部均值和局部方差：

$$
\mu(\mathbf{p})=\frac{1}{|\Omega_k|}\sum_{\mathbf{q}\in\Omega_k(\mathbf{p})}\mathbf{I}(\mathbf{q}),
$$

$$
\sigma^2(\mathbf{p})=\frac{1}{|\Omega_k|}\sum_{\mathbf{q}\in\Omega_k(\mathbf{p})}\mathbf{I}^2(\mathbf{q})-\mu^2(\mathbf{p}),
$$

其中，$\Omega_k(\mathbf{p})$ 表示大小为 $k\times k$ 的局部邻域，$\mathbf{q}$ 为邻域内像素位置，$|\Omega_k|$ 为窗口内像素数量。全局噪声方差估计为：

$$
\nu=\frac{1}{HW}\sum_{\mathbf{p}}\sigma^2(\mathbf{p}).
$$

Lee 滤波权重定义为：

$$
\alpha_L(\mathbf{p})=\frac{\sigma^2(\mathbf{p})}{\sigma^2(\mathbf{p})+\nu+\varepsilon},
$$

其中，$\varepsilon$ 为防止除零的数值稳定项。滤波后的图像表示为：

$$
\widetilde{\mathbf{I}}(\mathbf{p})=\mu(\mathbf{p})+\alpha_L(\mathbf{p})\left(\mathbf{I}(\mathbf{p})-\mu(\mathbf{p})\right).
$$

该滤波形式在均匀区域增强平滑，在结构边缘区域保留局部细节。随后，CGDR 使用共享特征编码器 $\phi(\cdot)$ 提取参考图像与待配准图像的结构特征：

$$
\mathbf{F}_f=\phi(\widetilde{\mathbf{I}}_f),\qquad
\mathbf{F}_m=\phi(\widetilde{\mathbf{I}}_m),
$$

其中，$\mathbf{F}_f,\mathbf{F}_m\in\mathbb{R}^{B\times C_\phi\times H\times W}$，$C_\phi$ 为特征通道数。

### Local Correlation Estimation

为度量两幅图像在局部邻域内的结构一致性，CGDR 计算归一化局部相关系数。对特征 $\mathbf{F}_1$ 和 $\mathbf{F}_2$，首先计算局部去均值特征：

$$
\bar{\mathbf{F}}_i(\mathbf{q})=\mathbf{F}_i(\mathbf{q})-\frac{1}{|\Omega_r|}\sum_{\mathbf{t}\in\Omega_r(\mathbf{q})}\mathbf{F}_i(\mathbf{t}),\qquad i\in\{1,2\},
$$

其中，$\Omega_r$ 为相关性计算窗口。局部相关系数定义为：

$$
\mathcal{C}(\mathbf{F}_1,\mathbf{F}_2)(\mathbf{p})
=
\frac{
\frac{1}{C_\phi|\Omega_r|}
\sum_{c=1}^{C_\phi}\sum_{\mathbf{q}\in\Omega_r(\mathbf{p})}
\bar{\mathbf{F}}_{1,c}(\mathbf{q})\bar{\mathbf{F}}_{2,c}(\mathbf{q})
}{
\sqrt{
\left(
\frac{1}{C_\phi|\Omega_r|}
\sum_{c=1}^{C_\phi}\sum_{\mathbf{q}\in\Omega_r(\mathbf{p})}
\bar{\mathbf{F}}_{1,c}^2(\mathbf{q})
\right)
\left(
\frac{1}{C_\phi|\Omega_r|}
\sum_{c=1}^{C_\phi}\sum_{\mathbf{q}\in\Omega_r(\mathbf{p})}
\bar{\mathbf{F}}_{2,c}^2(\mathbf{q})
\right)
+\varepsilon
}
}.
$$

在实现中，$\mathcal{C}(\cdot,\cdot)$ 被裁剪到 $[-1,1]$。相关系数越高，表示局部区域越可能具有可靠的结构对应关系；相关系数越低，则说明该区域可能存在跨模态不可比性、真实变化、弱纹理或 SAR 斑点干扰。

### Coarse-to-Fine Deformable Field Estimation

CGDR 采用粗到细策略估计位移场。首先，在低分辨率空间估计粗位移。令 $P_s(\cdot)$ 表示尺度因子为 $s$ 的平均池化操作，则低分辨率特征为：

$$
\mathbf{F}_f^{\downarrow}=P_s(\mathbf{F}_f),\qquad
\mathbf{F}_m^{\downarrow}=P_s(\mathbf{F}_m).
$$

低分辨率相关图为：

$$
\mathbf{C}_c=\mathcal{C}(\mathbf{F}_f^{\downarrow},\mathbf{F}_m^{\downarrow}).
$$

粗配准预测头的输入构造为：

$$
\mathbf{Z}_c=
\left[
\mathbf{F}_f^{\downarrow},
\mathbf{F}_m^{\downarrow},
\left|\mathbf{F}_f^{\downarrow}-\mathbf{F}_m^{\downarrow}\right|,
\mathbf{C}_c
\right],
$$

其中，$[\cdot]$ 表示沿通道维度拼接。粗位移场由卷积预测头 $h_c(\cdot)$ 输出：

$$
\mathbf{u}_c
=
s\cdot
\operatorname{Up}
\left(
\frac{M_c}{s}\tanh\left(h_c(\mathbf{Z}_c)\right)
\right),
$$

其中，$M_c$ 为最大粗位移幅度，$\operatorname{Up}(\cdot)$ 为双线性上采样操作，$\tanh(\cdot)$ 用于限制位移范围。基于粗位移场，待配准特征被初步对齐：

$$
\mathbf{F}_m^c=\mathcal{W}(\mathbf{F}_m,\mathbf{u}_c).
$$

在粗配准之后，CGDR 重新计算细粒度相关图：

$$
\mathbf{C}_f=\mathcal{C}(\mathbf{F}_f,\mathbf{F}_m^c).
$$

同时，为识别可能由真实变化或错误配准导致的不可靠区域，CGDR 计算粗对齐后的残差图：

$$
\mathbf{R}
=
\frac{1}{C}
\sum_{c=1}^{C}
\left|
\widetilde{\mathbf{I}}_{f,c}
-
\mathcal{W}(\widetilde{\mathbf{I}}_{m,c},\mathbf{u}_c)
\right|.
$$

残差图被归一化为：

$$
\widehat{\mathbf{R}}
=
\frac{\mathbf{R}-\min(\mathbf{R})}{\max(\mathbf{R})-\min(\mathbf{R})+\varepsilon}.
$$

残差可靠度定义为：

$$
\boldsymbol{\rho}_R=\exp(-\tau_R\widehat{\mathbf{R}}),
$$

其中，$\tau_R$ 为残差置信温度。残差越大，$\boldsymbol{\rho}_R$ 越小，表示该区域越不适合进行强形变配准。

### Correlation- and Scatter-Guided Gating

CGDR 支持两种区域置信分数。对于一般跨模态配准，使用细粒度相关图作为门控分数：

$$
\mathbf{S}=\mathbf{C}_f.
$$

对于 SAR 主导分支，进一步引入基于物理散射响应的区域分数。首先将 SAR 强度归一化为：

$$
\mathbf{A}=\operatorname{clip}\left(\frac{\operatorname{mean}_c(\widetilde{\mathbf{I}}_f)+1}{2},0,1\right),
$$

其中，$\operatorname{mean}_c(\cdot)$ 表示通道平均，$\operatorname{clip}(\cdot)$ 表示范围裁剪。SAR 散射分数定义为：

$$
\mathbf{S}_{\mathrm{sar}}
=
0.55\mathbf{A}
+0.25\mathcal{N}(\nabla\mathbf{A})
+0.20\mathcal{N}
\left(
\frac{\operatorname{Std}(\mathbf{A})}{\operatorname{Mean}(\mathbf{A})+\varepsilon}
\right),
$$

其中，$\nabla\mathbf{A}$ 表示强度梯度，$\operatorname{Std}(\cdot)$ 和 $\operatorname{Mean}(\cdot)$ 分别表示局部标准差和局部均值，$\mathcal{N}(\cdot)$ 表示 min-max 归一化。最终门控分数为：

$$
\mathbf{S}
=
\begin{cases}
\mathbf{C}_f, & \text{correlation mode},\\
\mathbf{S}_{\mathrm{sar}}, & \text{SAR scatter mode}.
\end{cases}
$$

为保证高置信区域比例稳定，CGDR 使用自适应阈值。设 $\tau_0$ 为基础阈值，$\lambda_a$ 为自适应混合系数，$r_t$ 为目标高置信比例，$Q_q(\cdot)$ 为第 $q$ 分位数，则：

$$
\tau
=
(1-\lambda_a)\tau_0+\lambda_a Q_{1-r_t}(\mathbf{S}).
$$

进一步地，为避免高置信区域过少或过多，阈值被限制为：

$$
\tau
\leftarrow
\operatorname{clip}
\left(
\tau,
Q_{1-r_{\max}}(\mathbf{S}),
Q_{1-r_{\min}}(\mathbf{S})
\right),
$$

其中，$r_{\min}$ 和 $r_{\max}$ 分别表示允许的最小和最大高置信区域比例。

基于阈值 $\tau$，软门控与硬门控分别定义为：

$$
\mathbf{G}_s=\sigma\left(\gamma(\mathbf{S}-\tau)\right),
$$

$$
\mathbf{G}_h=\mathbb{I}(\mathbf{S}\ge\tau),
$$

其中，$\sigma(\cdot)$ 为 Sigmoid 函数，$\gamma$ 为门控温度，$\mathbb{I}(\cdot)$ 为指示函数。为了同时保留硬选择能力和可导优化性质，CGDR 采用 straight-through gate：

$$
\mathbf{G}=\mathbf{G}_h+\mathbf{G}_s-\operatorname{sg}(\mathbf{G}_s),
$$

其中，$\operatorname{sg}(\cdot)$ 表示停止梯度操作。进一步结合残差可靠度对高残差区域进行抑制：

$$
\mathbf{G}
\leftarrow
\operatorname{clip}
\left(
\mathbf{G}\left[1-\lambda_R(1-\boldsymbol{\rho}_R)\right],
0,1
\right),
$$

其中，$\lambda_R$ 表示残差抑制强度。

### Fine Residual Flow and Confidence Blending

细粒度残差位移的输入由参考特征、粗对齐 moving 特征、特征差异、细粒度相关图和门控图共同构成：

$$
\mathbf{Z}_f
=
\left[
\mathbf{F}_f,
\mathbf{F}_m^c,
\left|\mathbf{F}_f-\mathbf{F}_m^c\right|,
\mathbf{C}_f,
\mathbf{G}
\right].
$$

细位移场为：

$$
\mathbf{u}_f=M_f\tanh\left(h_f(\mathbf{Z}_f)\right),
$$

其中，$h_f(\cdot)$ 为细位移预测头，$M_f$ 为最大细位移幅度。最终位移场定义为：

$$
\mathbf{u}
=
\mathbf{u}_c\left[\mathbf{G}+\eta(1-\mathbf{G})\right]
+\mathbf{G}\mathbf{u}_f,
$$

其中，$\eta\in[0,1]$ 为低置信区域粗位移缩放因子。当 $\mathbf{G}$ 较大时，模型执行粗配准加细粒度残差修正；当 $\mathbf{G}$ 较小时，细位移被抑制，仅保留缩放后的粗位移。该机制使 CGDR 能够在相关区域执行刚性细对齐，在低相关区域进行非相关性粗对齐，从而降低斑点噪声和真实变化对配准的干扰。

为了避免配准结果在不可靠区域完全覆盖原始图像，CGDR 使用置信混合机制生成最终对齐图像。置信融合权重为：

$$
\mathbf{B}
=
\beta_{\min}
+(1-\beta_{\min})
\left[
\operatorname{clip}
\left(
\frac{\mathbf{S}-\beta_0}{1-\beta_0},
0,1
\right)
\right]^2,
$$

其中，$\mathbf{B}$ 为配准融合置信图，$\beta_{\min}$ 为最小融合权重，$\beta_0$ 为置信起始阈值。最终配准图像为：

$$
\mathbf{I}_a
=
\mathbf{B}\mathcal{W}(\mathbf{I}_m,\mathbf{u})
+(1-\mathbf{B})\mathbf{I}_m.
$$

若启用变化保持机制，则根据归一化残差定义变化疑似图：

$$
\boldsymbol{\chi}
=
\sigma\left(\tau_\chi(\widehat{\mathbf{R}}-\delta_\chi)\right),
$$

其中，$\boldsymbol{\chi}$ 表示变化疑似区域，$\delta_\chi$ 为变化残差阈值，$\tau_\chi$ 为变化保持温度。融合权重进一步更新为：

$$
\mathbf{B}
\leftarrow
\mathbf{B}(1-\lambda_\chi\boldsymbol{\chi}),
$$

其中，$\lambda_\chi$ 为变化保持强度。该操作能够降低高残差变化疑似区域的配准强度，避免真实变化被错误消除。

### Registration Regularization

训练阶段，CGDR 采用多项正则约束稳定形变学习。总体配准损失定义为：

$$
\mathcal{L}_{\mathrm{CGDR}}
=
0.32\mathcal{L}_{\mathrm{inv}}
+0.23\mathcal{L}_{\mathrm{sm}}
+0.20\mathcal{L}_{\mathrm{sem}}
+0.15\mathcal{L}_{\mathrm{corr}}
+0.10\mathcal{L}_{\mathrm{low}}.
$$

其中，$\mathcal{L}_{\mathrm{inv}}$ 为正反向一致性损失，$\mathcal{L}_{\mathrm{sm}}$ 为边缘感知平滑损失，$\mathcal{L}_{\mathrm{sem}}$ 为语义一致性损失，$\mathcal{L}_{\mathrm{corr}}$ 为相关性一致性损失，$\mathcal{L}_{\mathrm{low}}$ 为低频一致性损失。

正反向一致性约束定义为：

$$
\mathcal{L}_{\mathrm{inv}}
=
\left\|
\mathbf{u}
+\mathcal{W}(\mathbf{u}^{-1},\mathbf{u})
\right\|_1,
$$

其中，$\mathbf{u}^{-1}$ 表示从参考图像到待配准图像的反向位移场。边缘感知平滑损失定义为：

$$
\mathcal{L}_{\mathrm{sm}}
=
\left\|
\partial_x\mathbf{u}\exp(-|\partial_x\mathbf{I}_f|)
\right\|_1
+
\left\|
\partial_y\mathbf{u}\exp(-|\partial_y\mathbf{I}_f|)
\right\|_1,
$$

其中，$\partial_x$ 和 $\partial_y$ 分别表示水平方向和垂直方向的梯度算子。语义一致性损失为：

$$
\mathcal{L}_{\mathrm{sem}}
=
\left\|
\psi(\widetilde{\mathbf{I}}_f)
-
\psi(\widetilde{\mathbf{I}}_a)
\right\|_1,
$$

其中，$\psi(\cdot)$ 表示语义映射网络。相关性一致性损失为：

$$
\mathcal{L}_{\mathrm{corr}}
=
\left\|
1-
\mathcal{C}
\left(
\mathbf{F}_f,
\phi(\widetilde{\mathbf{I}}_a)
\right)
\right\|_1.
$$

低频一致性损失为：

$$
\mathcal{L}_{\mathrm{low}}
=
\left\|
P_l(\widetilde{\mathbf{I}}_f)
-
P_l(\widetilde{\mathbf{I}}_a)
\right\|_1,
$$

其中，$P_l(\cdot)$ 表示低通平均池化算子。若训练数据提供未变化区域掩膜 $\mathbf{M}$，上述损失均可写作带掩膜形式：

$$
\mathcal{L}_{\mathrm{mask}}(\mathbf{X},\mathbf{Y};\mathbf{M})
=
\frac{
\sum_{\mathbf{p}}\mathbf{M}(\mathbf{p})\left|\mathbf{X}(\mathbf{p})-\mathbf{Y}(\mathbf{p})\right|
}{
\sum_{\mathbf{p}}\mathbf{M}(\mathbf{p})+\varepsilon
}.
$$

其中，$\mathbf{M}(\mathbf{p})=1$ 表示未变化区域，$\mathbf{M}(\mathbf{p})=0$ 表示变化区域或无效区域。通过仅在未变化区域施加配准约束，CGDR 能够避免将真实变化错误地作为配准误差进行优化。

### Integration with AEF-GLCD

在 AEF-GLCD 中，AlphaEarth 地理空间嵌入被用于提供全局生成先验，使扩散模型能够在地理坐标条件下生成与真实 SAR 场景物理一致的跨模态对齐特征。该过程主要缓解光学-SAR 之间的全局模态差异和语义漂移。CGDR 则进一步作用于局部空间层面，通过相关系数、散射响应、残差可靠度和变化保持门控共同约束可变形配准过程。

因此，AEF-GLCD 的整体优化可以理解为全局语义一致性与局部几何一致性的联合建模。AlphaEarth 嵌入提供跨模态生成先验，CGDR 提供局部配准先验，两者共同减少由模态差异、时相错位和 SAR 斑点噪声引起的伪变化响应。最终，配准后的图像 $\mathbf{I}_a$ 被送入后续变化检测网络进行判别，从而提升异构遥感图像变化检测的鲁棒性和精度。

## Symbol Definitions

| Symbol | Meaning |
| --- | --- |
| $\mathbf{I}_f$ | 参考图像，fixed image |
| $\mathbf{I}_m$ | 待配准图像，moving image |
| $\mathbf{I}_a$ | 配准后的图像 |
| $B$ | 批大小 |
| $C$ | 图像通道数 |
| $H,W$ | 图像高度和宽度 |
| $\mathbf{u}$ | 最终二维位移场 |
| $u_x,u_y$ | 水平和垂直方向位移 |
| $\mathbf{u}_c$ | 粗位移场 |
| $\mathbf{u}_f$ | 细粒度残差位移场 |
| $\mathcal{W}(\cdot,\cdot)$ | 可微 warping 算子 |
| $\Omega_k$ | Lee 滤波局部窗口 |
| $\Omega_r$ | 局部相关性计算窗口 |
| $\mu(\mathbf{p})$ | 局部均值 |
| $\sigma^2(\mathbf{p})$ | 局部方差 |
| $\nu$ | 噪声方差估计 |
| $\alpha_L$ | Lee 滤波权重 |
| $\varepsilon$ | 数值稳定项 |
| $\phi(\cdot)$ | 特征编码器 |
| $\psi(\cdot)$ | 语义映射网络 |
| $\mathbf{F}_f,\mathbf{F}_m$ | 参考图像和待配准图像的特征 |
| $C_\phi$ | 特征通道数 |
| $\mathcal{C}(\cdot,\cdot)$ | 局部相关系数函数 |
| $\mathbf{C}_c$ | 粗尺度相关图 |
| $\mathbf{C}_f$ | 细尺度相关图 |
| $P_s(\cdot)$ | 尺度为 $s$ 的平均池化 |
| $P_l(\cdot)$ | 低通平均池化 |
| $h_c(\cdot)$ | 粗位移预测头 |
| $h_f(\cdot)$ | 细位移预测头 |
| $M_c$ | 最大粗位移幅度 |
| $M_f$ | 最大细位移幅度 |
| $\mathbf{R}$ | 粗配准残差图 |
| $\widehat{\mathbf{R}}$ | 归一化残差图 |
| $\boldsymbol{\rho}_R$ | 残差可靠度 |
| $\tau_R$ | 残差置信温度 |
| $\mathbf{S}$ | 区域门控分数 |
| $\mathbf{S}_{\mathrm{sar}}$ | SAR 散射区域分数 |
| $\mathbf{A}$ | 归一化 SAR 强度图 |
| $\mathcal{N}(\cdot)$ | min-max 归一化 |
| $\tau_0$ | 基础门控阈值 |
| $\tau$ | 自适应门控阈值 |
| $\lambda_a$ | 自适应阈值混合系数 |
| $r_t$ | 目标高置信区域比例 |
| $r_{\min},r_{\max}$ | 最小和最大高置信区域比例 |
| $Q_q(\cdot)$ | 第 $q$ 分位数 |
| $\mathbf{G}_s$ | 软门控 |
| $\mathbf{G}_h$ | 硬门控 |
| $\mathbf{G}$ | straight-through 高置信门控 |
| $\gamma$ | 门控温度 |
| $\operatorname{sg}(\cdot)$ | 停止梯度操作 |
| $\lambda_R$ | 残差抑制强度 |
| $\eta$ | 低置信区域粗位移缩放因子 |
| $\mathbf{B}$ | 配准融合置信图 |
| $\beta_{\min}$ | 最小融合权重 |
| $\beta_0$ | 融合置信起始阈值 |
| $\boldsymbol{\chi}$ | 变化疑似区域门控 |
| $\delta_\chi$ | 变化残差阈值 |
| $\tau_\chi$ | 变化保持温度 |
| $\lambda_\chi$ | 变化保持强度 |
| $\mathbf{M}$ | 未变化区域或有效区域掩膜 |
| $\mathcal{L}_{\mathrm{CGDR}}$ | CGDR 总配准损失 |
| $\mathcal{L}_{\mathrm{inv}}$ | 正反向一致性损失 |
| $\mathcal{L}_{\mathrm{sm}}$ | 边缘感知平滑损失 |
| $\mathcal{L}_{\mathrm{sem}}$ | 语义一致性损失 |
| $\mathcal{L}_{\mathrm{corr}}$ | 相关性一致性损失 |
| $\mathcal{L}_{\mathrm{low}}$ | 低频一致性损失 |
