# Ideas

## 1. Spectral super-resolution reconstruction of crop canopy spectra based on ProSAIL

基于ProSAIL模型的作物冠层光谱超分辨率重建

- 拟解决的问题：许多作物长势监测和估产模型依赖多光谱、高光谱信息，但实际卫星影像能够提供的波段往往不能满足这些模型的需要。
- 创新点：利用ProSAIL构建植被冠层光谱库，基于这一光谱库，提出了作物冠层光谱超分辨率重建模型，并应用于Sentinel-2和Landsat-8卫星影像中。
- 目前进展：属于之前论文中的工作。

## 2. Auto-adaptive retrieval of ProSAIL based on deep neural networks

基于深度神经网络的ProSAIL模型自适应反演

- 拟解决的问题：现有的ProSAIL反演方法往往依赖于特定的输入数据组织形式（特定传感器、特定已知条件），因而其应用具有局限性。

- 创新点：本文提出的方法可以处理含有任意缺失值的输入数据，充分利用各类已知信息（如土壤水分、日照和观测几何等），进行ProSAIL模型的反演。
- 目前进展：初步实验，还需要进一步优化网络结构。

## 3. Spectral super-resolution reconstruction of spaceborne remote sensing images

基于UNet的星载遥感影像光谱超分辨率重建

- 拟解决的问题：许多地表和植被参数监测与反演模型依赖多光谱、高光谱信息，但实际卫星影像能够提供的波段往往不能满足这些模型的需要。
- 创新点：利用了空间信息和光谱信息中的内在联系，实现了卫星遥感影像的光谱超分辨率重建。
- 目前进展：还未开始。

## 4. Spectral denoise of crop canopy spectra based on ProSAIL and DAE

基于ProSAIL模型和降噪自编码器的作物冠层光谱去噪

- 拟解决的问题：受大气条件和仪器性能影响，光谱仪测量的作物冠层光谱有时会包含比较强的噪声，特别是在水汽吸收波段，光谱仪的测量结果无法使用，通常都被直接剔除。
- 创新点：利用ProSAIL构建冠层光谱库并训练降噪自编码器，与现有的S-G滤波等方法相比，可以更加有效地对噪声进行去除，并可以有效恢复水汽吸收波段的信息。
- 目前进展：属于之前论文中的工作。



## References

## 从RGB图像估算光谱信息（Kaya et al. 1812.00805）

[论文链接](https://arxiv.org/pdf/1812.00805.pdf)

## 光谱超分辨率方法归纳

- 基于高分辨率高光谱影像建立稀疏字典，然后利用正交匹配（Orthogonal Matching Pursuit，OMP）来进行稀疏光谱重建。
- 基于Tiramisu网络（常用于语义分割）的变种。
- A+超分辨率算法。
- 利用已知光谱响应函数的数码相机来提升超分的效果。

## 本文方法

### 1. RGB图像构建（重采样）

利用高光谱图像和敏感性函数（实际上就是遥感中的光谱响应函数）得到RGB图像。

### 2. 利用RGB图像估算敏感性函数

利用一个全卷积网络（12卷积层+4池化层）进行敏感性函数的估算。
输出为一个$3\times d$的矩阵，表示R、G、B三个通道在$d$个高光谱通道处的敏感性（也即R、G、B三个通道的光谱响应函数）。
损失函数分为三项：

- 第一项为重构误差$L_i=\frac{1}{n}||HS-H\hat{S}||^2_F$，表示原始高光谱影像重采样得到的RGB与利用预测得到的敏感性函数进行重采样得到的RGB之间的[Frobenius距离](http://mathworld.wolfram.com/FrobeniusNorm.html)，其表达式为$||A||_F=\sqrt{\sum^{m}_{i=1}\sum^{n}_{j=1}|a_{ij}|^2}=\sqrt{\mathrm{Tr}(AA^H)}$。
- 第二项为敏感性函数的预测误差$L_l=||S-\hat{S}||^2_F$。
- 第三项为正则项，通过对敏感性函数曲线求二阶导的方式来计算$L_s=||TS||^2_F$，这里的$T$为二阶导算子。
- 总的损失函数为三项的加权值：$L=\lambda_iL_i+\lambda_lL_l+\lambda_sL_s$

### 3. 基于RGB图像对敏感性函数分类

利用不同的敏感性函数分别生成不同的RGB图像，然后训练一个全卷积+softmax分类器网络来对这些RGB图像进行分类。

### 4. 高光谱图像重构

根据输入的不同，分为三种网络：
- 通用网络，输入为RGB图像，适用于敏感性函数未知的情形
- 条件网络，输入为RGB图像+敏感性函数
- 专用网络，输入为具有特定一种或几种敏感性函数的RGB图像

三种情形的网络结构基本一致，为带残差块的UNet结构。

## 结果评价

使用了ICVL、CAVE、NUS、NTIRE四个高光谱数据集，覆盖400到700nm之间的31个波段。
在光谱响应函数方面，分为连续谱和离散谱两类，连续谱基于高斯混合模型，在一定的限制条件下生成；离散谱基于现实中的光谱响应函数，重采样到31个波段，共40组。

评价指标：均方根误差RMSE、峰值信噪比PSNR、平均相对绝对误差MRAE、结构相似性SSIM。



## 利用GAN恢复光谱信息（Tran et al. 1812.04744）

[论文链接](https://arxiv.org/pdf/1812.04744.pdf)

目前的超宽带（Ultra Wide Band, UWB）雷达系统一般工作在100MHz至数GHz之间。其在光谱方面存在两大局限性，一是混淆，也即其他电磁波源的干扰；二是缺失，因为部分频带被禁止使用。本文作者训练了一个GAN来解决这两个问题。

![原文图1 SARGAN基本结构示意图](https://upload-images.jianshu.io/upload_images/2678339-0d990df803963009.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

训练数据：若干SAR影像对，每一影像对由一幅正常影像和对其进行频带干扰（混淆或去除部分频段）后得到的图像组成。

## 生成器损失函数构建

首先是表示重构误差的内容损失函数

$$l_{content}(G_{\theta_G}(Z),X)=||M\cdot FG_{\theta_G}(Z)-M\cdot FX||_1$$

其中$F$为Fourier矩阵，$M$为二值掩膜矩阵，用以表示哪些波段未受到干扰。

为了提高重构质量，再引入对抗损失函数

$$l_{adversarial}(G_{\theta_G}(Z))=-\log D_{\theta_D}(G_{\theta_G}(Z))$$

其中$D_{\theta_D}$表示判别器，则此时生成器总的损失函数为

$$\mathcal{L}(G_{\theta_G}(Z),X)=l_{content}+\lambda l_{adversarial}$$

其中$\lambda > 0$

## 判别器损失函数构建

判别器的目标是下面这一最优化问题

$$\max_{\theta_D}E_{X\sim p_{data}(X)}[\log D_{\theta_D}(X)]+E_{Z\sim p_G(Z)}[1-\log D_{\theta_D}(G_{\theta_G}(Z))]$$

也即，尽可能准确地判别哪些属于原始影像，哪些属于生成器重构的影像。



## Bendoumi et al., 2014 高光谱影像分辨率增强

【本文关键词】高光谱，多光谱，图像融合（image fusion），分辨率增强（resolution enhancement），混合像元分解（spectral unmixing）
【内容简介】将多光谱影像分成子图像分别进行端元提取（endmember extraction），进而提升高光谱分辨率增强的精度。
【原文链接】[网页](https://ieeexplore.ieee.org/abstract/document/6731532/)|[PDF](https://sci-hub.tw/https://ieeexplore.ieee.org/abstract/document/6731532/)

## 图像融合简史

本文的研究属于图像融合的范畴。在遥感领域，最早的图像融合技术是全色锐化（Panchromatic Sharpening, Pansharpening），也即利用全色波段的遥感影像提升多光谱波段遥感影像的空间分辨率。实现全色锐化的方法众多，主流的方法包括**主成分分析（Principal Component Analysis, PCA）**，**强度-色调-饱和度（Intensity-Hue-Saturation, IHS）变换**等。在PCA方法中，首先对多光谱波段进行PCA，然后将得到的第一个主成分用全色影像代替。在IHS方法中，则在对RGB波段进行IHS变换后，用全色影像代替I分量。此后的多光谱影像已不限于RGB三个波段，因此，在IHS方法的基础上，又发展出了**广义IHS（Generalized IHS）**方法和**广义强度调制（Generalized Intensity Modulation）**方法。后来又出现了，**高通滤波（High-pass Filtering）**和**高通调制（High-pass Modulation）**方法，基本思路是提取全色影像的高频分量，然后将其以适当的方式添加到每一个多光谱波段中。随着多分辨率技术的发展，**高斯金字塔（Gaussian Pyramids）**、**拉普拉斯金字塔（Laplacian Pyramids）**，以及**离散小波变换（Discrete Wavelet Transform, DWT）**也被应用于全色锐化中。

此后，全色锐化技术被进一步应用于多光谱和高光谱影像，在方法上取得了长足的进步和发展。首先出现的是基于**2-D和3-D小波**的方法，但它的性能取决于空间和光谱重采样效果的好坏。另一种方法是使用贝叶斯统计中的**最大后验（Maximum *a posteriori*, MAP）估计**来建立多光谱和高光谱影像之间的联系。在此基础上，再引入**随机混合模型（Stochastic Mixture Model, SMM）**作为约束条件，可以进一步提升融合效果。而如果在小波域中使用MAP估计，则可以取得很好的抗噪声性能。

**耦合非负矩阵分解（Coupled Nonnegative Matrix Factorization, CNMF）**是一种较为新颖的方法。该方法根据**线性混合模型（Linear mixing model, LMM）**对高光谱和多光谱影像分别利用NMF进行**光谱解混（Spectral Unmixing）**，在已知传感器观测模型的基础上，综合利用了高光谱影像中的端元和多光谱影像丰富的空间信息。CNMF在短波红外波段比MAP效果要好很多。

## 本文方法简介

本文的方法是，对高光谱和多光谱影像利用LMM进行解混，但使用的不是整景影像，而是影像中的子图，这样可以达到减小高光谱影像重构误差的效果。与CNMF类似，本文方法也需要已知高光谱和多光谱传感器的观测模型。在验证部分，作者将该方法的结果与CNMF和MAP-SMM进行了比较。

## 本文方法详解

### 第一步 观测建模

假设高光谱影像有$p$个波段，$m$个像元，可表示为$X_{p\times m}$；多光谱影像有$q$个波段，$n$个像元，表示为$Y_{q\times n}$。显然，有$p>q$和$n>m$。假设两幅影像已经经过几何配准，而待预测的高分辨率高光谱影像为$Z_{p\times n}$，则可得到下面两个式子：

$$X=ZW+E_s$$
$$Y=RZ+E_r$$

其中$W$是空间扩散变换矩阵（Spatial Spread Transform Matrix），$R$是光谱响应变换矩阵（Spectral Response Transfrom Matrix）。$W$为$n\times m$的稀疏矩阵，含有$(n/m)^2$个非零元，代表高光谱影像的空间**点扩散函数（Point Spread Function, PSF）**。$R$为$q\times p$的稀疏矩阵，每一行代表多光谱影像一个波段的光谱响应函数。$W$和$R$均为不可逆的非方阵，在本文方法中为已知量。$E_s$和$E_r$为噪声。

### 第二步 线性光谱混合模型

考虑待预测的高分辨率高光谱影像$Z$，它的LMM表示为：

$$Z=SA+N_z$$

其中$S$为$p\times d$的端元特征矩阵（$d$个端元，每个有$p$个波段的反射率），$A$为$d\times n$的丰度比例矩阵（$n$个像元中的每一个所对应的各个端元的比例），$N_z$为噪声。

将LMM代换到$X$和$Y$的表达式中，忽略噪声，可以得到：

$$X\simeq SA_h$$
$$Y\simeq S_mA$$

其中$A_h$是$d\times m$的丰度比例矩阵（$m$个像元中的每一个所对应的各个端元的比例），$S_m$是$q\times d$的单元特征矩阵（$d$个端元，每个有$q$个波段的反射率）。将上面几个式子中的噪声部分均忽略不计，可以得到：

$$A_h\simeq AW$$
$$S_m\simeq RS$$

### 第三步 基于光谱解混的图像融合

![算法图解](https://upload-images.jianshu.io/upload_images/2678339-922ec6245b9e61ac.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在$d=q$时，有$A=S_m^{-1}Y$；在$d<q$时，有$A=(S_m^TS_m)^{-1}S_m^TY$。为了最小化重构误差，选择提取$d=q$个端元。具体的算法流程如下：

1. 利用**顶成分分析（Vertex Component Analysis, VCA）**提取高光谱影像中的端元

$$S=VCA(X,q)$$

2. 对提取的端元进行光谱重采样

$$S_m=RS$$

3. 计算高分辨率多光谱影像中对应的丰度矩阵

$$A=S_m^{-1}Y$$

4. 对得到的丰度矩阵进行重采样

$$A_h=AW$$

5. 用最小二乘法估计高光谱影像的端元矩阵

$$S=LSE(X,A_h)=XA_h^T(A_hA_h^T)^{-1}$$

6. 重构高分辨率高光谱影像

$$Z=SA$$

> 最后得到的$Z$的表达式为
> $$Z=X(AW)^T((AW)(AW)^T)^{-1}(RS)^{-1}Y$$

### 第四步 子图处理

第三步中考虑的是$d=q$的情形，但有时$q$个端元不足以最小化重构误差，因此还必须处理$d>q$的情形。本文使用了分割子图的方法，分别对每一幅子图执行第三步中的融合算法，则最终可以得到$kq$个端元（$k$为子图数量）。由于分割的尺度对最后的重构误差影响很大，本文采用了多个分割尺度，最后将多个尺度的结果进行融合。具体的算法流程如下：

1. 计算每个分割尺度下，低分辨率高光谱影像每一个像元的重构误差

$$r_i=RMSE(X^i,X_r^i)$$

2. 找出每个像元取得最小重构误差时对应的分割尺度

$$r_{min}=min(r_{1i},r_{2i},...,r_{mi})$$
$$r_{min}={r_{1i_1},r_{2i_2},...,r_{mi_m}}$$

3. 找到在高光谱影像上对应的空间位置

4. 找到其在多光谱影像上对应的空间位置

5. 得到最终的高分辨率高光谱影像

在这一过程中，可能得到高度相关的端元，需要进行排除。这里可以利用**方差扩大因子（Variation Inflation Factor, VIF）**进行判断：

$$VIF_i=\frac{1}{1-R_i^2}$$

如果含有VIF超过10的端元，则对应的子图需要在融合过程中加以排除。

## 个人理解

本文方法的创新点在于，在多个尺度下分别进行融合算法，最后得到了多幅高分辨率高光谱影像，然后再重采样到高光谱影像原分辨率，计算每一个像元的重构误差，然后选择最小误差的结果。这样实际上是把得到的多幅高分辨率高光谱影像再进行筛选，合并得到了最终结果。简单来说也就是将多个结果组合为最终结果。这一思想可以进一步拓展，比如应用于热红外？应用于光谱而非像元？

## 附：图像融合质量的评价指标

图像融合效果的好坏很难通过目视判断，需要借助一些定量的指标。常用的指标包括：

**峰值信噪比（Peak Signal Noise Ratio, PSNR）**，表达式为：

$$PNSR_k=10\log_{10}\left(\frac{MAX_k^2}{MSE_k}\right)$$

其中$MSE_k$为第$k$个波段的均方误差，$MAX_k$为第$k$个波段的最大值。

**光谱角制图（Spectral Angle Mapper, SAM）**，表达式为：

$$SAM=\arccos\left(\frac{<z,z'>}{||z||_2||z'||_2}\right)$$

一般来说，PNSR越大，SAM越小，表示融合效果越好。

此外，还可以使用**结构相似度指数（Structural Similarity Index, SSIM Index）**来度量，其表达式为：

$$SSIM=\frac{(2\mu_Z\mu_{SA}+C_1)(2\sigma_Z\sigma_{SA}+C_2)}{(\mu_Z^2+\mu_{SA}^2+C_1)(\sigma_Z^2+\sigma_{SA}^2+C_2)}$$

其中$C_1$和$C_2$为常数项，可保证表达式有意义。