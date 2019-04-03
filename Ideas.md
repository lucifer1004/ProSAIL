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

### Outline

- Introduction
  - Why is it important to denoise vegetation spectra?
    - Vegetation spectra are widely used.
    - Noise caused by measurement and water vapor absorption limits the applications of measured spectra.
  - Current denoising techniques and their limitations
    - Filters
    - PCA
    - Wavelets
  - What does this paper do?
- Material and Methods
  - Simulated data
    - Use PROSAIL to generate canopy spectra
    - Noise-adding strategies
      - Gaussian noise for all bands
      - Gaussian noise for water-vapor-absorption bands
  - Field data
    - Field spectra collected in Yucheng, Hengshui and Luohe
- Results
  - Relative error (Mean, Median, Max)
  - Green Maximum
  - Red Minimum
  - NIR Maximum
  - NIR Minimum
  - SWIR Maximum
- Discussions
  - Choice of network structure
  - Different training data
- Conclusions