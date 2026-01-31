# JSF-SRNet: Joint Spatial and Frequency Domain Learning for Efficient Single Image Super-Resolution

## 项目简介

JSF-SRNet (Joint Spatial and Frequency Domain Learning for Super-Resolution Network) 是一种高效单图像超分辨率网络，它结合了空间域和频域信息处理的优势，旨在提高图像超分辨率的质量和效率。

该项目基于PyTorch实现，采用了先进的深度学习技术，包括空间注意力机制、频域滤波、双分支架构等，能够有效提升低分辨率图像的质量，生成高质量的高分辨率图像。

## 主要特性

- **联合时空域学习**：同时利用空间域和频域信息进行图像重建
- **高效架构**：采用BSConvU卷积和SDB（Spatial Distilled Block）模块减少计算复杂度
- **多频段处理**：使用高通、中通、低通滤波器分别处理不同频率成分
- **注意力机制**：集成CCALayer（Channel and Contrast Attention）和SKA（Scalable Kernel Attention）增强特征表达能力
- **灵活上采样**：支持多种上采样策略（PixelShuffle、NearestConv等）

## 网络架构

JSF-SRNet采用双分支架构：

### 空间域分支
- 使用多个SDB（Spatial Distilled Block）模块提取空间特征
- SDB模块融合了特征蒸馏和注意力机制
- 每个SDB包含多个卷积层和CCALayer注意力模块

### 频域分支
- 通过FFT变换将输入图像转换到频域
- 使用FDB（Frequency Domain Block）处理不同频率成分：
  - 低通滤波（Low-pass）
  - 中通滤波（Mid-pass） 
  - 高通滤波（High-pass）
- 处理后的频域特征通过IFFT转换回空间域

### 特征融合
- 将空间域和频域特征进行拼接融合
- 通过全局特征聚合模块整合多层次信息
- 最终通过上采样模块生成高分辨率图像

## 技术组件

### 核心模块
- **BSConvU**：双向卷积单元，平衡性能和效率
- **SDB**：空间蒸馏块，提取关键空间特征
- **FDB**：频域块，处理不同频率成分
- **SKA**：可扩展核注意力机制
- **CCALayer**：通道和对比度注意力层

### 损失函数
- Transformer感知损失
- 多尺度损失
- L1损失

## 文件结构

```
JSF-SRNet/
├── src/
│   ├── JSFSRNet_arch.py      # 网络架构定义
│   ├── MSID_arch-reference.py # 参考架构
│   └── Upsamplers.py         # 上采样模块
├── main.py                   # 主程序入口
├── option.py                 # 参数配置
├── train.py                  # 训练和测试逻辑
├── utils.py                  # 工具函数
└── README.md                 # 项目说明文档
```

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- CUDA (推荐)
- 其他依赖包（见requirements）

## 安装指南

1. 克隆项目：
```bash
git clone <repository-url>
cd JSF-SRNet
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 安装额外依赖（如果需要）：
```bash
pip install torch torchvision
pip install opencv-python scikit-image numpy pillow
pip install tensorboard thop tqdm
```

## 使用方法

### 训练模型

```bash
python main.py --train=train --data_train DF2K --data_test Set5 --scale 4 --batch_size 24 --n_epochs 700 --lr 1e-3
```

### 测试模型

```bash
python main.py --train=test --data_test Set5 --scale 4 --model_path models/JSFSRNet_X4
```

### 主要参数说明

| 参数 | 说明 |
|------|------|
| `--scale` | 超分倍数（默认4） |
| `--batch_size` | 批次大小（默认24） |
| `--n_epochs` | 训练轮数（默认700） |
| `--lr` | 学习率（默认1e-3） |
| `--patch_size` | 训练样本空间分辨率（默认48） |
| `--data_train` | 训练数据集（默认DF2K） |
| `--data_test` | 测试数据集（默认Set5） |

## 性能指标

模型在多个基准数据集上进行了评估，主要指标包括：
- **PSNR**：峰值信噪比
- **SSIM**：结构相似性指数
- **SAM**：光谱角映射
- **VIF**：视觉保真度
- **BRI**：图像质量评分

## 模型特点

1. **高效性**：通过BSConvU和特征蒸馏技术降低计算复杂度
2. **鲁棒性**：频域处理增强了对噪声和模糊的鲁棒性
3. **高质量重建**：联合时空域学习保留更多细节信息
4. **可扩展性**：支持不同的放大倍数和分辨率

## 训练流程

1. 数据预处理：加载HR/LR图像对
2. 前向传播：通过网络生成超分图像
3. 损失计算：计算感知损失、多尺度损失和L1损失
4. 反向传播：更新网络参数
5. 模型验证：在测试集上评估性能

## 参考文献

该项目受到以下研究工作的启发：
- 深度卷积神经网络在图像超分辨率中的应用
- 注意力机制在计算机视觉任务中的改进
- 频域处理技术在图像恢复中的有效性

## 许可证

请参阅项目中的LICENSE文件获取详细许可信息。

## 致谢

感谢所有为此项目做出贡献的研究人员和开发者。