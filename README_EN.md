# JSF-SRNet: Joint Spatial and Frequency Domain Learning for Efficient Single Image Super-Resolution

## Project Overview

JSF-SRNet (Joint Spatial and Frequency Domain Learning for Super-Resolution Network) is an efficient single-image super-resolution network that combines the strengths of spatial-domain and frequency-domain information processing to improve both the quality and efficiency of image super-resolution.

The project is implemented in PyTorch and incorporates advanced deep-learning techniques, including spatial attention mechanisms, frequency-domain filtering, and a dual-branch architecture. It can effectively enhance low-resolution images and generate high-quality high-resolution outputs.

## Key Features

- **Joint Spatial and Frequency Domain Learning**: Utilizes both spatial-domain and frequency-domain information for image reconstruction
- **Efficient Architecture**: Employs BSConvU convolution and SDB (Spatial Distilled Block) modules to reduce computational complexity
- **Multi-Band Processing**: Uses high-pass, mid-pass, and low-pass filters to process different frequency components separately
- **Attention Mechanisms**: Integrates CCALayer (Channel and Contrast Attention) and SKA (Scalable Kernel Attention) to enhance feature representation
- **Flexible Upsampling**: Supports multiple upsampling strategies, including PixelShuffle and NearestConv

## Network Architecture

JSF-SRNet adopts a dual-branch architecture:

### Spatial-Domain Branch

- Uses multiple SDB (Spatial Distilled Block) modules to extract spatial features
- Each SDB combines feature distillation with attention mechanisms
- Each SDB contains multiple convolutional layers and CCALayer attention modules

### Frequency-Domain Branch

- Transforms the input image into the frequency domain using FFT
- Uses FDB (Frequency Domain Block) modules to process different frequency components:
  - Low-pass filtering
  - Mid-pass filtering
  - High-pass filtering
- Converts the processed frequency-domain features back to the spatial domain using IFFT

### Feature Fusion

- Concatenates and fuses spatial-domain and frequency-domain features
- Integrates multi-level information through a global feature aggregation module
- Generates the final high-resolution image through the upsampling module

## Technical Components

### Core Modules

- **BSConvU**: Bidirectional convolution unit that balances performance and efficiency
- **SDB**: Spatial Distilled Block for extracting key spatial features
- **FDB**: Frequency Domain Block for processing different frequency components
- **SKA**: Scalable Kernel Attention mechanism
- **CCALayer**: Channel and Contrast Attention layer

### Loss Functions

- Transformer perceptual loss
- Multi-scale loss
- L1 loss

## Project Structure

```text
JSF-SRNet/
├── src/
│   ├── JSFSRNet_arch.py       # Network architecture definition
│   ├── MSID_arch-reference.py # Reference architecture
│   └── Upsamplers.py          # Upsampling modules
├── main.py                    # Main entry point
├── option.py                  # Parameter configuration
├── train.py                   # Training and testing logic
├── utils.py                   # Utility functions
└── README.md                  # Project documentation
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- CUDA (recommended)
- Other dependencies (see `requirements`)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd JSF-SRNet
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Install additional dependencies if needed:

```bash
pip install torch torchvision
pip install opencv-python scikit-image numpy pillow
pip install tensorboard thop tqdm
```

## Usage

### Training

```bash
python main.py --train=train --data_train DF2K --data_test Set5 --scale 4 --batch_size 24 --n_epochs 700 --lr 1e-3
```

### Testing

```bash
python main.py --train=test --data_test Set5 --scale 4 --model_path models/JSFSRNet_X4
```

### Main Parameters

| Parameter | Description |
|-----------|-------------|
| `--scale` | Super-resolution scale factor (default: 4) |
| `--batch_size` | Batch size (default: 24) |
| `--n_epochs` | Number of training epochs (default: 700) |
| `--lr` | Learning rate (default: 1e-3) |
| `--patch_size` | Spatial resolution of training patches (default: 48) |
| `--data_train` | Training dataset (default: DF2K) |
| `--data_test` | Testing dataset (default: Set5) |

## Evaluation Metrics

The model has been evaluated on multiple benchmark datasets using the following metrics:

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **SAM**: Spectral Angle Mapper
- **VIF**: Visual Information Fidelity
- **BRI**: Image quality score

## Model Characteristics

1. **Efficiency**: Reduces computational complexity through BSConvU and feature distillation techniques
2. **Robustness**: Frequency-domain processing improves robustness to noise and blur
3. **High-Quality Reconstruction**: Joint spatial and frequency domain learning preserves more image details
4. **Scalability**: Supports different upscaling factors and image resolutions

## Training Pipeline

1. **Data Preprocessing**: Load paired HR/LR images
2. **Forward Propagation**: Generate super-resolved images through the network
3. **Loss Calculation**: Compute perceptual loss, multi-scale loss, and L1 loss
4. **Backpropagation**: Update network parameters
5. **Model Validation**: Evaluate performance on the test set

## References

This project is inspired by research in the following areas:

- Applications of deep convolutional neural networks to image super-resolution
- Improvements to computer vision tasks using attention mechanisms
- Effectiveness of frequency-domain processing techniques in image restoration

## License

Please refer to the `LICENSE` file in this project for detailed licensing information.

## Acknowledgements

We thank all researchers and developers who have contributed to this project.
