# ResNet（残差网络）

ResNet 由何恺明等人于 2015 年提出，获得 ImageNet 冠军。核心创新是**残差连接（Residual Connection / Skip Connection）**，解决了深层网络的梯度消失和退化问题，使训练上百甚至上千层的网络成为可能。

### 核心思想：残差块

普通网络直接学习 `H(x)`，残差网络学习残差 `F(x) = H(x) - x`，即输出为 `F(x) + x`。

```
输入 x
  ├──────────────┐
  → Conv → BN → ReLU → Conv → BN
  └──────────────┘ (+)
  → ReLU
  → 输出
```

恒等映射 `x` 保证了即使残差为 0，网络也不会变差，从而可以安全地加深。

### 1. 定义残差块

```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```

#### 测试残差块

```python
# 不改变通道数和尺寸
blk = Residual(3, 3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
print(Y.shape)
[out]:
torch.Size([4, 3, 6, 6])

# 通道数翻倍，尺寸减半（使用1×1卷积匹配形状）
blk = Residual(3, 6, use_1x1conv=True, strides=2)
print(blk(X).shape)
[out]:
torch.Size([4, 6, 3, 3])
```

### 2. 定义 ResNet 模型

ResNet-18 结构：

```
输入 (1×224×224)
  → Conv(64, 7×7, stride=2, padding=3) → BN → ReLU
  → MaxPool(3×3, stride=2, padding=1)
  → 残差块 stage1 (64 通道) × 2
  → 残差块 stage2 (128 通道, 下采样) × 2
  → 残差块 stage3 (256 通道, 下采样) × 2
  → 残差块 stage4 (512 通道, 下采样) × 2
  → Global AvgPool → FC(10)
```

```python
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(), nn.Linear(512, 10))
```

### 3. 查看各层形状

```python
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
[out]:
Sequential output shape: torch.Size([1, 64, 56, 56])
Sequential output shape: torch.Size([1, 64, 56, 56])
Sequential output shape: torch.Size([1, 128, 28, 28])
Sequential output shape: torch.Size([1, 256, 14, 14])
Sequential output shape: torch.Size([1, 512, 7, 7])
AdaptiveAvgPool2d output shape: torch.Size([1, 512, 1, 1])
Flatten output shape: torch.Size([1, 512])
Linear output shape:  torch.Size([1, 10])
```

### 4. 训练

```python
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
[out]:
training on cuda:0
loss 0.019, train acc 0.995, test acc 0.918
3124.5 examples/sec on cuda:0
```

### 小结

- 残差连接让梯度可以直接回传到浅层，解决了深度网络退化问题
- BatchNorm 加速训练并起到一定正则化作用
- ResNet 是现代深度学习的基础架构之一，思想被广泛应用于 CV、NLP 等领域
- 后续变体：ResNeXt、DenseNet、Wide ResNet 等
