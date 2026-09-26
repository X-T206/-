# AlexNet

AlexNet 由 Krizhevsky 等人于 2012 年提出，在 ImageNet 竞赛中以巨大优势夺冠，标志着深度学习时代的开启。

### 与 LeNet 的关键区别

- 更深的网络（8 层：5 卷积 + 3 全连接）
- 使用 **ReLU** 替代 Sigmoid，训练更快
- 使用 **Dropout** 控制过拟合
- 使用最大池化（而非平均池化）
- 数据增强（翻转、裁剪、颜色变化）
- GPU 并行训练

### 网络结构

```
输入 (3×224×224)
  → Conv(96, 11×11, stride=4) → ReLU → MaxPool(3×3, stride=2)
  → Conv(256, 5×5, padding=2) → ReLU → MaxPool(3×3, stride=2)
  → Conv(384, 3×3, padding=1) → ReLU
  → Conv(384, 3×3, padding=1) → ReLU
  → Conv(256, 3×3, padding=1) → ReLU → MaxPool(3×3, stride=2)
  → Flatten → FC(4096) → ReLU → Dropout(0.5)
  → FC(4096) → ReLU → Dropout(0.5)
  → FC(1000)
```

### 1. 定义模型

```python
import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    # 这里使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以类别数为10
    nn.Linear(4096, 10))
```

### 2. 查看各层形状

```python
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
[out]:
Conv2d output shape:    torch.Size([1, 96, 54, 54])
ReLU output shape:      torch.Size([1, 96, 54, 54])
MaxPool2d output shape: torch.Size([1, 96, 26, 26])
Conv2d output shape:    torch.Size([1, 256, 26, 26])
ReLU output shape:      torch.Size([1, 256, 26, 26])
MaxPool2d output shape: torch.Size([1, 256, 12, 12])
Conv2d output shape:    torch.Size([1, 384, 12, 12])
ReLU output shape:      torch.Size([1, 384, 12, 12])
Conv2d output shape:    torch.Size([1, 384, 12, 12])
ReLU output shape:      torch.Size([1, 384, 12, 12])
Conv2d output shape:    torch.Size([1, 256, 12, 12])
ReLU output shape:      torch.Size([1, 256, 12, 12])
MaxPool2d output shape: torch.Size([1, 256, 5, 5])
Flatten output shape:  torch.Size([1, 6400])
Linear output shape:    torch.Size([1, 4096])
ReLU output shape:      torch.Size([1, 4096])
Dropout output shape:   torch.Size([1, 4096])
Linear output shape:    torch.Size([1, 4096])
ReLU output shape:      torch.Size([1, 4096])
Dropout output shape:   torch.Size([1, 4096])
Linear output shape:    torch.Size([1, 10])
```

### 3. 训练（Fashion-MNIST，图像 resize 到 224×224）

```python
batch_size = 128
# 将图像大小调整为224×224以匹配AlexNet输入
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
[out]:
training on cuda:0
loss 0.328, train acc 0.880, test acc 0.873
1247.3 examples/sec on cuda:0
```

### 小结

- AlexNet 证明了深度卷积网络在视觉任务上的强大能力
- ReLU、Dropout、MaxPooling、GPU 训练是其成功的关键
- 网络越深、参数越多，需要的数据和算力也越多
