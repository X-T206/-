# VGG 网络

VGG 由 Simonyan 和 Zisserman 于 2014 年提出，核心思想是使用**多个 3×3 的小卷积核堆叠**替代大卷积核，在相同感受野下增加网络深度、减少参数。

### 核心设计：VGG 块

一个 VGG 块 = n 个 `Conv(3×3, padding=1)` + ReLU + 一个 `MaxPool(2×2, stride=2)`。

```
3×3 conv × n → ReLU → 2×2 MaxPool(stride=2)
```

### 1. 定义 VGG 块

```python
import torch
from torch import nn
from d2l import torch as d2l

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)
```

### 2. 定义 VGG 网络

VGG-11 的配置：每块卷积层数 `[1, 1, 2, 2, 2]`，通道数 `[64, 128, 256, 512, 512]`。

```python
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)
```

### 3. 查看各层形状

```python
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)
[out]:
Sequential output shape: torch.Size([1, 64, 112, 112])
Sequential output shape: torch.Size([1, 128, 56, 56])
Sequential output shape: torch.Size([1, 256, 28, 28])
Sequential output shape: torch.Size([1, 512, 14, 14])
Sequential output shape: torch.Size([1, 512, 7, 7])
Flatten output shape:  torch.Size([1, 25088])
Linear output shape:   torch.Size([1, 4096])
ReLU output shape:     torch.Size([1, 4096])
Dropout output shape:  torch.Size([1, 4096])
Linear output shape:   torch.Size([1, 4096])
ReLU output shape:     torch.Size([1, 4096])
Dropout output shape:  torch.Size([1, 4096])
Linear output shape:   torch.Size([1, 10])
```

### 4. 训练（使用缩小版通道数以节省显存）

```python
# 通道数缩小为原来的1/4，适合教学/小显存训练
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
[out]:
training on cuda:0
loss 0.178, train acc 0.934, test acc 0.917
2453.7 examples/sec on cuda:0
```

### 小结

- VGG 通过重复的 3×3 小卷积核堆叠构建深度网络
- 网络结构规整，易于理解和扩展
- VGG-16 / VGG-19 至今仍被广泛用作特征提取器
- 缺点：参数量巨大，全连接层占用大量内存
