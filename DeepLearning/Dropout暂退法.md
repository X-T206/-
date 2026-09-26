# Dropout（暂退法）

Dropout 是另一种常用的正则化技术：训练时以概率 `p` 随机将一些神经元输出置零，测试时不丢弃但需按比例缩放，从而防止网络过度依赖某些神经元。

### 1. 从零开始实现

```python
import torch
from torch import nn
from d2l import torch as d2l

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)
```

#### 测试 Dropout 层

```python
X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))
[out]:
tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11., 12., 13., 14., 15.]])
tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11., 12., 13., 14., 15.]])
tensor([[ 0.,  0.,  4.,  0.,  0., 10.,  0., 14.],
        [ 0., 18.,  0., 22.,  0.,  0., 28.,  0.]])
tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.]])
```

### 2. 定义模型

两个隐藏层，每层 256 个单元，分别使用 dropout 概率 0.2 和 0.5。

```python
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out

net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
```

### 3. 训练

```python
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
[out]:
epoch 1, loss 1.1023, train acc 0.5732, test acc 0.7123
epoch 2, loss 0.5821, train acc 0.7834, test acc 0.7892
epoch 3, loss 0.4823, train acc 0.8231, test acc 0.8213
epoch 4, loss 0.4312, train acc 0.8423, test acc 0.8342
epoch 5, loss 0.4023, train acc 0.8534, test acc 0.8431
epoch 6, loss 0.3812, train acc 0.8612, test acc 0.8512
epoch 7, loss 0.3643, train acc 0.8678, test acc 0.8563
epoch 8, loss 0.3512, train acc 0.8723, test acc 0.8612
epoch 9, loss 0.3401, train acc 0.8767, test acc 0.8643
epoch 10, loss 0.3312, train acc 0.8801, test acc 0.8672
```

### 4. 简洁实现

```python
net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

### 小结

- Dropout 训练时随机丢弃神经元，测试时保留全部
- 丢弃后需除以 `(1-p)` 保证期望值不变
- 通常在隐藏层后使用，概率 0.1~0.5
- PyTorch 中 `nn.Dropout(p)` 会自动在 eval 模式下关闭
