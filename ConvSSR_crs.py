# -*- coding: utf-8 -*-
"""
@author: Zhang Yucong(zhangyucong20@mails.ucas.ac.cn)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


    
# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)   
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out

# 改进后的网络配置
class Config():
    def __init__(self, in_ch):
        self.in_ch = in_ch
        self.encoder = [
            ('conv', 'relu', in_ch, 64, 3, 1, 1),       # 初始卷积
            ('resblock', '', 64, 128, '', '', 2),       # 残差块，步幅为2，下采样
            ('resblock', '', 128, 256, '', '', 2),      # 再次下采样
            ('resblock', '', 256, 512, '', '', 2),     # 再次下采样
            ('avgpool', '', 512, '', 3, '', ''),      # 全局平均池化
            ('linear', '', 512, 1, '', '', '')         # 全连接层
        ]
        

#基础卷积模块：用于k和x的特征提取
class ResConvBranch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = []
        for idx, params in enumerate(config.encoder):
            setattr(self, params[0] + '_' + str(idx), self._make_layer(*params))
            self.layers.append(params[0] + '_' + str(idx))

    def _make_layer(self, type, activation, in_ch, out_ch, kernel_size, padding, stride):
        layers = []
        if type == 'conv':
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))   
        elif type == 'resblock':
            layers.append(ResidualBlock(in_ch, out_ch, stride=2))  # 默认步幅为2
        elif type == 'avgpool':
            #layers.append(nn.AvgPool2d(kernel_size=kernel_size))  # 全局平均池化
            layers.append(nn.AdaptiveAvgPool2d(1))   
        elif type == 'linear':
            layers.append(nn.Flatten())
            layers.append(nn.Dropout(p=0.1))
            layers.append(nn.Linear(in_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = getattr(self, layer)(x)
        return x


#主网络
class ConvSS2t1(nn.Module):
    def __init__(self,config):
        super(ConvSS2t1, self).__init__()
        self.in_ch = config.in_ch
        
        self.resconv_layer = ResConvBranch(config)  # x 处理分支

    def forward(self, x_input):
        x = self.resconv_layer(x_input)  # 计算 f(x)
        return x
    

# 测试代码
if __name__ == "__main__":
    from thop import profile
    
    # 创建配置
    inch=19
    config = Config(inch)  
    model = ConvSS2t1(config)

    # 打印网络结构
    #print(model)
    x_input = torch.randn(8, inch, 21, 21)  # (B, C, H, W)
    x_input = torch.clamp(x_input,min=0.0)
    flops, params = profile(model, inputs=(x_input,))
    print("FLOPs:", flops / 1e9, " 1e9, params:",params / 1e6, " 1e6, memory:", params*8/1024/1024, "Mb")

    # 测试输入
    output = model(x_input)
    print("Output shape:", output.shape)  # 应为 (8, 1)