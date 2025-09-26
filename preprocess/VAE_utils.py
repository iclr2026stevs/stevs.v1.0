import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import numpy as np
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.decomposition import PCA
import torch
import numpy as np
import harmonypy as hm
import scanpy as sc
import anndata as ad
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import numpy as np

from math import exp

class PIDControl():
    """incremental PID controller"""
    def __init__(self, Kp, Ki, init_beta, min_beta, max_beta):
        """define them out of loop"""
        self.W_k1 = init_beta
        self.W_min = min_beta
        self.W_max = max_beta
        self.e_k1 = 0.0
        self.Kp = Kp
        self.Ki = Ki

    def _Kp_fun(self, Err, scale=1):
        return 1.0/(1.0 + float(scale)*exp(Err))

    def pid(self, exp_KL, kl_loss):
        """
        Incremental PID algorithm
        Input: KL_loss
        return: weight for KL divergence, beta
        """
        error_k = (exp_KL - kl_loss) * 5.   # we enlarge the error 5 times to allow faster tuning of beta
        ## comput U as the control factor
        # print(f'error_k={error_k}, self.e_k1={self.e_k1}')
        dP = self.Kp * (self._Kp_fun(error_k) - self._Kp_fun(self.e_k1))
        dI = self.Ki * error_k

        if self.W_k1 < self.W_min:
            dI = 0
        dW = dP + dI
        ## update with previous W_k1
        Wk = dW + self.W_k1
        self.W_k1 = Wk
        self.e_k1 = error_k

        ## min and max value
        if Wk < self.W_min:
            Wk = self.W_min
        if Wk > self.W_max:
            Wk = self.W_max

        return Wk, error_k


# class ImageEncoder(nn.Module):
#     def __init__(self, input_channels, hidden_dims, output_dim, activation="elu", dropout=0.):
#         super(ImageEncoder, self).__init__()
#         # hidden_dims += [4]
#         self.conv_layers = self.build_conv_layers(input_channels, hidden_dims, activation, dropout)
#         self.flatten = nn.Flatten()
#         self.fc_mu = None  # 延迟初始化
#         self.fc_var = None  # 延迟初始化
#         self.output_dim = output_dim
#
#     def build_conv_layers(self, input_channels, hidden_dims, activation, dropout):
#         layers = []
#         in_channels = input_channels
#         for out_channels in hidden_dims:
#             layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
#             layers.append(nn.BatchNorm2d(out_channels))
#             if activation == "relu":
#                 layers.append(nn.ReLU())
#             elif activation == "elu":
#                 layers.append(nn.ELU())
#             layers.append(nn.Dropout(p=dropout))
#             in_channels = out_channels
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         # 通过卷积层
#         h = self.conv_layers(x)
#         h_flat = self.flatten(h)
#
#         # 动态初始化全连接层
#         if self.fc_mu is None or self.fc_var is None:
#             # 获取展平后的特征图大小
#             input_dim = h_flat.size(1)
#             self.fc_mu = nn.Linear(input_dim, self.output_dim).to(h_flat.device)  # 确保正确的设备
#             self.fc_var = nn.Linear(input_dim, self.output_dim).to(h_flat.device)
#
#         # 通过全连接层计算 mu 和 var
#         mu = self.fc_mu(h_flat)
#         logvar = self.fc_var(h_flat).clamp(-15, 15)
#         return mu, logvar
#
# # ImageDecoder 实现
# class ImageDecoder(nn.Module):
#     def __init__(self, input_dim, hidden_dims, output_channels, img_shape=(3, 64, 64), activation="elu", dropout=0.):
#         super(ImageDecoder, self).__init__()
#         self.img_shape = img_shape
#         self.fc = nn.Linear(input_dim, hidden_dims[0] * 8 * 8)  # 假设我们将解码器的初始特征图恢复为8x8
#         self.deconv_layers = self.build_deconv_layers(hidden_dims, output_channels, activation, dropout)
#
#         # 上采样到目标尺寸的模块
#         self.upsample = nn.Upsample(size=(img_shape[1], img_shape[2]), mode='bilinear', align_corners=False)
#
#         self.downsample = nn.AdaptiveAvgPool2d(output_size=(img_shape[1], img_shape[2]))
#
#     def build_deconv_layers(self, hidden_dims, output_channels, activation, dropout):
#         layers = []
#         in_channels = hidden_dims[0]
#         for out_channels in hidden_dims[1:]:
#             layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
#             layers.append(nn.BatchNorm2d(out_channels))
#             if activation == "relu":
#                 layers.append(nn.ReLU())
#             elif activation == "elu":
#                 layers.append(nn.ELU())
#             layers.append(nn.Dropout(p=dropout))
#             in_channels = out_channels
#         layers.append(nn.ConvTranspose2d(in_channels, output_channels, kernel_size=4, stride=2, padding=1))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.fc(x)
#         x = x.view(x.size(0), -1, 8, 8)  # 恢复到8x8的特征图
#         x = self.deconv_layers(x)
#         # 如果特征图尺寸大于目标尺寸，则下采样
#         if x.size(2) > self.img_shape[1] or x.size(3) > self.img_shape[2]:
#             x = self.downsample(x)
#         else:
#             x = self.upsample(x)
#         assert [i for i in x.shape[1:]] == self.img_shape
#         return torch.sigmoid(x)  # 输出图像值范围限制在 [0, 1]


import torch
from torch import nn
import math
import torch.nn.functional as F
from timm.layers import DropPath  # 您可能需要 pip install timm

def adata_img_sort_indices(adata_img):
    try:
        pixel_numbers = [int(name.split('_')[1]) for name in adata_img.var_names]
    except (ValueError, IndexError):
        raise ValueError("无法从 var_names 中解析出 'pixel_NUMBER' 格式。请检查您的变量名。")

    # 2. 获取能将这些数字从小到大排列的索引
    # np.argsort() 返回的是排序后的元素在原始数组中的索引
    sort_indices = np.argsort(pixel_numbers)

    # 3. 使用排序索引对 AnnData 对象进行切片
    # adata[:, sort_indices] 会创建一个新的 AnnData 对象，
    # 其 .var, .X, .layers 等所有列相关的数据都按新顺序排列。
    return adata_img[:, sort_indices]

# --- 1. 从新代码中提取的核心模块 ---
# ConvMLP 是 ConvBlock 的一个依赖项
class ConvMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(ConvMLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 这是我们将用来替换 ResidualBlock 的新砖块
class ConvBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super(ConvBlock, self).__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)  # 深度卷积模拟注意力
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x):
        # 注入位置信息
        x = x + self.pos_embed(x)
        # 核心的 "注意力" + 残差连接
        x = x + self.drop_path(
            self.conv2(self.attn(self.conv1(self.norm1(x))))
        )
        # MLP + 残差连接
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# --- 2. 构建使用新砖块的 ModernizedVAE ---
import torch
from torch import nn
import math
import torch.nn.functional as F


# 假设 ConvBlock 和 ConvMLP 类已定义
# from timm.layers import DropPath

# class ConvMLP(nn.Module):
#     ...
# class ConvBlock(nn.Module):
#     ...
#
class ResidualVAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128, img_size=32, hidden_dims=None):
        # ... (前面的代码不变) ...
        super(ResidualVAE, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        encoder_dims = [input_channels] + hidden_dims
        self.encoder_dims = encoder_dims  # <--- 1. 在这里新增一行，保存 encoder_dims
        decoder_dims = hidden_dims[::-1]

        # --- 编码器 (Encoder) ---
        # ... (编码器构建部分的代码不变) ...
        self.encoder_stages = nn.ModuleList()
        for i in range(len(encoder_dims) - 1):
            stage = nn.Sequential(
                ConvBlock(dim=encoder_dims[i], mlp_ratio=4.),
                nn.Conv2d(encoder_dims[i], encoder_dims[i + 1], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(encoder_dims[i + 1]),
                nn.LeakyReLU(inplace=True)
            )
            self.encoder_stages.append(stage)

        # ... (后续 __init__ 的代码不变) ...
        self.feature_map_size = img_size // (2 ** len(hidden_dims))
        self.fc_input_size = encoder_dims[-1] * (self.feature_map_size ** 2)
        self.fc_mu = nn.Linear(self.fc_input_size, latent_dim)
        self.fc_log_var = nn.Linear(self.fc_input_size, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, self.fc_input_size)
        self.decoder_stages = nn.ModuleList()
        for i in range(len(decoder_dims) - 1):
            stage = nn.Sequential(
                nn.ConvTranspose2d(decoder_dims[i], decoder_dims[i + 1], kernel_size=2, stride=2),
                nn.BatchNorm2d(decoder_dims[i + 1]),
                nn.LeakyReLU(inplace=True),
                ConvBlock(dim=decoder_dims[i + 1], mlp_ratio=4.)
            )
            self.decoder_stages.append(stage)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(decoder_dims[-1], input_channels, kernel_size=2, stride=2),
            nn.Tanh()
        )

    # ... (encode 和 reparameterize 方法不变) ...
    def encode(self, x):
        for stage in self.encoder_stages:
            x = stage(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x).clamp(-15, 15)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)

        # <--- 2. 修改这一行代码 ---
        deepest_dim = self.encoder_dims[-1]  # 直接从保存的列表中获取最深的维度
        x = x.view(-1, deepest_dim, self.feature_map_size, self.feature_map_size)

        for stage in self.decoder_stages:
            x = stage(x)
        reconstructed_x = self.final_layer(x)
        return reconstructed_x

    # ... (forward 方法不变) ...
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, log_var

#
# import torch
# from torch import nn
# import math
# import torch.nn.functional as F
#
#
# class ResidualVAE(nn.Module):
#     def __init__(self, input_channels=3, latent_dim=128, img_size=32, hidden_dims=None):
#         """
#         一个纯全连接层 (MLP) 的VAE版本，用于测试显存基线。
#         所有的卷积层都已被移除。
#         外部接口与之前的版本完全兼容。
#
#         Args:
#             input_channels (int): 输入图像的通道数。
#             latent_dim (int): 潜在空间的维度。
#             img_size (int): 输入图像的边长。
#             hidden_dims (list, optional):
#                 一个整数列表，用于定义MLP的隐藏层维度。
#                 如果为 None，则默认为 [512]。
#         """
#         super(ResidualVAE, self).__init__()
#         self.latent_dim = latent_dim
#         self.img_size = img_size
#         self.input_channels = input_channels
#
#         if hidden_dims is None:
#             hidden_dims = [512]  # 为MLP定义一个默认的隐藏层结构
#
#         # --- 1. 计算拉平后的输入维度 ---
#         flattened_size = input_channels * img_size * img_size
#
#         # --- 2. 构建编码器 (纯MLP) ---
#         encoder_layer_dims = [flattened_size] + hidden_dims
#
#         encoder_layers = []
#         for i in range(len(encoder_layer_dims) - 1):
#             encoder_layers.extend([
#                 nn.Linear(encoder_layer_dims[i], encoder_layer_dims[i + 1]),
#                 nn.LeakyReLU(inplace=True)
#             ])
#         self.encoder_mlp = nn.Sequential(*encoder_layers)
#
#         # 最终输出mu和log_var的层
#         self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
#         self.fc_log_var = nn.Linear(hidden_dims[-1], latent_dim)
#
#         # --- 3. 构建解码器 (纯MLP) ---
#         decoder_layer_dims = [latent_dim] + hidden_dims[::-1]
#
#         decoder_layers = []
#         for i in range(len(decoder_layer_dims) - 1):
#             decoder_layers.extend([
#                 nn.Linear(decoder_layer_dims[i], decoder_layer_dims[i + 1]),
#                 nn.LeakyReLU(inplace=True)
#             ])
#         # 最后一层将输出还原到拉平的图像维度
#         decoder_layers.append(nn.Linear(hidden_dims[0], flattened_size))
#         self.decoder_mlp = nn.Sequential(*decoder_layers)
#
#         # 最终的激活函数
#         self.final_activation = nn.Tanh()
#
#     def encode(self, x):
#         # 1. 将输入的图像 (B, C, H, W) 拉平为向量 (B, C*H*W)
#         x = torch.flatten(x, start_dim=1)
#         # 2. 通过MLP
#         x = self.encoder_mlp(x)
#         # 3. 计算 mu 和 log_var
#         mu = self.fc_mu(x)
#         log_var = self.fc_log_var(x).clamp(-15, 15)
#         return mu, log_var
#
#     def reparameterize(self, mu, log_var):
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     def decode(self, z):
#         # 1. 通过MLP，将潜在向量z重建为拉平的图像向量
#         x = self.decoder_mlp(z)
#         # 2. 将向量 reshape 回图像的形状 (B, C, H, W)
#         reconstructed_x = x.view(-1, self.input_channels, self.img_size, self.img_size)
#         # 3. 应用最终的激活函数
#         reconstructed_x = self.final_activation(reconstructed_x)
#         return reconstructed_x
#
#     def forward(self, x):
#         mu, log_var = self.encode(x)
#         z = self.reparameterize(mu, log_var)
#         reconstructed_x = self.decode(z)
#         return reconstructed_x, mu, log_var

class ImageEncoder(nn.Module):
    """
    An improved Image Encoder that uses strided convolutions for downsampling
    and adaptive pooling for a robust connection to the fully-connected layers.
    """
    def __init__(self, input_channels=3, latent_dim=8, img_size=32,devcie=None, dtype=None, **kwargs):
        super().__init__()
        self.model = ResidualVAE(input_channels=input_channels, latent_dim=latent_dim, img_size=img_size).to(dtype=dtype)
        self.to(devcie)
    def forward(self, x):
        mu, logvar = self.model.encode(x)
        return mu, logvar

    def get_vae_model(self):
        return self.model

class ImageDecoder(nn.Module):
    """
    An improved Image Decoder that mirrors the encoder's architecture, using
    Upsampling + Conv2d to avoid checkerboard artifacts. The final layer has no
    activation function, outputting raw pixel values (logits).
    """
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model

    def forward(self, rmu):
        reconstructed_x = self.model.decode(rmu)
        return reconstructed_x


import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Helper Modules for Cleaner Code ---
#
# class ConvBlock(nn.Module):
#     """A standard convolutional block: Conv -> BatchNorm -> Activation."""
#
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, activation="elu"):
#         super().__init__()
#         layers = [
#             nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
#             nn.BatchNorm2d(out_channels)
#         ]
#         if activation == "relu":
#             layers.append(nn.ReLU(inplace=True))
#         elif activation == "elu":
#             layers.append(nn.ELU(inplace=True))
#         self.block = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.block(x)
#
#
# class DeconvBlock(nn.Module):
#     """A modern deconvolutional block: Upsample -> Conv -> BatchNorm -> Activation."""
#
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, scale_factor=2, activation="elu"):
#         super().__init__()
#         layers = [
#             nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
#             nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
#             nn.BatchNorm2d(out_channels)
#         ]
#         if activation == "relu":
#             layers.append(nn.ReLU(inplace=True))
#         elif activation == "elu":
#             layers.append(nn.ELU(inplace=True))
#         self.block = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.block(x)
#
#
# # --- Modernized VAE Architecture ---
# class ImageEncoder(nn.Module):
#     """
#     An improved Image Encoder that uses strided convolutions for downsampling
#     and adaptive pooling for a robust connection to the fully-connected layers.
#     """
#
#     def __init__(self, input_channels, hidden_dims, output_dim, activation="elu"):
#         super().__init__()
#
#         # Build the convolutional backbone
#         layers = []
#         in_channels = input_channels
#         for h_dim in hidden_dims:
#             layers.append(ConvBlock(in_channels, h_dim, stride=2, activation=activation))
#             in_channels = h_dim
#         self.conv_layers = nn.Sequential(*layers)
#
#         # Adaptive pooling makes the model robust to input image size
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.flatten = nn.Flatten()
#
#         # Fully-connected layers are now initialized statically in __init__
#         self.fc_mu = nn.Linear(hidden_dims[-1], output_dim)
#         self.fc_var = nn.Linear(hidden_dims[-1], output_dim)
#         self.output_dim = output_dim
#
#     def forward(self, x):
#         # Pass through convolutional layers to extract features and downsample
#         h = self.conv_layers(x)
#
#         # Pool features to a fixed size and flatten
#         h = self.adaptive_pool(h)
#         h_flat = self.flatten(h)
#
#         # Calculate mu and logvar
#         mu = self.fc_mu(h_flat)
#         logvar = self.fc_var(h_flat).clamp(-15, 15)
#         return mu, logvar
#
#
# class ImageDecoder(nn.Module):
#     """
#     An improved Image Decoder that mirrors the encoder's architecture, using
#     Upsampling + Conv2d to avoid checkerboard artifacts. The final layer has no
#     activation function, outputting raw pixel values (logits).
#     """
#
#     def __init__(self, input_dim, hidden_dims, output_channels, activation="elu", start_resolution=4):
#         super().__init__()
#
#         # hidden_dims should be the reverse of the encoder's hidden_dims
#         self.hidden_dims = hidden_dims
#         self.start_res = start_resolution
#
#         # Project the latent vector `z` and reshape it into a starting feature map
#         self.fc = nn.Linear(input_dim, hidden_dims[0] * (self.start_res ** 2))
#
#         # Build the deconvolutional (upsampling) backbone
#         layers = []
#         in_channels = hidden_dims[0]
#         for h_dim in hidden_dims[1:]:
#             layers.append(DeconvBlock(in_channels, h_dim, activation=activation))
#             in_channels = h_dim
#         self.deconv_layers = nn.Sequential(*layers)
#
#         # Final convolutional layer to produce the output image
#         # This layer has no activation function.
#         self.final_conv = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             nn.Conv2d(hidden_dims[-1], output_channels, kernel_size=3, stride=1, padding=1),
#             # NO FINAL ACTIVATION (e.g., Sigmoid or Tanh)
#         )
#
#     def forward(self, x):
#         # Project and reshape latent vector
#         x = self.fc(x)
#         x = x.view(x.size(0), self.hidden_dims[0], self.start_res, self.start_res)
#         # Pass through upsampling layers
#         x = self.deconv_layers(x)
#         # Generate the final image
#         reconstructed_image = self.final_conv(x)
#         return F.sigmoid(reconstructed_image)

class DenseEncoder(nn.Module):
    def __init__(self,hidden_dims, output_dim, input_dim=None, activation="relu", dropout=0, dtype=torch.float32, norm="batchnorm"):
        super(DenseEncoder, self).__init__()
        if input_dim is not None:
            input_d = [input_dim] + hidden_dims
        else:
            input_d = hidden_dims
        self.layers = buildNetwork(input_d, network="decoder", activation=activation, dropout=dropout, dtype=dtype, norm=norm)
        self.enc_mu = nn.Linear(hidden_dims[-1], output_dim)
        self.enc_var = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        h = self.layers(x)
        mu = self.enc_mu(h)
        logvar = self.enc_var(h).clamp(-15, 15)
        return mu, logvar

import torch
from torch import nn


class Pos_GP(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(Pos_GP, self).__init__()
        output_dim = input_dim
        self.f1 = nn.Linear(input_dim, hidden_dims)
        self.enc_mu = nn.Linear(hidden_dims, output_dim)
        self.enc_var = nn.Linear(hidden_dims, output_dim)
        self.initialize_weights_to_zero()

    def initialize_weights_to_zero(self):
        """将f1和f2的权重和偏置都初始化为0"""
        print("Initializing Pos_GP residual layers to zero.")
        nn.init.zeros_(self.f1.weight)
        if self.f1.bias is not None:
            nn.init.zeros_(self.f1.bias)

        nn.init.zeros_(self.enc_mu.weight)
        if self.enc_mu.bias is not None:
            nn.init.zeros_(self.enc_mu.bias)

        nn.init.zeros_(self.enc_var.weight)
        if self.enc_var.bias is not None:
            nn.init.zeros_(self.enc_var.bias)

    def forward(self, x):
        h = F.softmax(self.f1(x))
        mu = self.enc_mu(h)
        logvar = self.enc_var(h).clamp(-15, 15)
        return mu, logvar


def buildNetwork(layers, network="decoder", activation="relu", dropout=0., dtype=torch.float32, norm="batchnorm"):
    net = []
    if network == "encoder" and dropout > 0:
        net.append(nn.Dropout(p=dropout))
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if norm == "batchnorm":
            net.append(nn.BatchNorm1d(layers[i]))
        elif norm == "layernorm":
            net.append(nn.LayerNorm(layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        elif activation=="elu":
            net.append(nn.ELU())
        elif activation == "softmax":
            net.append(nn.Softmax(dim=layers[i]))
        elif activation == "softplus":
            net.append(nn.Softplus())  # 添加 Softplus 激活函数
        if dropout > 0:
            net.append(nn.Dropout(p=dropout))
    return nn.Sequential(*net)


class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.exp(x).clamp(min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return F.softplus(x).clamp(min=1e-4, max=1e4)

'''
NBLoss 类实现了基于负二项分布的损失函数，
这种损失函数特别适用于具有超额离散性（方差大于均值）的计数数据。
这种损失函数在RNA测序数据分析等应用中常见，因为这些数据可能表现出高变异性。
'''
class NBLoss(nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, x, mean, disp, scale_factor=None):
        eps = 1e-10
        if scale_factor is not None:
            scale_factor = scale_factor[:, None]
            mean = mean * scale_factor

        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        log_nb = t1 + t2
#        result = torch.mean(torch.sum(result, dim=1))
        result = torch.sum(log_nb)
        return result

class MixtureNBLoss(nn.Module):
    def __init__(self):
        super(MixtureNBLoss, self).__init__()

    def forward(self, x, mean1, mean2, disp, pi_logits, scale_factor=None):
        eps = 1e-10
        if scale_factor is not None:
            scale_factor = scale_factor[:, None]
            mean1 = mean1 * scale_factor
            mean2 = mean2 * scale_factor

        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2_1 = (disp+x) * torch.log(1.0 + (mean1/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean1+eps)))
        log_nb_1 = t1 + t2_1

        t2_2 = (disp+x) * torch.log(1.0 + (mean2/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean2+eps)))
        log_nb_2 = t1 + t2_2

        logsumexp = torch.logsumexp(torch.stack((- log_nb_1, - log_nb_2 - pi_logits)), dim=0)
        softplus_pi = F.softplus(-pi_logits)

        log_mixture_nb = logsumexp - softplus_pi
        result = torch.sum(-log_mixture_nb)
        return result


class PoissonLoss(nn.Module):
    def __init__(self):
        super(PoissonLoss, self).__init__()

    def forward(self, x, mean, scale_factor=1.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        result = mean - x * torch.log(mean+eps) + torch.lgamma(x+eps)
        result = torch.sum(result)
        return result

'''
 gauss_cross_entropy 用于计算两个正态分布之间的交叉熵。在机器学习和统计模型中，
 交叉熵常用于度量两个概率分布之间的差异性，在变分推断和优化算法中评估模型的逼近质量.
'''
def gauss_cross_entropy(mu1, var1, mu2, var2):
    """
    Computes the element-wise cross entropy
    Given q(z) ~ N(z| mu1, var1)
    returns E_q[ log N(z| mu2, var2) ]
    args:
        mu1:  mean of expectation (batch, tmax, 2) tf variable
        var1: var  of expectation (batch, tmax, 2) tf variable
        mu2:  mean of integrand (batch, tmax, 2) tf variable
        var2: var of integrand (batch, tmax, 2) tf variable
    returns:
        cross_entropy: (batch, tmax, 2) tf variable
    """

    term0 = 1.8378770664093453  # log(2*pi)
    term1 = torch.log(var2)
    term2 = (var1 + mu1 ** 2 - 2 * mu1 * mu2 + mu2 ** 2) / var2

    cross_entropy = -0.5 * (term0 + term1 + term2)

    return cross_entropy


def gmm_fit(data: np.ndarray, mode_coeff=0.6, min_thres=0.3):
    """Returns delta estimate using GMM technique"""
    # Custom definition
    gmm = GaussianMixture(n_components=3)
    gmm.fit(data[:, None])
    vals = np.sort(gmm.means_.squeeze())
    res = mode_coeff * np.abs(vals[[0, -1]]).mean()
    res = np.maximum(min_thres, res)
    return res



from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.sparse import issparse
def PCC_trans_matrix(src_cor, tgt_cor, src_exp, tgt_exp, k_list=[]):
    """
    计算齐次变换矩阵，根据输入维度动态调整返回值。

    Args:
        src_cor (numpy.ndarray): 源点坐标数组 (N_src, D)，其中 N_src 是点数，D 是维度。
        tgt_cor (numpy.ndarray): 目标点坐标数组 (N_tgt, D)。
        src_exp (numpy.ndarray): 源点表达特征数组 (N_src, F)。
        tgt_exp (numpy.ndarray): 目标点表达特征数组 (N_tgt, F)。
        k_list (list[int]): 用于最近邻平滑的 k 值列表（可选）。

    Returns:
        numpy.ndarray: 齐次变换矩阵 ((D+1) x (D+1))。
    """
    num_dims = src_cor.shape[1]  # 自动检测维度
    if len(k_list) != 0:
        # 对源点数据进行处理
        knn_src_exp = src_exp.copy()
        kd_tree = cKDTree(src_cor)
        for k in k_list:
            distances, indices = kd_tree.query(src_cor, k=k)
            src_exp = src_exp + np.mean(knn_src_exp[indices], axis=1)

        # 对目标点数据进行处理
        knn_tgt_exp = tgt_exp.copy()
        kd_tree = cKDTree(tgt_cor)
        for k in k_list:
            distances, indices = kd_tree.query(tgt_cor, k=k)
            tgt_exp = tgt_exp + np.mean(knn_tgt_exp[indices], axis=1)

    # 计算相关性并匹配点
    corr = np.corrcoef(src_exp, tgt_exp)[:src_exp.shape[0], src_exp.shape[0]:]
    matched_src_cor = src_cor[np.argmax(corr, axis=0), :]

    # 计算平移和旋转矩阵
    mean_source = np.mean(matched_src_cor, axis=0)
    mean_target = np.mean(tgt_cor, axis=0)
    centered_source = matched_src_cor - mean_source
    centered_target = tgt_cor - mean_target
    rotation_matrix = np.dot(centered_source.T, centered_target)
    u, _, vt = np.linalg.svd(rotation_matrix)
    rotation = np.dot(vt.T, u.T)

    # 如果维度是3且检测到镜像翻转，则修正
    if num_dims == 3 and np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = np.dot(vt.T, u.T)

    translation = mean_target - mean_source

    # 构建齐次变换矩阵
    M = np.eye(num_dims + 1)
    M[:num_dims, :num_dims] = rotation
    M[:num_dims, -1] = translation
    M[-1, 0] = 1

    return M


def find_best_matching(src_cor, tgt_cor, src_exp, tgt_exp, k_list=[3, 10, 40]):
    """
    找到最佳匹配的点对，并基于相关性分析和动态分割法，提取可能的重叠区域点对。

    Args:
        src_cor (numpy.ndarray): 源点的空间坐标 (N_src, D)。
        tgt_cor (numpy.ndarray): 目标点的空间坐标 (N_tgt, D)。
        src_exp (numpy.ndarray): 源点的表达数据 (N_src, F)。
        tgt_exp (numpy.ndarray): 目标点的表达数据 (N_tgt, F)。
        k_list (list[int]): 最近邻平滑的 k 值列表。

    Returns:
        numpy.ndarray: 筛选后的源点坐标子集。
        numpy.ndarray: 筛选后的目标点坐标子集。
        pd.DataFrame: 匹配结果 DataFrame，包含匹配点索引及相关性。
    """
    # 处理源点数据
    knn_src_exp = src_exp.copy()
    if issparse(knn_src_exp):
        knn_src_exp = knn_src_exp.todense()
    kd_tree = cKDTree(src_cor)
    for k in k_list:
        distances, indices = kd_tree.query(src_cor, k=k)
        knn_src_exp += np.mean(knn_src_exp[indices, :], axis=1)

    # 处理目标点数据
    knn_tgt_exp = tgt_exp.copy()
    if issparse(knn_tgt_exp):
        knn_tgt_exp = knn_tgt_exp.todense()
    kd_tree = cKDTree(tgt_cor)
    for k in k_list:
        distances, indices = kd_tree.query(tgt_cor, k=k)
        knn_tgt_exp += np.mean(knn_tgt_exp[indices, :], axis=1)

    # 计算相关性矩阵
    corr = np.corrcoef(knn_src_exp, knn_tgt_exp)[:src_exp.shape[0], src_exp.shape[0]:]

    # 利用相关性排序和动态分割法寻找重叠点
    def detect_inflection_point(corr_vector):
        y = np.sort(np.max(corr_vector, axis=0))[::-1]
        data = np.array(y).reshape(-1, 1)
        algo = rpt.Dynp(model="l1").fit(data)
        result = algo.predict(n_bkps=1)
        return result[0]

    first_inflection_point_src = detect_inflection_point(corr)
    first_inflection_point_tgt = detect_inflection_point(corr.T)

    # 提取匹配点
    set1 = np.array([[index, value] for index, value in enumerate(np.argmax(corr, axis=0))])
    set1 = np.column_stack((set1, np.max(corr, axis=0)))
    set1 = pd.DataFrame(set1, columns=['tgt_index', 'src_index', 'corr'])
    set1.sort_values(by='corr', ascending=False, inplace=True)
    set1 = set1.iloc[:first_inflection_point_tgt, :]

    set2 = np.array([[index, value] for index, value in enumerate(np.argmax(corr, axis=1))])
    set2 = np.column_stack((set2, np.max(corr, axis=1)))
    set2 = pd.DataFrame(set2, columns=['src_index', 'tgt_index', 'corr'])
    set2.sort_values(by='corr', ascending=False, inplace=True)
    set2 = set2.iloc[:first_inflection_point_src, :]

    # 合并匹配点
    result = pd.merge(set1, set2, on=['tgt_index', 'src_index'], how='inner')
    matched_src_cor = src_cor[result['src_index'].to_numpy().astype(int), :]
    matched_tgt_cor = tgt_cor[result['tgt_index'].to_numpy().astype(int), :]

    return matched_src_cor, matched_tgt_cor, result


def calculate_transformation_matrix(src_cor, tgt_cor):
    """
    根据最佳匹配的点对计算齐次变换矩阵。

    Args:
        src_cor (numpy.ndarray): 源点的匹配子集坐标 (N, D)。
        tgt_cor (numpy.ndarray): 目标点的匹配子集坐标 (N, D)。

    Returns:
        numpy.ndarray: 齐次变换矩阵 ((D+1) x (D+1))。
    """
    num_dims = src_cor.shape[1]

    # 计算均值并进行去中心化
    mean_src = np.mean(src_cor, axis=0)
    mean_tgt = np.mean(tgt_cor, axis=0)
    centered_src = src_cor - mean_src
    centered_tgt = tgt_cor - mean_tgt

    # 计算旋转矩阵
    cov_matrix = np.dot(centered_src.T, centered_tgt)
    u, _, vt = np.linalg.svd(cov_matrix)
    rotation = np.dot(vt.T, u.T)

    # 防止 3D 情况下的镜像翻转
    if num_dims == 3 and np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = np.dot(vt.T, u.T)

    translation = mean_tgt - mean_src

    # 构建齐次变换矩阵
    M = np.eye(num_dims + 1)
    M[:num_dims, :num_dims] = rotation
    M[:num_dims, -1] = translation
    M[-1, 0] = 1

    return M

# 主流程
def match_and_transform(src_cor, tgt_cor, src_exp, tgt_exp, k_list=[], sample_match=True):
    """
    完成匹配点筛选和齐次变换矩阵的计算。

    Args:
        src_cor, tgt_cor, src_exp, tgt_exp: 数据集坐标和表达特征。
        k_list: 最近邻平滑参数。

    Returns:
        numpy.ndarray: 齐次变换矩阵。
    """
    if sample_match:
        transformation_matrix = PCC_trans_matrix(src_cor, tgt_cor, src_exp, tgt_exp, k_list)
    else:
        matched_src, matched_tgt, _ = find_best_matching(src_cor, tgt_cor, src_exp, tgt_exp, k_list)
        transformation_matrix = calculate_transformation_matrix(matched_src, matched_tgt)
    return transformation_matrix


def initialization_trans_matrix(src_cor, tgt_cor, src_exp, tgt_exp, method='PCC', sample_match=True):
    if method=='PCC':
        return match_and_transform(src_cor, tgt_cor, src_exp, tgt_exp,sample_match=sample_match)




from scipy.spatial import cKDTree
def PCC_trans_matrix(src_cor, tgt_cor, src_exp, tgt_exp, k_list = []):
    if len(k_list) != 0:
        # process source slice
        knn_src_exp = src_exp.copy()
        kd_tree = cKDTree(src_cor)
        for k in k_list:
            distances, indices = kd_tree.query(src_cor, k=k)  # (source_num_points, k)
            src_exp = src_exp + np.array(np.mean(knn_src_exp[indices, :], axis=1))

        # process target slice
        knn_tgt_exp = tgt_exp.copy()
        kd_tree = cKDTree(tgt_cor)
        for k in k_list:
            distances, indices = kd_tree.query(tgt_cor, k=k)  # (source_num_points, k)
            tgt_exp = tgt_exp + np.array(np.mean(knn_tgt_exp[indices, :], axis=1))

    corr = np.corrcoef(src_exp, tgt_exp)[:src_exp.shape[0],src_exp.shape[0]:]  # (src_points, tgt_points)
    matched_src_cor = src_cor[np.argmax(corr, axis=0), :]

    # Calculate transformation: translation and rotation
    mean_source = np.mean(matched_src_cor, axis=0)
    mean_target = np.mean(tgt_cor, axis=0)
    centered_source = matched_src_cor - mean_source
    centered_target = tgt_cor - mean_target
    rotation_matrix = np.dot(centered_source.T, centered_target)
    u, _, vt = np.linalg.svd(rotation_matrix)
    rotation = np.dot(vt.T, u.T)
    translation = mean_target - mean_source
    M = np.zeros((3, 3))
    M[:2, :2] = rotation
    M[:2, 2] = translation
    M[2, 2] = 1
    M[2, 0] =  1 # 缩放因子 临时存储

    return M


import umap
import numpy as np

def apply_umap_to_image(img, n_components=3):
    """
    对每个像素的100维通道数据进行UMAP降维，降到3维（RGB通道）。
    img: 输入的超图像，形状为(height, width, 100)
    n_components: UMAP降维后的维数，默认为3（RGB）
    """
    height, width, channels = img.shape
    umap_model = umap.UMAP(n_components=n_components)

    # 将每个像素的100维数据展平进行降维
    img_reshaped = img.reshape(-1, channels)  # 将 (height * width, 100) 展平
    img_reduced = umap_model.fit_transform(img_reshaped)  # 使用UMAP降到3维

    # 还原为图像形状 (height, width, 3)     # 归一化到 [0, 1]
    img_reduced = img_reduced.reshape(height, width, n_components)
    img_reduced_min = img_reduced.min()
    img_reduced_max = img_reduced.max()
    img_reduced = (img_reduced - img_reduced_min) / (img_reduced_max - img_reduced_min)

    img_reduced = (img_reduced * 255).astype(np.uint8)  # 映射到[0, 255]并转换为uint8类型
    return img_reduced

def convert_to_image(position, xdata,):
        """
        将 (x, y, R, G, B) 格式的数据转化为图像。

        data: DataFrame格式的数据，包含列 [x, y, R, G, B]
        返回值: 转化后的图像（NumPy数组）
        """
        # 假设图像的尺寸是最大x和最大y的坐标
        height = int(position[:,0].max())
        width = int(position[:, 1].max())
        beta = xdata.shape[0] / (height*width)

        # 初始化一个空的RGB图像，形状为 (height, width, 3)
        position = position * beta
        height = int(position[:, 0].max()) +1
        width = int(position[:, 1].max()) +1
        img = np.zeros((height, width, xdata.shape[-1]), dtype=np.uint8)
        # 将数据中的RGB值填入到对应的 (x, y) 坐标位置
        for index, rowdata in zip(position, xdata):
            img[int(index[0]), int(index[1])] = rowdata  # 注意y, x的顺序，符合图像矩阵的坐标系统
        return img

def feature_matching_SIGT(imgA, imgB):
    """
    对降维后的图像进行特征匹配，并计算旋转、缩放和平移变换。
    imgA, imgB: 输入的降维后的图像，形状为(height, width, 3)
    """
    # 转换为灰度图像，因为特征匹配通常在灰度图像上进行
    # 对图像A和B进行PCA降维到3维（RGB）
    grayA = cv2.cvtColor(imgA.astype(np.float32), cv2.COLOR_RGB2GRAY)
    grayB = cv2.cvtColor(imgB.astype(np.float32), cv2.COLOR_RGB2GRAY)

    grayA = np.uint8(grayA)
    grayB = np.uint8(grayB)
    # 使用SIFT检测特征点和描述符
    sift = cv2.SIFT_create()
    kpA, desA = sift.detectAndCompute(grayA, None)
    kpB, desB = sift.detectAndCompute(grayB, None)

    # 使用暴力匹配器进行匹配，并采用比率测试来过滤错误匹配
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = bf.knnMatch(desA, desB, k=2)

    # 使用比率测试（Ratio Test）来过滤匹配
    matches = []
    for m, n in raw_matches:
        if m.distance < 0.9 * n.distance:  # 0.75为SIFT的比率测试常见阈值
            matches.append(m)

    # 排序匹配
    matches = sorted(matches, key=lambda x: x.distance)

    # 提取匹配的关键点坐标
    ptsA = np.float32([kpA[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    ptsB = np.float32([kpB[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 计算仿射变换矩阵
    M, mask = cv2.estimateAffine2D(ptsA, ptsB)

    # 旋转矩阵、平移向量
    rotation_matrix = M[:2, :2]  # 旋转矩阵是仿射矩阵的前2x2部分
    translation_vector = M[:, 2]  # 平移向量是仿射矩阵的最后一列

    # 旋转角度
    a, b, c, d = M.flatten()[:4]  # 仿射矩阵中的旋转、缩放元素

    # 缩放因子
    scale = np.sqrt(a ** 2 + b ** 2)

    return rotation_matrix, translation_vector, scale


def SIGT_trans_matrix(src_cor, tgt_cor, src_exp, tgt_exp):
    imgA = convert_to_image(src_cor, src_exp)
    imgB = convert_to_image(tgt_cor, tgt_exp)
    imgA = apply_umap_to_image(imgA, n_components=3)
    imgB = apply_umap_to_image(imgB, n_components=3)
    rotation_matrix, translation_vector, rotation_angle, scale = feature_matching_SIGT(imgA, imgB)
    M = np.zeros((3, 3))
    M[:2, :2] = rotation_matrix
    M[:2, 2] = translation_vector
    M[2, 2] = 1
    M[2, 0] =  scale
    return M


def initialization_trans_matrix(src_cor, tgt_cor, src_exp, tgt_exp, method='PCC', **kwargs):
    if method=='PCC':
        return PCC_trans_matrix(src_cor, tgt_cor, src_exp, tgt_exp)
    elif method=='SIGT':
        return SIGT_trans_matrix(src_cor, tgt_cor, src_exp, tgt_exp)

    def voxel_data(
            coords: np.ndarray,
            gene_exp: np.ndarray,
            voxel_size: float = None,
            voxel_num: int = 10000,
    ):
        """
        Voxelization of the data.
        Parameters
        ----------
        coords: np.ndarray
            The coordinates of the data points. Shape (N, D)
        gene_exp: np.ndarray
            The gene expression of the data points. Shape (N, G)
        voxel_size: float
            The size of the voxel.
        voxel_num: int
            The number of voxels.
        Returns
        -------
        voxel_coords: np.ndarray
            The coordinates of the voxels.
        voxel_gene_exp: np.ndarray
            The gene expression of the voxels.
        """
        N, D = coords.shape

        # 确保输入是 numpy 数组
        coords = np.asarray(coords)
        gene_exp = np.asarray(gene_exp)

        # 创建体素网格
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)

        # 计算体素大小
        if voxel_size is None:
            voxel_size = np.sqrt(np.prod(max_coords - min_coords)) / (np.sqrt(N) / 5)

        # 计算每个维度的体素数量
        grid_size = int(np.sqrt(voxel_num))
        voxel_steps = (max_coords - min_coords) / grid_size

        # 生成体素网格坐标
        voxel_coords_list = [
            np.arange(min_coord, max_coord, voxel_step)
            for min_coord, max_coord, voxel_step in zip(min_coords, max_coords, voxel_steps)
        ]

        # 创建网格坐标
        voxel_coords = np.stack(np.meshgrid(*voxel_coords_list), axis=-1).reshape(-1, D)
        voxel_gene_exps = np.zeros((voxel_coords.shape[0], gene_exp.shape[1]))
        is_voxels = np.zeros(voxel_coords.shape[0], dtype=bool)

        # 将数据点分配到体素
        for i, voxel_coord in enumerate(voxel_coords):
            # 计算到当前体素的距离
            dists = np.sqrt(np.sum((coords - voxel_coord) ** 2, axis=1))
            # 找到距离小于体素半径的点
            mask = dists < voxel_size / 2
            if np.any(mask):
                # 计算这些点的平均基因表达
                voxel_gene_exps[i] = np.mean(gene_exp[mask], axis=0)
                is_voxels[i] = True

        # 只保留有数据的体素
        voxel_coords = voxel_coords[is_voxels]
        voxel_gene_exps = voxel_gene_exps[is_voxels]

        return voxel_coords, voxel_gene_exps

def normalize_coordinates(coordsA, coordsB, separate_mean=True, separate_scale=True, verbose=False):
    """
    使用PyTorch实现的空间坐标归一化函数

    参数:
    coordsA (torch.Tensor): 第一个样本的坐标矩阵，形状为 [N, D]
    coordsB (torch.Tensor): 第二个样本的坐标矩阵，形状为 [M, D]
    separate_mean (bool): 是否分别计算均值，默认为True
    separate_scale (bool): 是否分别计算缩放因子，默认为True
    verbose (bool): 是否打印归一化参数信息

    返回:
    tuple: 归一化后的坐标 (coordsA_norm, coordsB_norm), 缩放因子, 均值

    异常:
    AssertionError: 当两个坐标矩阵的维度不一致时抛出
    """
    # 确保输入是PyTorch张量
    if not isinstance(coordsA, torch.Tensor):
        coordsA = torch.tensor(coordsA)
    if not isinstance(coordsB, torch.Tensor):
        coordsB = torch.tensor(coordsB)

    # 检查维度是否一致
    assert coordsA.shape[1] == coordsB.shape[1], "两个坐标矩阵的维度必须一致"

    D = coordsA.shape[1]  # 坐标维度
    coords = [coordsA, coordsB]
    normalize_scales = torch.zeros(2, dtype=coordsA.dtype, device=coordsA.device)
    normalize_means = torch.zeros(2, D, dtype=coordsA.dtype, device=coordsA.device)

    # 计算每个坐标矩阵的均值
    for i in range(len(coords)):
        normalize_mean = torch.mean(coords[i], dim=0)
        normalize_means[i] = normalize_mean

    # 如果不分别计算均值，则使用全局均值
    if not separate_mean:
        global_mean = torch.mean(normalize_means, dim=0)
        normalize_means = global_mean.expand(2, -1)

    # 中心化坐标并计算缩放因子
    for i in range(len(coords)):
        coords[i] -= normalize_means[i]
        squared_sum = torch.sum(coords[i] * coords[i]) / coords[i].shape[0]
        normalize_scale = torch.sqrt(squared_sum)
        normalize_scales[i] = normalize_scale

    # 如果不分别计算缩放因子，则使用全局缩放因子
    if not separate_scale:
        global_scale = torch.mean(normalize_scales)
        normalize_scales = torch.full_like(normalize_scales, global_scale)

    # 检查归一化因子是否有效，确保数据类型一致
    for i in range(len(normalize_scales)):
        if torch.isclose(normalize_scales[i],
                         torch.tensor(0.0, dtype=normalize_scales.dtype, device=normalize_scales.device)):
            raise ValueError(f"第{i + 1}个归一化因子接近零，无法进行归一化")

    # 应用缩放因子
    for i in range(len(coords)):
        coords[i] /= normalize_scales[i]

    # 打印归一化信息(如果需要)
    if verbose:
        print(f"空间坐标归一化参数:")
        print(f"缩放因子: {normalize_scales}")
        print(f"均值: {normalize_means}")

    return coords[0], coords[1], normalize_scales, normalize_means

def normalize_expression_matrices(XA, XB, verbose=False):
    """
    使用PyTorch实现的基因表达矩阵归一化函数

    参数:
    XA (torch.Tensor): 第一个样本的基因表达矩阵
    XB (torch.Tensor): 第二个样本的基因表达矩阵
    verbose (bool): 是否打印归一化参数信息

    返回:
    tuple: 归一化后的矩阵 (XA_norm, XB_norm)

    异常:
    ValueError: 当无法计算归一化因子时抛出
    """
    # 确保输入是PyTorch张量
    if not isinstance(XA, torch.Tensor):
        XA = torch.tensor(XA)
    if not isinstance(XB, torch.Tensor):
        XB = torch.tensor(XB)

    # 计算归一化因子
    normalize_scale = 0

    # 计算每个矩阵的Frobenius范数并累加
    for X in [XA, XB]:
        # 计算矩阵元素平方和的均值
        squared_sum = torch.sum(X * X) / X.shape[0]
        # 取平方根并累加到归一化因子
        normalize_scale += torch.sqrt(squared_sum)

    # 计算平均归一化因子
    normalize_scale /= 2.0

    # 检查归一化因子是否有效，确保数据类型一致
    if torch.isclose(normalize_scale,
                     torch.tensor(0.0, dtype=normalize_scale.dtype, device=normalize_scale.device)):
        raise ValueError("归一化因子接近零，无法进行归一化")

    # 应用归一化
    XA_norm = XA / normalize_scale
    XB_norm = XB / normalize_scale

    # 打印归一化信息(如果需要)
    if verbose:
        print(f"基因表达归一化参数:")
        print(f"缩放因子: {normalize_scale.item():.6f}")

    return XA_norm, XB_norm

def torch_corrcoef(src_exp, tgt_exp):
    """
    计算两个矩阵之间的皮尔逊相关系数矩阵
    """
    # 确保每个矩阵的列都去均值
    src_exp_centered = src_exp - src_exp.mean(dim=0, keepdim=True)  # 每列去均值
    tgt_exp_centered = tgt_exp - tgt_exp.mean(dim=0, keepdim=True)  # 每列去均值

    # 计算协方差矩阵
    cov_matrix = torch.mm(src_exp_centered, tgt_exp_centered.T) / src_exp_centered.shape[-1]

    # 计算每列的范数（标准差）
    src_norm = torch.norm(src_exp_centered, dim=1)  # [n_features]
    tgt_norm = torch.norm(tgt_exp_centered, dim=1)  # [m_features]
    # 归一化协方差矩阵
    corr_matrix = cov_matrix / (src_norm.view(-1, 1) * tgt_norm.view(1, -1))  # 广播得到 [n_features, m_features]
    positive_corr = torch.clamp(corr_matrix, min=1e-8)

    return positive_corr

def harmony_integration(exp_A, exp_B):
    """使用Harmony进行批次校正并返回降维特征

    参数：
        exp_A : 样本数 × 基因数的矩阵 (numpy数组或稀疏矩阵)
        exp_B : 样本数 × 基因数的矩阵 (numpy数组或稀疏矩阵)

    返回：
        corrected_A : 校正后的A样本特征矩阵 (numpy数组)
        corrected_B : 校正后的B样本特征矩阵 (numpy数组)
    """
    # 创建AnnData对象并添加批次信息
    adata_A = ad.AnnData(exp_A)
    adata_A.obs['batch'] = 'batch_0'

    adata_B = ad.AnnData(exp_B)
    adata_B.obs['batch'] = 'batch_1'

    # 合并数据集
    adata = adata_A.concatenate(adata_B, join='outer', batch_key='batch')

    # 预处理流程
    sc.pp.normalize_total(adata, target_sum=1e4)  # CPM归一化
    sc.pp.log1p(adata)  # log1p转换
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)  # 筛选高变基因
    adata = adata[:, adata.var.highly_variable]  # 保留高变基因

    # PCA降维
    sc.tl.pca(adata, svd_solver='arpack', n_comps=50)

    # Harmony批次校正
    harmony_emb = hm.run_harmony(adata.obsm['X_pca'], adata.obs, 'batch')
    adata.obsm['X_pca_harmony'] = harmony_emb.Z_corr.T

    # 按原始样本顺序分割数据
    orig_order = np.concatenate([
        np.arange(exp_A.shape[0]),
        exp_A.shape[0] + np.arange(exp_B.shape[0])
    ])
    adata = adata[orig_order].copy()  # 保持原始样本顺序

    # 分割校正后的特征
    corrected_A = adata[:exp_A.shape[0]].obsm['X_pca_harmony']
    corrected_B = adata[exp_A.shape[0]:].obsm['X_pca_harmony']

    return corrected_A, corrected_B

# 计算基因表达矩阵之间的相关性（P）
def kl_distance_backend(
        X,
        Y,
        probabilistic: bool = True,
        eps: float = 1e-8,
        device=None,
):
    """
    Compute the pairwise KL divergence between all pairs of samples in matrices X and Y.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Matrix with shape (N, D), where each row represents a sample.
    Y : np.ndarray or torch.Tensor
        Matrix with shape (M, D), where each row represents a sample.
    probabilistic : bool, optional
        If True, normalize the rows of X and Y to sum to 1 (to interpret them as probabilities).
        Default is True.
    eps : float, optional
        A small value to avoid division by zero. Default is 1e-8.

    Returns
    -------
    np.ndarray
        Pairwise KL divergence matrix with shape (N, M).

    Raises
    ------
    AssertionError
        If the number of features in X and Y do not match.
    """
    if isinstance(X, np.ndarray):
        X = torch.tensor(X).to(device)
    if isinstance(Y, np.ndarray):
        Y = torch.tensor(Y).to(device)
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    X = (X - X.min()) / (X.max() - X.min())
    Y = (Y - Y.min()) / (Y.max() - Y.min())
    X = X + 0.01
    Y = Y + 0.01
    # Normalize rows to sum to 1 if probabilistic is True
    if probabilistic:
        X = X / torch.sum(X, dim=1, keepdims=True)
        Y = Y / torch.sum(Y, dim=1, keepdims=True)

    # Compute log of X and Y
    log_X = torch.log(X + eps)  # Adding epsilon to avoid log(0)
    log_Y = torch.log(Y + eps)  # Adding epsilon to avoid log(0)

    # Compute X log X and the pairwise KL divergence
    X_log_X = torch.sum(X * log_X, dim=1, keepdims=True)

    D = X_log_X - torch.matmul(X, log_Y.T)

    return D


def get_rotation_angle(R, reture_axis=False):
    """获取旋转轴和角度，支持2D和3D"""
    D = R.shape[0]
    device = R.device

    if D == 2:
        # 2D情况：旋转轴固定为Z轴(0,0,1)，只需计算角度
        cos_theta = (R[0, 0] + R[1, 1]) / 2
        sin_theta = (R[1, 0] - R[0, 1]) / 2
        theta = torch.atan2(sin_theta, cos_theta)
        angle_deg = torch.rad2deg(theta)
        axis = torch.tensor([0, 0, 1], device=device)  # 2D旋转轴为Z轴

    elif D == 3:
        # 3D情况：计算轴和角度
        trace = torch.trace(R)
        cos_theta = (trace - 1) / 2
        theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))

        # 处理特殊情况：旋转角度接近0（此时轴向量不稳定）
        if torch.abs(theta) < 1e-6:
            # 零旋转，返回任意单位轴（例如X轴）
            return torch.tensor([1, 0, 0], device=device), torch.tensor(0.0, device=device)

        # 处理特殊情况：旋转角度接近180度（此时需要特殊处理）
        elif torch.abs(theta - torch.pi) < 1e-6:
            # 当theta=π时，直接从矩阵元素计算轴向量
            xx = (R[0, 0] + 1) / 2
            yy = (R[1, 1] + 1) / 2
            zz = (R[2, 2] + 1) / 2
            xy = (R[0, 1] + R[1, 0]) / 4
            xz = (R[0, 2] + R[2, 0]) / 4
            yz = (R[1, 2] + R[2, 1]) / 4

            # 选择最大对角元素以获得数值稳定性
            max_val = max(xx, yy, zz)

            if max_val == xx:
                x = torch.sqrt(xx)
                y = xy / x
                z = xz / x
            elif max_val == yy:
                y = torch.sqrt(yy)
                x = xy / y
                z = yz / y
            else:  # max_val == zz
                z = torch.sqrt(zz)
                x = xz / z
                y = yz / z

            axis = torch.stack([x, y, z])

        else:
            # 一般情况：通过反对称矩阵计算轴
            K = (R - R.T) / 2
            axis = torch.stack([
                K[2, 1],  # R[2,1] - R[1,2]
                K[0, 2],  # R[0,2] - R[2,0]
                K[1, 0]  # R[1,0] - R[0,1]
            ])
            axis = axis / (2 * torch.sin(theta))

        # 归一化轴向量
        axis = axis / torch.norm(axis)
        angle_deg = torch.rad2deg(theta)

    else:
        # 非2D/3D情况返回零旋转
        axis, angle_deg=torch.tensor([1, 0, 0], device=device), torch.tensor(0.0, device=device)
    if reture_axis:
        return f'{angle_deg} axis={axis}'
    else:
        return angle_deg

def voxel_data_torch(
        coords: torch.Tensor,
        gene_exp: torch.Tensor,
        voxel_size: float = None,
        voxel_num: int = 10000,
):
    """
    PyTorch 版体素化函数（支持 GPU 加速）

    参数：
        coords: 空间坐标张量 (N, D)
        gene_exp: 基因表达矩阵 (N, G)
        voxel_size: 体素边长（自动计算时设为None）
        voxel_num: 目标体素数量（当voxel_size为None时生效）
        device: 计算设备 ('cuda' 或 'cpu')

    返回：
        voxel_coords: 有效体素坐标 (M, D)
        voxel_gene_exp: 体素基因表达 (M, G)
    """
    # 设备配置
    device = coords.device
    dtype = coords.dtype
    N, D = coords.shape

    # 计算坐标边界
    min_coords = torch.min(coords, dim=0).values
    max_coords = torch.max(coords, dim=0).values
    spatial_range = max_coords - min_coords

    # 自动计算体素尺寸
    if voxel_size is None:
        spatial_range = max_coords - min_coords
        volume = torch.prod(spatial_range)
        points_per_voxel = 5 ** D
        voxel_size = (volume * points_per_voxel / N) ** (1 / D)

    # 生成体素网格 - 根据维度动态计算网格大小
    # 关键修改：使用维度相关的网格大小计算
    if D == 2:
        # 2D: 使用平方根
        grid_size = int(torch.sqrt(torch.tensor(voxel_num)))
    else:
        # 3D+: 使用立方根
        grid_size = int(round(torch.pow(torch.tensor(voxel_num), 1 / D).item()))

    voxel_steps = spatial_range / grid_size

    # 创建多维网格
    grid = [torch.arange(min_coords[d], max_coords[d], voxel_steps[d],
                         device=device, dtype=dtype) for d in range(D)]
    mesh = torch.meshgrid(grid, indexing='ij')
    voxel_coords = torch.stack(mesh, dim=-1).reshape(-1, D)  # (M, D)

    # 批量距离计算
    dists = torch.cdist(coords, voxel_coords, p=2)  # (N, M)
    valid_mask = dists < (voxel_size / 2)  # (N, M)

    # 向量化聚合计算
    mask_sum = valid_mask.sum(dim=0)  # (M,)
    valid_voxels = mask_sum > 0  # (M,)
    valid_mask = valid_mask.float().T.to(dtype=gene_exp.dtype)
    aggregated = torch.mm(valid_mask, gene_exp)  # (M, G)

    # 计算平均表达
    voxel_gene_exp = aggregated / mask_sum[:, None]  # (M, G)
    voxel_gene_exp[~valid_voxels] = 0  # 处理空体素

    # 筛选有效体素
    return voxel_coords[valid_voxels], voxel_gene_exp[valid_voxels]


# def inner_NN(exp_A, exp_B, X_A, X_B, max_iter=100, tol=1e-7, exp_P=None):
#     D = X_A.shape[1]
#     device = X_A.device
#     # 可以调整的参数
#     # 计算表达相似性
#     if exp_P is None:
#         exp_P = cal_exp_P(exp_A, exp_B)
#
#     # 初始化旋转矩阵和平移向量
#     R = torch.eye(D, dtype=torch.float64, device=device)
#     t = torch.zeros(D, dtype=torch.float64, device=device)
#
#     prev_R, prev_t = R.clone(), t.clone()
#     decay_factor = 1
#     for iter_idx in range(max_iter):
#         # ----------------------------
#         # 1. 计算点对距离矩阵
#         # ----------------------------
#         # 计算X_A和X_B之间的欧氏距离矩阵
#         dist_matrix = torch.cdist(X_A @ R.T + t, X_B, p=2.0)
#         median_dist = torch.median(dist_matrix)
#         threshold = 3 * median_dist
#
#         # 将超过阈值的值截断为阈值
#         dist_matrix = torch.where(dist_matrix > threshold, threshold, dist_matrix)
#         # 转换为相似度矩阵 (距离越远，相似度越低)
#         # 使用高斯核而非log
#         # ----------------------------
#         # 3. 计算均值与去中心化坐标
#         # ----------------------------
#         sigma = median_dist / 2.0
#         spatial_sim = torch.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
#         spatial_P1 = spatial_sim / (spatial_sim.sum(dim=0, keepdim=True) + 1e-15)
#         spatial_P2 = spatial_sim / (spatial_sim.sum(dim=1, keepdim=True) + 1e-15)
#         spatial_P = (spatial_P1 + spatial_P2) / 2.0
#         eps = 1e-15
#         spatial_P = torch.log(spatial_P + eps) - torch.log(torch.min(spatial_P) + eps)
#         spatial_P = spatial_P / torch.max(spatial_P)
#         exp_P_clamped = exp_P * spatial_P
#         P = exp_P_clamped / (exp_P_clamped.sum(dim=1, keepdim=True) + 1e-15) + 1e-15
#         total_weight = P.sum()
#
#         # 目标点云加权均值
#         weights_B = P.sum(dim=0)  # [N_B]
#         mu_XB = (X_B.T @ weights_B) / total_weight  # [D]
#
#         # 源点云加权均值
#         weights_A = P.sum(dim=1)  # [N_A]
#         mu_XA = (X_A.T @ weights_A) / total_weight  # [D]
#
#         # 去中心化坐标
#         XB_centered = X_B - mu_XB  # [N_B, D]
#         XA_centered = X_A - mu_XA  # [N_A, D]
#
#         A = XB_centered.T @ (P.T @ XA_centered)  # [D, D]
#
#         U, S, Vh = torch.linalg.svd(A)
#         V = Vh.T
#
#         # 保证右手坐标系
#         C = torch.eye(D, dtype=U.dtype, device=device)
#         if torch.det(U @ V.T) < 0:
#             C[-1, -1] = -1.0
#
#         R = U @ C @ V.T
#
#         # ----------------------------
#         # 6. 计算平移向量
#         # ----------------------------
#         t = mu_XB - R @ mu_XA
#         print(
#             f"\r迭代 {iter_idx + 1}/{max_iter} - 旋转角度: {get_rotation_angle(R):.4f}°, 平移向量: [{', '.join([f'{v:.6f}' for v in t])}]",
#             end='')
#         # 检查收敛
#         if (torch.norm(R - prev_R) < tol) and (torch.norm(t - prev_t) < tol):
#             print(f"迭代 {iter_idx + 1}/{max_iter} 已收敛")
#             break
#
#         prev_R, prev_t = R.clone(), t.clone()
#
#     dist_matrix = torch.cdist(X_A @ R.T + t, X_B, p=2.0)
#     spatial_sim = torch.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
#     spatial_P = spatial_sim / (spatial_sim.sum(dim=1, keepdim=True) + 1e-15)
#     P = exp_P * clamp_P(spatial_P)
#
#     # 构建齐次变换矩阵
#     M = torch.eye(D + 1, dtype=R.dtype, device=device)
#     M[:D, :D] = R
#     M[:D, D] = t
#     return M.cpu().detach(), P.cpu().detach()
#
def cal_single_P(A, B, k_beta=None, sigma=None, eps=1e-15, method='kl', use_threshold=False):
    if method == 'kl':
        exp_P = kl_distance_backend(A, B)
    if method == 'dist':
        exp_P = torch.cdist(A, B, p=2.0)

    median_dist = 1.5 * torch.median(exp_P)

    if sigma is None:
        sigma = median_dist / 2.0
    if use_threshold:
        threshold = 3 * median_dist
        exp_P = torch.where(exp_P > threshold, threshold, exp_P)
    exp_P = torch.exp(-exp_P ** 2 / (2 * sigma ** 2))
    exp_P1 = exp_P / (exp_P.sum(dim=1, keepdim=True) + 1e-15)
    exp_P2 = exp_P / (exp_P.sum(dim=0, keepdim=True) + 1e-15)
    if not k_beta is None:
        exp_P1 = clamp_P(exp_P1, k_beta, True)
        exp_P2 = clamp_P(exp_P2.T, k_beta, True).T
    exp_P = (exp_P1 + exp_P2) / 2.0
    exp_P = torch.log(exp_P + eps) - torch.log(torch.min(exp_P) + eps)
    exp_P = exp_P / torch.max(exp_P)
    return exp_P



def cal_all_P(spatial_P, exp_P, alpha=0.5):
    return (1 - alpha) * spatial_P + alpha * exp_P


def inner_NN(exp_A, exp_B, X_A, X_B, max_iter=100, tol=1e-7, alpha=0.5, exp_P=None):
    D = X_A.shape[1]
    device = X_A.device
    dtype = X_A.dtype
    # 可以调整的参数
    # 计算表达相似性
    if exp_P is None:
        exp_P = cal_single_P(exp_A, exp_B, k_beta=0.1)

    # 初始化旋转矩阵和平移向量
    R = torch.eye(D, dtype=dtype, device=device)
    t = torch.zeros(D, dtype=dtype, device=device)

    prev_R, prev_t = R.clone(), t.clone()
    decay_factor = 1
    for iter_idx in range(max_iter):
        # ----------------------------
        # 1. 计算点对距离矩阵
        # ----------------------------
        # 计算X_A和X_B之间的欧氏距离矩阵
        spatial_P = cal_single_P(X_A @ R.T + t, X_B, method='dist', use_threshold=True)

        P = cal_all_P(spatial_P, exp_P, alpha=alpha)  # exp_P * spatial_P #  ((1-alpha) * spatial_P.T + alpha * exp_P)
        # P = exp_P_clamped / (exp_P_clamped.sum(dim=1, keepdim=True) + 1e-15) + 1e-15
        total_weight = P.sum()

        # 目标点云加权均值
        weights_B = P.sum(dim=0)  # [N_B]
        mu_XB = (X_B.T @ weights_B) / total_weight  # [D]

        # 源点云加权均值
        weights_A = P.sum(dim=1)  # [N_A]
        mu_XA = (X_A.T @ weights_A) / total_weight  # [D]

        # 去中心化坐标
        XB_centered = X_B - mu_XB  # [N_B, D]
        XA_centered = X_A - mu_XA  # [N_A, D]

        A = XB_centered.T @ (P.T @ XA_centered)  # [D, D]

        U, S, Vh = torch.linalg.svd(A)
        V = Vh.T

        # 保证右手坐标系
        C = torch.eye(D, dtype=U.dtype, device=device)
        if torch.det(U @ V.T) < 0:
            C[-1, -1] = -1.0

        R = U @ C @ V.T

        # ----------------------------
        # 6. 计算平移向量
        # ----------------------------
        t = mu_XB - R @ mu_XA
        alpha = alpha * 0.9
        print(
            f"\r迭代 {iter_idx + 1}/{max_iter} - 旋转角度: {get_rotation_angle(R):.4f}°, 平移向量: [{', '.join([f'{v:.6f}' for v in t])}]",
            end='')
        # 检查收敛
        if (torch.norm(R - prev_R) < tol) and (torch.norm(t - prev_t) < tol):
            print(f"迭代 {iter_idx + 1}/{max_iter} 已收敛")
            break

        prev_R, prev_t = R.clone(), t.clone()

    spatial_P = cal_single_P(X_A @ R.T + t, X_B, method='dist', use_threshold=True)
    P = exp_P * clamp_P(spatial_P)  # 联合概率分布

    # 构建齐次变换矩阵
    M = torch.eye(D + 1, dtype=R.dtype, device=device)
    M[:D, :D] = R
    M[:D, D] = t
    return M.cpu().detach(), P.cpu().detach()
#
#
# def cal_single_P(A, B, k_beta=None, sigma=None, eps=1e-15, method='kl', use_threshold=False):
#     """计算点对点相似性矩阵，支持任意维度"""
#     if method == 'kl':
#         exp_P = kl_distance_backend(A, B)  # 确保此函数支持高维
#     elif method == 'dist':
#         # 支持任意维度的距离计算
#         exp_P = torch.cdist(A, B, p=2.0)
#     else:
#         raise ValueError(f"未知方法: {method}")
#
#     # 自动确定sigma参数
#     median_dist = torch.median(exp_P[exp_P > 0]) if exp_P.numel() > 0 else 1.0
#     median_dist = 1.5 * median_dist
#
#     if sigma is None:
#         sigma = median_dist / 2.0
#
#     # 可选的距离截断
#     if use_threshold:
#         threshold = 3 * median_dist
#         exp_P = torch.where(exp_P > threshold, threshold, exp_P)
#
#     # 计算高斯相似性
#     exp_P = torch.exp(-exp_P ** 2 / (2 * sigma ** 2))
#
#     # 行归一化和列归一化
#     exp_P1 = exp_P / (exp_P.sum(dim=1, keepdim=True) + eps
#                       exp_P2 = exp_P/ (exp_P.sum(dim=0, keepdim=True) + eps
#
#     # 可选的双向截断
#     if k_beta is not None:
#         exp_P1 = clamp_P(exp_P1, k_beta, clamp_rows=True)
#     exp_P2 = clamp_P(exp_P2.T, k_beta, clamp_rows=True).T
#
#     # 对称化概率矩阵
#     exp_P = (exp_P1 + exp_P2) / 2.0
#
#     # 数值稳定化处理
#     min_val = torch.min(exp_P) + eps
#     exp_P = torch.log(exp_P + eps) - torch.log(min_val)
#     exp_P = exp_P / torch.max(exp_P)
#
#     return exp_P

#
# def cal_all_P(spatial_P, exp_P, alpha=0.5):
#     """组合空间和表达相似性"""
#     return (1 - alpha) * spatial_P + alpha * exp_P
#
#
# def inner_NN(exp_A, exp_B, X_A, X_B, max_iter=100, tol=1e-7, alpha=0.5, exp_P=None):
#     """
#     点云配准方法，支持2D、3D和更高维度
#     参数:
#         exp_A, exp_B: 表达特征矩阵 [N_A, F], [N_B, F]
#         X_A, X_B: 空间坐标矩阵 [N_A, D], [N_B, D] (D=2,3,4...)
#     """
#     # 检查维度一致性
#     assert X_A.shape[1] == X_B.shape[1], "空间维度必须一致"
#     D = X_A.shape[1]  # 空间维度 (2,3,4...)
#     device = X_A.device
#     dtype = X_A.dtype
#
#     # 计算表达相似性矩阵
#     if exp_P is None:
#         exp_P = cal_single_P(exp_A, exp_B, k_beta=0.1)
#
#     # 初始化变换参数
#     R = torch.eye(D, dtype=dtype, device=device)  # 旋转矩阵
#     t = torch.zeros(D, dtype=dtype, device=device)  # 平移向量
#
#     # 迭代优化
#     prev_loss = float('inf')
#     for iter_idx in range(max_iter):
#         # 1. 应用当前变换
#         transformed_A = X_A @ R.T + t
#
#         # 2. 计算空间相似性
#         spatial_P = cal_single_P(transformed_A, X_B, method='dist', use_threshold=True)
#
#         # 3. 组合空间和表达相似性
#         P = cal_all_P(spatial_P, exp_P, alpha=alpha)
#
#         # 4. 计算加权质心
#         total_weight = P.sum()
#         weights_B = P.sum(dim=0)  # [N_B]
#         weights_A = P.sum(dim=1)  # [N_A]
#
#         mu_B = (X_B.T @ weights_B) / total_weight  # [D]
#         mu_A = (X_A.T @ weights_A) / total_weight  # [D]
#
#         # 5. 去中心化
#         XB_centered = X_B - mu_B
#         XA_centered = X_A - mu_A
#
#         # 6. 计算协方差矩阵
#         A = XB_centered.T @ (P.T @ XA_centered)  # [D, D]
#
#         # 7. SVD分解求解最优旋转
#         U, S, Vh = torch.linalg.svd(A)
#         V = Vh.T
#
#         # 8. 处理反射情况 (确保是纯旋转)
#         det_UV = torch.det(U @ V.T)
#         correction = torch.eye(D, device=device, dtype=dtype)
#         if det_UV < 0:
#             correction[-1, -1] = -1.0
#
#         R = U @ correction @ V.T
#
#         # 9. 计算平移
#         t = mu_B - R @ mu_A
#
#         # 10. 计算损失并检查收敛
#         current_loss = torch.norm(transformed_A - X_B, p='fro').item()
#         loss_delta = abs(prev_loss - current_loss)
#
#         # 打印迭代信息 (维度无关)
#         print(f"Iter {iter_idx + 1}/{max_iter}: Loss={current_loss:.6f}, Δ={loss_delta:.6f}, "
#               f"Trans={' '.join([f'{x:.4f}' for x in t])}")
#
#         # 检查收敛条件
#         if loss_delta < tol:
#             print(f"Converged at iteration {iter_idx + 1}")
#             break
#
#         prev_loss = current_loss
#         alpha *= 0.95  # 衰减表达相似性的权重
#
#     # 最终变换和概率矩阵
#     final_transformed = X_A @ R.T + t
#     spatial_P = cal_single_P(final_transformed, X_B, method='dist', use_threshold=True)
#     P = exp_P * clamp_P(spatial_P)  # 联合概率分布
#
#     # 构建齐次变换矩阵 [D+1, D+1]
#     M = torch.eye(D + 1, dtype=dtype, device=device)
#     M[:D, :D] = R
#     M[:D, D] = t
#
#     return M.cpu().detach(), P.cpu().detach()

def clamp_P(P_matrix, k_beta=0.1, min_k=5, reture_values=True):
    """
    按行筛选矩阵，保留每行中最大的k_beta比例的元素（至少min_k个），其余设为0

    参数:
    P_matrix: 输入矩阵
    k_beta: 每行保留的元素比例 (默认0.1，即10%)
    min_k: 每行至少保留的元素数量 (默认5)

    返回:
    筛选后的矩阵
    """
    # 确保矩阵元素非负
    P_clamped = torch.clamp(P_matrix, 0, float('inf'))

    # 获取矩阵形状
    rows, cols = P_clamped.shape

    # 计算每行需要保留的元素数量（至少min_k个）
    k_per_row = max(int(cols * k_beta), min_k)

    # 确保k_per_row不超过列数
    k_per_row = min(k_per_row, cols)

    # 创建掩码矩阵，用于标记每行中需要保留的元素
    mask = torch.zeros_like(P_clamped, dtype=torch.bool)
    result = torch.zeros_like(P_clamped)

    # 对每行找到最大的k_per_row个元素的索引
    for i in range(rows):
        # 获取当前行中最大的k_per_row个元素的索引
        _, indices = torch.topk(P_clamped[i], k_per_row)
        # 在掩码中标记这些位置为True
        if reture_values:
            mask[i, indices] = True
        else:
            result[i, indices] = 1
    # 应用掩码，将非标记位置设为0
    if reture_values:
        result = torch.where(mask, P_clamped, torch.zeros_like(P_clamped))
        return result
    else:
        return result

def calculate_data_scale(X_A, X_B):
    """计算数据的尺度特征，用于自适应参数设置"""
    # 合并所有点
    all_points = torch.cat([X_A, X_B])

    # 计算点云的空间范围
    min_vals = all_points.min(dim=0).values
    max_vals = all_points.max(dim=0).values
    spatial_range = max_vals - min_vals

    # 计算点云密度特征
    dists = torch.cdist(X_A, X_A)
    dists.fill_diagonal_(float('inf'))  # 忽略自身距离
    min_dists = dists.min(dim=1).values
    median_dist = torch.median(min_dists).item()

    # 计算空间尺度特征
    spatial_scale = torch.norm(spatial_range).item()

    return {
        'spatial_scale': spatial_scale,  # 点云整体尺度
        'median_dist': median_dist,  # 点云中位距离
        'dims': X_A.shape[1]  # 空间维度
    }


# 辅助函数: 应用齐次变换矩阵
def transform_points(points, M):
    n = points.shape[0]
    points_homo = torch.cat([points, torch.ones(n, 1, device=points.device)], dim=1)
    transformed = torch.mm(points_homo, M.t())
    return transformed[:, :points.shape[1]]


# 辅助函数: 创建扰动变换矩阵
# 辅助函数 - 维度无关的扰动矩阵创建
# def create_perturbation_matrix(rot_params, trans_vector, D, device, dtype):
#     """
#     根据维度创建扰动变换矩阵
#     参数:
#         rot_params: 旋转参数列表
#         trans_vector: 平移向量 (D,)
#         D: 空间维度
#     """
#     # 创建基础单位矩阵
#     M = torch.eye(D + 1, dtype=dtype, device=device)
#
#     # 应用平移
#     M[:D, D] = trans_vector
#
#     # 根据维度应用旋转
#     if D == 2:
#         # 2D: 绕Z轴旋转
#         angle = torch.deg2rad(torch.tensor(rot_params[0], device=device))
#         cos_a = torch.cos(angle)
#         sin_a = torch.sin(angle)
#         R = torch.tensor([
#             [cos_a, -sin_a],
#             [sin_a, cos_a]
#         ], device=device, dtype=dtype)
#         M[:2, :2] = R
#
#     elif D == 3:
#         # 3D: 欧拉角 (ZYX顺序)
#         angles = torch.deg2rad(torch.tensor(rot_params, device=device))
#
#         # 绕Z轴旋转
#         cos_z = torch.cos(angles[2])
#         sin_z = torch.sin(angles[2])
#         Rz = torch.tensor([
#             [cos_z, -sin_z, 0],
#             [sin_z, cos_z, 0],
#             [0, 0, 1]
#         ], device=device, dtype=dtype)
#
#         # 绕Y轴旋转
#         cos_y = torch.cos(angles[1])
#         sin_y = torch.sin(angles[1])
#         Ry = torch.tensor([
#             [cos_y, 0, sin_y],
#             [0, 1, 0],
#             [-sin_y, 0, cos_y]
#         ], device=device, dtype=dtype)
#
#         # 绕X轴旋转
#         cos_x = torch.cos(angles[0])
#         sin_x = torch.sin(angles[0])
#         Rx = torch.tensor([
#             [1, 0, 0],
#             [0, cos_x, -sin_x],
#             [0, sin_x, cos_x]
#         ], device=device, dtype=dtype)
#
#         # 组合旋转 R = Rz * Ry * Rx
#         R = Rz @ Ry @ Rx
#         M[:3, :3] = R
#
#     elif D >= 4:
#         # 4D+: 使用轴角表示法
#         angle = torch.deg2rad(torch.tensor(rot_params[0], device=device))
#         # 随机选择旋转平面 (实际应用中可能需要更复杂处理)
#         axis1 = torch.zeros(D, device=device)
#         axis2 = torch.zeros(D, device=device)
#         axis1[0] = 1
#         axis2[1] = 1
#
#         # 创建旋转矩阵 (简化版)
#         cos_a = torch.cos(angle)
#         sin_a = torch.sin(angle)
#
#         R = torch.eye(D, device=device, dtype=dtype)
#         R[0, 0] = cos_a
#         R[0, 1] = -sin_a
#         R[1, 0] = sin_a
#         R[1, 1] = cos_a
#
#         M[:D, :D] = R
#     return M


def create_perturbation_matrix(rot_params, trans_vector, D, device, dtype, deg=False):
    """
    根据维度创建扰动变换矩阵，保持梯度传播
    参数:
        rot_params: 旋转参数张量（可微分）
        trans_vector: 平移向量张量 (D,)（可微分）
        D: 空间维度
    """
    # 创建基础单位矩阵（保持梯度）
    M = torch.eye(D + 1, dtype=dtype, device=device)

    # 应用平移（保持梯度）
    M = M.clone()  # 创建可修改的副本
    M[:D, D] = trans_vector[:D]

    # 根据维度应用旋转
    if D == 2:
        # 2D: 绕Z轴旋转
        if deg == False:
            angle = torch.deg2rad(rot_params[0])
        else:
            angle = rot_params[0]
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        # 直接构建旋转矩阵（保持梯度）
        M[0, 0] = cos_a
        M[0, 1] = -sin_a
        M[1, 0] = sin_a
        M[1, 1] = cos_a

    elif D == 3:
        # 3D: 欧拉角 (ZYX顺序)
        if deg == False:
            angles = torch.deg2rad(rot_params)
        else:
            angles = rot_params
        # 直接构建旋转矩阵（避免创建新张量）
        cos_z = torch.cos(angles[2])
        sin_z = torch.sin(angles[2])
        cos_y = torch.cos(angles[1])
        sin_y = torch.sin(angles[1])
        cos_x = torch.cos(angles[0])
        sin_x = torch.sin(angles[0])

        # 构建旋转矩阵元素（保持梯度）
        M[0, 0] = cos_z * cos_y
        M[0, 1] = cos_z * sin_y * sin_x - sin_z * cos_x
        M[0, 2] = cos_z * sin_y * cos_x + sin_z * sin_x

        M[1, 0] = sin_z * cos_y
        M[1, 1] = sin_z * sin_y * sin_x + cos_z * cos_x
        M[1, 2] = sin_z * sin_y * cos_x - cos_z * sin_x

        M[2, 0] = -sin_y
        M[2, 1] = cos_y * sin_x
        M[2, 2] = cos_y * cos_x

    elif D >= 4:
        # 4D+: 使用轴角表示法（简化版）
        angle = torch.deg2rad(rot_params[0])
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        # 直接修改旋转矩阵部分（保持梯度）
        M[0, 0] = cos_a
        M[0, 1] = -sin_a
        M[1, 0] = sin_a
        M[1, 1] = cos_a

    return M


def create_affine_matrix(rot_params, trans_vector, scale_params, D, device, dtype, deg=False):
    """
    根据维度创建完整的仿射变换矩阵（平移、旋转、错切、缩放）。

    参数:
        rot_params: 旋转参数
        trans_vector: 平移向量 (D,)
        scale_params: 缩放因子 (D,)
        shear_params: 错切因子
        D: 空间维度 (2 或 3)
    """
    # 最终的 (D+1)x(D+1) 齐次变换矩阵
    M = torch.eye(D + 1, dtype=dtype, device=device)

    # 1. 构造 DxD 的线性变换部分 (A = R * H * S)
    # R: Rotation, H: Shear, S: Scale

    # -- 缩放矩阵 S --
    S = torch.diag(scale_params[:D])
    # -- 旋转矩阵 R --
    R = torch.eye(D, dtype=dtype, device=device)
    if D == 2:
        angle = rot_params[0] if deg else torch.deg2rad(rot_params[0])
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        R[0, 0], R[0, 1] = cos_a, -sin_a
        R[1, 0], R[1, 1] = sin_a, cos_a
    elif D == 3:
        angles = rot_params if deg else torch.deg2rad(rot_params)
        cos_z, sin_z = torch.cos(angles[2]), torch.sin(angles[2])
        cos_y, sin_y = torch.cos(angles[1]), torch.sin(angles[1])
        cos_x, sin_x = torch.cos(angles[0]), torch.sin(angles[0])

        # ZYX 顺序欧拉角
        Rx = torch.eye(3, dtype=dtype, device=device)
        Rx[1, 1], Rx[1, 2] = cos_x, -sin_x
        Rx[2, 1], Rx[2, 2] = sin_x, cos_x

        Ry = torch.eye(3, dtype=dtype, device=device)
        Ry[0, 0], Ry[0, 2] = cos_y, sin_y
        Ry[2, 0], Ry[2, 2] = -sin_y, cos_y

        Rz = torch.eye(3, dtype=dtype, device=device)
        Rz[0, 0], Rz[0, 1] = cos_z, -sin_z
        Rz[1, 0], Rz[1, 1] = sin_z, cos_z

        R = Rz @ Ry @ Rx

    # 2. 组合线性变换: A = Rotation * Scale
    # 注意：矩阵乘法顺序会影响最终效果，R*H*S 是一个常用约定
    A = R @ S

    # 3. 将线性部分和变换部分放入齐次矩阵 M
    M[:D, :D] = A
    M[:D, D] = trans_vector[:D]

    return M

def compute_matching_score(X_A_trans, X_B, match_threshold, exp_P, alpha, similarity_mode):
    """计算当前变换下的匹配分数"""
    eps = 1e-15
    sigma = match_threshold/2.0
    spatial_P = cal_single_P(X_A_trans, X_B, sigma=sigma, method='dist', use_threshold=False)

    # 结合表达信息 (如果提供)
    if exp_P is not None:
        exp_P_normalized = torch.clamp(exp_P, 0, 1)

        if similarity_mode == "probabilistic":
            combined_prob = cal_all_P(spatial_P, exp_P_normalized, alpha=alpha)
        else:
            exp_weight = 1 - alpha * exp_P_normalized
            combined_prob = spatial_P * exp_weight
    else:
        combined_prob = spatial_P

    # 计算每个点的最大匹配概率
    combined_prob = torch.nan_to_num(combined_prob, nan=eps)
    max_prob1, _ = combined_prob.max(dim=0)
    max_prob2, _ = combined_prob.max(dim=1)
    max_prob = torch.cat([max_prob1, max_prob2], dim=0)
    valid_mask = max_prob >= max_prob.mean()/100.0
    valid_probs = max_prob[valid_mask]
    # 返回匹配分数 (所有点的最大匹配概率之和)
    score = valid_probs.mean()
    if valid_probs.numel() == 0 or torch.isnan(score):
        return torch.tensor(0.0, device=max_prob.device)
    return score


def maximize_matches(X_A, X_B, M_init,method='Bayesian',**kwargs):
    if method == 'Bayesian':
        return maximize_matches_Bayesian(X_A, X_B, M_init,**kwargs)
    else:
        return maximize_matches_RandomPerturbation(X_A, X_B, M_init,**kwargs)


def maximize_matches_RandomPerturbation(X_A, X_B, M_init,
                     match_threshold=None, angle_range=(-25, 25),
                     trans_range=None, num_samples=2048, exp_P=None,
                     alpha=0.5, similarity_mode="probabilistic"):
    """
    在初始对齐的基础上，通过微调变换最大化匹配点对的概率

    参数:
        X_A, X_B: 空间坐标矩阵
        M_init: 初始变换矩阵 (D+1 x D+1)
        match_threshold: 匹配点对的距离阈值 (自动计算)
        angle_range: 旋转角度采样范围 (度)
        trans_range: 平移距离采样范围 (自动计算)
        num_samples: 采样次数
        exp_P: 表达相似性矩阵
        alpha: 表达信息权重强度 (0-1)
        similarity_mode: 相似性融合模式 ("weighted" 或 "probabilistic")
    """
    eps = 1e-15

    device = X_A.device
    D = X_A.shape[1]
    dtype = X_A.dtype
    M_init = M_init.to(device)

    # 自动计算数据尺度特征
    data_scale = calculate_data_scale(X_A, X_B)

    # 设置默认匹配阈值 (基于点云密度)
    if match_threshold is None:
        match_threshold = 8.0 * data_scale['median_dist']

    # 设置默认平移范围 (基于空间尺度)
    if trans_range is None:
        scale_factor = 0.05 * data_scale['spatial_scale']
        trans_range = (-scale_factor, scale_factor)

    print(f"自适应参数设置: 匹配阈值={match_threshold:.4f}, 平移范围=[{trans_range[0]:.4f}, {trans_range[1]:.4f}]")

    # 初始变换后的点云
    X_A_trans_init = transform_points(X_A, M_init)

    init_score = compute_matching_score(
        X_A_trans_init, X_B,
        match_threshold,
        exp_P, alpha, similarity_mode
    )
    best_score = init_score
    M_best = M_init.clone()
    # 在初始变换附近采样微调参数
    for i in range(num_samples):
        # 生成随机扰动 (旋转角度 + 平移向量)
        angle = torch.FloatTensor(1).uniform_(angle_range[0], angle_range[1]).item()

        # 为每个维度生成独立的平移扰动
        trans = torch.FloatTensor(D).uniform_(trans_range[0], trans_range[1]).to(device)

        # 构建扰动变换矩阵
        M_perturb = create_perturbation_matrix(angle, trans, D, device, dtype)

        # 组合变换: M_perturb * M_init
        M_current = M_perturb @ M_init

        # 应用变换
        X_A_trans = transform_points(X_A, M_current)
        current_score = compute_matching_score(
            X_A_trans, X_B,
            match_threshold,
            exp_P, alpha, similarity_mode
        )
        # 更新最优结果
        if current_score > best_score:
            improvement = current_score - best_score
            best_score = current_score
            M_best = M_current.clone()

    return M_best, best_score

#
#
# def maximize_matches_Bayesian(X_A, X_B, M_init,
#                               match_threshold=None, angle_range=(-25, 25),
#                               trans_range=None, n_calls=100, exp_P=None,
#                               alpha=0.5, similarity_mode="probabilistic"):
#     """
#     使用贝叶斯优化在初始对齐的基础上微调变换，最大化匹配点对的概率
#
#     参数:
#         X_A, X_B: 空间坐标矩阵 (N x D)
#         M_init: 初始变换矩阵 (D+1 x D+1)
#         match_threshold: 匹配点对的距离阈值 (自动计算)
#         angle_range: 旋转角度范围 (度)
#         trans_range: 平移距离范围 (自动计算)
#         n_calls: 贝叶斯优化评估次数
#         exp_P: 表达相似性矩阵 (N x M)
#         alpha: 表达信息权重强度 (0-1)
#         similarity_mode: 相似性融合模式 ("weighted" 或 "probabilistic")
#     """
#     device = X_A.device
#     dtype = X_A.dtype
#     D = X_A.shape[1]
#     M_init = M_init.to(device)
#
#     # 自动计算数据尺度特征
#     data_scale = calculate_data_scale(X_A, X_B)
#
#     # 设置默认匹配阈值 (基于点云密度)
#     if match_threshold is None:
#         match_threshold = 1.5 * data_scale['median_dist']
#
#     # 设置默认平移范围 (基于空间尺度)
#     if trans_range is None:
#         scale_factor = 0.05 * data_scale['spatial_scale']
#         trans_range = (-scale_factor, scale_factor)
#
#     print(f"自适应参数设置: 匹配阈值={match_threshold:.4f}, 平移范围=[{trans_range[0]:.4f}, {trans_range[1]:.4f}]")
#
#     # 定义参数空间 (角度 + D个平移维度)
#     dimensions = [
#         Real(low=angle_range[0], high=angle_range[1], name='angle')
#     ]
#     for i in range(D):
#         dimensions.append(Real(low=trans_range[0], high=trans_range[1], name=f'trans_{i}'))
#
#     # 缓存机制避免重复计算
#     cache = {}
#
#     def objective(params):
#         """目标函数: 计算当前变换下的匹配分数"""
#         # 尝试从缓存中获取结果
#         params_key = tuple(params)
#         if params_key in cache:
#             return cache[params_key]
#
#         # 解析参数
#         angle = params[0]
#         trans = torch.tensor(params[1:], device=device, dtype=dtype)
#
#         # 构建扰动变换矩阵
#         M_perturb = create_perturbation_matrix(angle, trans, D, device, dtype)
#
#         # 组合变换: M_perturb * M_init
#         M_current = M_perturb @ M_init
#
#         # 应用变换
#         X_A_trans = transform_points(X_A, M_current)
#
#         # 计算匹配分数
#         score = compute_matching_score(
#             X_A_trans, X_B,
#             match_threshold,
#             exp_P, alpha, similarity_mode
#         )
#
#         # 贝叶斯优化最小化目标，所以我们返回负分数
#         result = -score.item()
#
#         # 缓存结果
#         cache[params_key] = result
#         return result
#
#     # 初始点 (零扰动)
#     x0 = [0.0] + [0.0] * D
#
#     # 运行贝叶斯优化
#     res = gp_minimize(
#         func=objective,
#         dimensions=dimensions,
#         n_calls=n_calls,
#         x0=x0,
#         random_state=42,
#         n_jobs=1,  # 单线程以避免GPU冲突
#         verbose=False
#     )
#
#     # 提取最优参数
#     best_params = res.x
#     best_angle = best_params[0]
#     best_trans = torch.tensor(best_params[1:], device=device, dtype=dtype)
#
#     # 构建最优变换矩阵
#     M_perturb = create_perturbation_matrix(best_angle, best_trans, D, device, dtype)
#     M_best = M_perturb @ M_init
#
#     # 计算最优分数
#     X_A_trans_best = transform_points(X_A, M_best)
#     best_score = compute_matching_score(
#         X_A_trans_best, X_B,
#         match_threshold,
#         exp_P, alpha, similarity_mode
#     ).item()
#     return M_best, best_score
#
# def maximize_matches_Bayesian(X_A, X_B, M_init,
#                               match_threshold=None, angle_range=(-25, 25),
#                               trans_range=None, n_calls=100, exp_P=None,
#                               alpha=0.5, similarity_mode="probabilistic"):
#     """
#     使用贝叶斯优化在初始对齐的基础上微调变换，最大化匹配点对的概率
#     支持2D、3D、4D及更高维度的空间数据
#
#     参数:
#         X_A, X_B: 空间坐标矩阵 (N x D)
#         M_init: 初始变换矩阵 (D+1 x D+1)
#         match_threshold: 匹配点对的距离阈值 (自动计算)
#         angle_range: 旋转角度范围 (度) - 仅适用于2D/3D
#         trans_range: 平移距离范围 (自动计算)
#         n_calls: 贝叶斯优化评估次数
#         exp_P: 表达相似性矩阵 (N x M)
#         alpha: 表达信息权重强度 (0-1)
#         similarity_mode: 相似性融合模式 ("weighted" 或 "probabilistic")
#     """
#     device = X_A.device
#     dtype = X_A.dtype
#     M_init = M_init.to(device)
#     D = X_A.shape[1]  # 空间维度
#
#     # 自动计算数据尺度特征
#     data_scale = calculate_data_scale(X_A, X_B)
#
#     # 设置默认匹配阈值 (基于点云密度)
#     if match_threshold is None:
#         match_threshold = 1.5 * data_scale['median_dist']
#
#     # 设置默认平移范围 (基于空间尺度)
#     if trans_range is None:
#         scale_factor = 0.05 * data_scale['spatial_scale']
#         trans_range = (-scale_factor, scale_factor)
#
#     print(f"自适应参数设置: 匹配阈值={match_threshold:.4f}, 平移范围=[{trans_range[0]:.4f}, {trans_range[1]:.4f}]")
#
#     # 动态创建参数空间
#     dimensions = []
#
#     # 根据维度添加旋转参数
#     if D == 2:
#         # 2D: 单个旋转角度
#         dimensions.append(Real(low=angle_range[0], high=angle_range[1], name='rot_z'))
#     elif D == 3:
#         # 3D: 三个欧拉角
#         dimensions.append(Real(low=angle_range[0], high=angle_range[1], name='rot_x'))
#         dimensions.append(Real(low=angle_range[0], high=angle_range[1], name='rot_y'))
#         dimensions.append(Real(low=angle_range[0], high=angle_range[1], name='rot_z'))
#     elif D >= 4:
#         # 4D+: 使用旋转向量 (角度+轴) 或只优化平移
#         print(f"警告: {D}D空间使用简化旋转表示")
#         # 添加一个全局旋转角度
#         dimensions.append(Real(low=angle_range[0], high=angle_range[1], name='rotation_angle'))
#
#     # 添加平移参数 (每个维度一个)
#     for i in range(D):
#         dimensions.append(Real(low=trans_range[0], high=trans_range[1], name=f'trans_{i}'))
#
#     # 缓存机制避免重复计算
#     cache = {}
#
#     def objective(params):
#         """目标函数: 计算当前变换下的匹配分数"""
#         # 尝试从缓存中获取结果
#         params_key = tuple(params)
#         if params_key in cache:
#             return cache[params_key]
#
#         # 解析参数 - 根据维度确定旋转参数数量
#         if D == 2:
#             rot_params = [params[0]]
#             trans_params = params[1:]
#         elif D == 3:
#             rot_params = params[:3]
#             trans_params = params[3:]
#         else:  # D >= 4
#             rot_params = [params[0]]  # 只使用一个旋转角度
#             trans_params = params[1:]
#
#         # 构建扰动变换矩阵
#         M_perturb = create_perturbation_matrix(
#             torch.tensor(rot_params, device=device, dtype=dtype),
#             torch.tensor(trans_params, device=device, dtype=dtype),
#             D,
#             device,
#             dtype
#         )
#
#         # 组合变换: M_perturb * M_init
#         M_current = M_perturb @ M_init
#
#         # 应用变换
#         X_A_trans = transform_points(X_A, M_current)
#
#         # 计算匹配分数
#         score = compute_matching_score(
#             X_A_trans, X_B,
#             match_threshold,
#             exp_P, alpha, similarity_mode
#         )
#
#         # 贝叶斯优化最小化目标，所以我们返回负分数
#         result = -score.item()
#
#         # 缓存结果
#         cache[params_key] = result
#         return result
#
#     # 初始点 (零扰动)
#     x0 = [0.0] * len(dimensions)
#
#     # 运行贝叶斯优化
#     res = gp_minimize(
#         func=objective,
#         dimensions=dimensions,
#         n_calls=n_calls,
#         x0=x0,
#         random_state=42,
#         n_jobs=1,  # 单线程以避免GPU冲突
#         verbose=False
#     )
#
#     # 提取最优参数
#     best_params = res.x
#
#     # 解析最优参数 - 与objective函数一致
#     if D == 2:
#         best_rot = [best_params[0]]
#         best_trans = torch.tensor(best_params[1:], device=device, dtype=dtype)
#     elif D == 3:
#         best_rot = best_params[:3]
#         best_trans = torch.tensor(best_params[3:], device=device, dtype=dtype)
#     else:  # D >= 4
#         best_rot = [best_params[0]]
#         best_trans = torch.tensor(best_params[1:], device=device, dtype=dtype)
#
#     # 构建最优变换矩阵
#     M_perturb = create_perturbation_matrix(
#         torch.tensor(best_rot, device=device, dtype=dtype),
#         best_trans,
#         D,
#         device,
#         dtype
#     )
#     M_best = M_perturb @ M_init
#
#     # 计算最优分数
#     X_A_trans_best = transform_points(X_A, M_best)
#     best_score = compute_matching_score(
#         X_A_trans_best, X_B,
#         match_threshold,
#         exp_P, alpha, similarity_mode
#     ).item()
#
#     return M_best, best_score

# 假设以下的库和辅助函数已经存在
# from skopt import gp_minimize
# from skopt.space import Real
# import torch
# def calculate_data_scale(X_A, X_B): ...
# def create_perturbation_matrix(rot_params, trans_params, D, device, dtype): ...
# def transform_points(X_A, M): ...
# def compute_matching_score(X_A_trans, X_B, threshold, exp_P, alpha, mode): ...


def maximize_matches_Bayesian(X_A, X_B, M_init,
                                        match_threshold=None, angle_range=(-25, 25),
                                        trans_range=None, n_calls=100, exp_P=None,
                                        alpha=0.5, similarity_mode="probabilistic",
                                        # --- 用于迭代优化的新参数 ---
                                        max_rounds=2, edge_threshold=0.95):
    """
    使用迭代式贝叶斯优化来微调变换，最大化匹配点对的概率。
    如果某一轮优化的最优解位于搜索空间的边界附近，则会以该解为基础，
    启动新一轮的优化。
    支持2D、3D、4D及更高维度的空间数据。

    参数:
        (原始参数保持不变)
        ...
        max_rounds: 优化的最大轮数。
        edge_threshold: 阈值 (0-1)，用于判断一个参数是否“在边界上”。
                        例如，0.95表示如果最优值处于其范围的顶部5%或底部5%内，
                        则视为在边界上。
    """
    device = X_A.device
    dtype = X_A.dtype
    D = X_A.shape[1]  # 空间维度

    # --- 为迭代循环进行初始化 ---
    M_current_init = M_init.to(device)
    current_round = 0
    best_score_overall = -float('inf')
    M_best_overall = M_current_init

    while current_round < max_rounds:
        print(f"\n--- 开始优化第 {current_round + 1}/{max_rounds} 轮 ---")

        # 当前轮次的目标函数所使用的 M_init 是上一轮的优化结果，
        # 或者在第一轮时是用户传入的初始矩阵。
        _M_init_round = M_current_init

        # 在所有轮次中保持参数（如阈值和范围）的一致性
        # 也可以重新计算，但保持一致性更简单、更稳健。
        data_scale = calculate_data_scale(X_A, X_B)
        if match_threshold is None:
            _match_threshold = 1.5 * data_scale['median_dist']
        else:
            _match_threshold = match_threshold

        if trans_range is None:
            scale_factor = 0.05 * data_scale['spatial_scale']
            _trans_range = (-scale_factor, scale_factor)
        else:
            _trans_range = trans_range

        if current_round == 0:
            print(
                f"自适应参数设置: 匹配阈值={_match_threshold:.4f}, 平移范围=[{_trans_range[0]:.4f}, {_trans_range[1]:.4f}]")

        # 为skopt定义搜索空间 (dimensions)
        dimensions = []
        if D == 2:
            dimensions.append(Real(low=angle_range[0], high=angle_range[1], name='rot_z'))
        elif D == 3:
            dimensions.append(Real(low=angle_range[0], high=angle_range[1], name='rot_x'))
            dimensions.append(Real(low=angle_range[0], high=angle_range[1], name='rot_y'))
            dimensions.append(Real(low=angle_range[0], high=angle_range[1], name='rot_z'))
        elif D >= 4:
            dimensions.append(Real(low=angle_range[0], high=angle_range[1], name='rotation_angle'))

        for i in range(D):
            dimensions.append(Real(low=_trans_range[0], high=_trans_range[1], name=f'trans_{i}'))

        cache = {}

        def objective(params):
            params_key = tuple(params)
            if params_key in cache:
                return cache[params_key]

            if D == 2:
                rot_params, trans_params = [params[0]], params[1:]
            elif D == 3:
                rot_params, trans_params = params[:3], params[3:]
            else:
                rot_params, trans_params = [params[0]], params[1:]

            M_perturb = create_perturbation_matrix(
                torch.tensor(rot_params, device=device, dtype=dtype),
                torch.tensor(trans_params, device=device, dtype=dtype),
                D, device, dtype
            )
            # 在上一轮结果的基础上应用扰动
            M_current = M_perturb @ _M_init_round
            X_A_trans = transform_points(X_A, M_current)

            score = compute_matching_score(
                X_A_trans, X_B, _match_threshold, exp_P, alpha, similarity_mode
            )
            result = -score.item()
            cache[params_key] = result
            return result

        res = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            x0=[0.0] * len(dimensions),
            random_state=42 + current_round,  # 为每轮优化更改随机种子
            n_jobs=1,
            verbose=False
        )

        best_params_round = res.x

        # --- 判断是否需要进行下一轮的决策逻辑 ---
        is_on_edge = False
        for i, param in enumerate(best_params_round):
            dim = dimensions[i]
            low, high = dim.low, dim.high
            # 检查参数是否在边界阈值内
            if param <= low + (1.0 - edge_threshold) * (high - low) or \
                    param >= low + edge_threshold * (high - low):
                print(f"  -> 警告: 参数 '{dim.name}' ({param:.4f}) 已接近其搜索范围 [{low:.4f}, {high:.4f}] 的边界。")
                is_on_edge = True
                break

        # 构建当前轮次的最佳变换矩阵
        if D == 2:
            best_rot, best_trans = [best_params_round[0]], best_params_round[1:]
        elif D == 3:
            best_rot, best_trans = best_params_round[:3], best_params_round[3:]
        else:
            best_rot, best_trans = [best_params_round[0]], best_params_round[1:]

        M_perturb_best = create_perturbation_matrix(
            torch.tensor(best_rot, device=device, dtype=dtype),
            torch.tensor(best_trans, device=device, dtype=dtype),
            D, device, dtype
        )
        M_best_round = M_perturb_best @ _M_init_round

        # 目标函数返回的是负分，所以这里取反
        best_score_round = -res.fun

        print(f"第 {current_round + 1} 轮结果: 最高分数 = {best_score_round:.6f}")

        # 更新全局找到的最佳结果
        if best_score_round > best_score_overall:
            best_score_overall = best_score_round
            M_best_overall = M_best_round

        current_round += 1

        # 如果在边界上且未达到最大轮数，则继续
        if is_on_edge and current_round < max_rounds:
            print(f"最优解位于边界。准备新一轮优化。")
            # 下一轮的 M_init 就是当前轮次的最佳结果
            M_current_init = M_best_round
        else:
            if is_on_edge:
                print("已达到最大优化轮数。优化结束。")
            else:
                print("最优解已在搜索空间内找到。优化结束。")
            break  # 退出 while 循环

    print("\n--- 迭代式贝叶斯优化完成 ---")
    print(f"最终找到的全局最高分数: {best_score_overall:.6f}")
    return M_best_overall, best_score_overall

import os
import shutil


def copy_py_files(destination_dir):
    # 获取当前目录
    current_dir = os.getcwd()

    # 确保目标目录存在
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print(f"创建目录: {destination_dir}")

    # 遍历当前目录下的所有文件
    for filename in os.listdir(current_dir):
        # 检查是否为.py文件
        if filename.endswith('.py') and os.path.isfile(os.path.join(current_dir, filename)):
            # 构建源文件和目标文件的完整路径
            source_path = os.path.join(current_dir, filename)
            dest_path = os.path.join(destination_dir, filename)

            # 复制文件
            shutil.copy2(source_path, dest_path)
            print(f"已复制: {filename} -> {destination_dir}")

    print("复制完成!")

from sklearn.cluster import KMeans

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist


def KMeans_inducing_points(pos, n_inducing_points=100, min_distance=1e-3):
    """使用KD树优化的诱导点筛选算法（修正版）"""
    # 确保输入是NumPy数组
    if not isinstance(pos, np.ndarray):
        pos = np.array(pos)

    # 如果点数量小于要求，直接返回所有点
    if len(pos) <= n_inducing_points:
        return pos

    # K-means聚类
    kmeans = KMeans(n_clusters=min(n_inducing_points, len(pos)), random_state=42)
    kmeans.fit(pos)
    inducing_points = kmeans.cluster_centers_

    # 如果只有一个点或不需要筛选，直接返回
    if len(inducing_points) <= 1 or min_distance <= 0:
        return inducing_points

    # 构建KD树进行高效近邻搜索
    tree = KDTree(inducing_points)

    # 使用KDTree的query_ball_point方法替代query_radius
    neighbors = tree.query_ball_point(inducing_points, r=min_distance)

    # 贪心算法筛选点 - 更高效的实现
    keep_indices = []
    visited = set()

    for i in range(len(inducing_points)):
        if i not in visited:
            keep_indices.append(i)
            # 标记所有在当前点邻域内的点为已访问
            visited.update(neighbors[i])

    return inducing_points[keep_indices]


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycpd import DeformableRegistration
from scipy.interpolate import griddata
import time


def calculate_displacement_field(source, target, normalize=True, add_noise=True, noise_level=0.005,
                                 max_iterations=100, beta=5, lam=1e-3):
    """
    使用CPD算法计算从源点云到目标点云的位移场

    参数:
    source: 源点云，形状为 (N, D)
    target: 目标点云，形状为 (M, D)
    normalize: 是否对坐标进行归一化处理
    add_noise: 是否添加随机微扰
    noise_level: 微扰的幅度，相对于点云范围的比例
    max_iterations: CPD算法的最大迭代次数
    beta: 控制变形场的平滑性（值越大，变形越平滑）
    lam: 正则化系数，平衡拟合精度与平滑度
    """
    start_time = time.time()

    # 保存原始点云用于后续恢复
    # original_source = source.copy()
    # original_target = target.copy()

    # 1. 归一化处理（可选）
    if normalize:
        # 计算源点云的统计信息
        source_mean = np.mean(source, axis=0)
        source_std = np.std(source, axis=0)

        # 归一化源点云和目标点云
        source = (source - source_mean) / (source_std + 1e-8)
        target = (target - source_mean) / (source_std + 1e-8)

    # 2. 添加随机微扰（可选）
    if add_noise:
        # 计算点云的范围
        source_range = np.max(source, axis=0) - np.min(source, axis=0)

        # 添加高斯噪声
        noise = np.random.normal(0, noise_level * source_range, source.shape)
        source = source + noise

    # 3. 创建并运行CPD配准
    reg = DeformableRegistration(X=target, Y=source, max_iterations=max_iterations, beta=beta, lam=lam)
    deformed_source, (_, _) = reg.register()

    # 4. 计算位移矢量
    displacement = deformed_source - source

    # 5. 如果进行了归一化，恢复原始坐标
    if normalize:
        deformed_source = deformed_source * (source_std + 1e-8) + source_mean
        displacement = displacement * (source_std + 1e-8)

    # 计算计算时间
    elapsed_time = time.time() - start_time
    print(f"CPD配准完成，耗时: {elapsed_time:.2f}秒")

    return displacement, deformed_source


import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
import scanpy as sc

import numpy as np
from sklearn.cluster import MiniBatchKMeans  # 更高效的聚类算法
from scipy.spatial import KDTree
import scanpy as sc
from tqdm import tqdm  # 进度条


def cluster_downsample_adata(adata, spatial_key='spatial', target_size=10000):
    """
    对adata对象使用聚类中心法进行下采样

    参数:
        adata: AnnData对象
        spatial_key: 空间坐标在obsm中的键名，默认为'spatial'
        target_size: 目标采样点数，默认为10000

    返回:
        下采样后的AnnData对象
    """
    # 1. 输入验证
    if spatial_key not in adata.obsm:
        raise ValueError(f"空间坐标键 '{spatial_key}' 不存在于 adata.obsm 中")

    coords = adata.obsm[spatial_key]
    if coords.ndim != 2:
        raise ValueError("空间坐标应为二维数组")

    n_cells = adata.shape[0]
    if n_cells <= target_size:
        print(f"数据点数量 ({n_cells}) 已小于等于目标数量 ({target_size})，无需下采样")
        return adata.copy()  # 返回副本以保持原始数据不变

    print(f"开始下采样: {n_cells} -> {target_size}")

    # 2. 使用更高效的MiniBatchKMeans
    print(f"执行MiniBatchKMeans聚类，聚类数: {target_size}")
    kmeans = MiniBatchKMeans(
        n_clusters=target_size,
        random_state=42,
        batch_size=min(1000, n_cells // 10)  # 自适应批次大小
    )
    kmeans.fit(coords)
    cluster_centers = kmeans.cluster_centers_

    # 3. 批量查询最近邻
    print("查找最近邻点...")
    kdtree = KDTree(coords)

    # 使用批量查询提高效率
    _, nearest_indices = kdtree.query(cluster_centers, k=1, workers=-1)  # 使用所有CPU核心

    # 4. 处理重复点并补充
    unique_indices = np.unique(nearest_indices)
    selected_indices = unique_indices

    if len(unique_indices) < target_size:
        print(f"发现 {len(unique_indices)} 个唯一索引，补充 {target_size - len(unique_indices)} 个点")
        remaining_indices = np.setdiff1d(np.arange(n_cells), unique_indices)

        # 确保有足够的点可以补充
        n_needed = min(len(remaining_indices), target_size - len(unique_indices))

        additional_indices = np.random.choice(
            remaining_indices,
            size=n_needed,
            replace=False
        )
        selected_indices = np.concatenate([unique_indices, additional_indices])
    elif len(unique_indices) > target_size:
        # 理论上不会发生，但添加保护
        selected_indices = unique_indices[:target_size]

    # 5. 创建下采样后的adata
    downsampled_adata = adata[selected_indices].copy()
    print(f"下采样完成: {n_cells} -> {len(downsampled_adata)}")

    # 可选：添加下采样信息到obs
    downsampled_adata.obs['downsampled'] = True

    return downsampled_adata


import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import scanpy as sc


def fast_spatial_downsample(adata, spatial_key='spatial',spatial_dim=2, target_size=10000, grid_strategy='adaptive'):
    """
    高效的空间下采样方法
    参数:
        adata: AnnData对象
        spatial_key: 空间坐标在obsm中的键名
        target_size: 目标采样点数
        grid_strategy: 网格策略 ('auto', 'fixed', 'adaptive')

    返回:
        下采样后的AnnData对象
    """
    # 1. 输入验证
    if spatial_key not in adata.obsm:
        raise ValueError(f"空间坐标键 '{spatial_key}' 不存在于 adata.obsm 中")

    coords = adata.obsm[spatial_key]
    if coords.shape[-1] != spatial_dim:
        coords_old = coords
        coords = coords[:,:spatial_dim]
    else:
        coords_old = coords
        coords = coords

    n_cells = adata.shape[0]
    if n_cells <= target_size:
        print(f"数据点数量 ({n_cells}) ≤ 目标数量 ({target_size})，无需下采样")
        return adata

    print(f"开始高效下采样: {n_cells} -> {target_size}")

    # 2. 使用网格化方法进行初始采样
    if grid_strategy == 'auto':
        # 根据数据规模自动选择策略
        grid_strategy = 'adaptive' if n_cells > 1000000 else 'fixed'

    if grid_strategy == 'fixed':
        # 固定网格大小
        grid_size = int(np.sqrt(target_size) * 2)
        sampled_indices = grid_downsample(coords, grid_size)
    else:
        # 自适应网格大小
        sampled_indices = adaptive_grid_downsample(coords, target_size)

        # 3. 如果网格采样点数不足，补充随机采样
    if len(sampled_indices) < target_size:
        remaining_indices = np.setdiff1d(np.arange(n_cells), sampled_indices)
        n_needed = min(len(remaining_indices), target_size - len(sampled_indices))
        additional_indices = np.random.choice(remaining_indices, n_needed, replace=False)
        sampled_indices = np.concatenate([sampled_indices, additional_indices])

    # 4. 创建下采样后的adata
    # 创建一个空的AnnData容器（最快方式）
    downsampled_adata = anndata.AnnData(
        X=None,  # 先不填充数据
        obs=adata[sampled_indices].obs.copy(),  # 只复制需要的obs列
        var=adata.var.copy(),  # 复制所有var数据
        obsm={
            'spatial': adata[sampled_indices].obsm['spatial'].copy(),  # 复制需要的obsm
            'mask': adata[sampled_indices].obsm['mask'].copy()
        },
        layers={
            'rawX': adata[sampled_indices].layers['rawX'].copy()  # 复制需要的layers
        }
    )
    downsampled_adata.X = adata[sampled_indices].X.copy()
    print(f"下采样完成: {n_cells} -> {len(downsampled_adata)}")
    return downsampled_adata


def grid_downsample(coords, grid_size):
    """使用固定网格进行下采样"""
    # 1. 计算空间范围
    min_vals = np.min(coords, axis=0)
    max_vals = np.max(coords, axis=0)

    # 2. 创建网格
    grid_x = np.linspace(min_vals[0], max_vals[0], grid_size)
    grid_y = np.linspace(min_vals[1], max_vals[1], grid_size)

    # 3. 为每个点分配网格ID
    grid_ids = pd.cut(coords[:, 0], bins=grid_x, labels=False, include_lowest=True)
    grid_ids = grid_ids.astype(str) + "_" + pd.cut(coords[:, 1], bins=grid_y, labels=False, include_lowest=True).astype(
        str)

    # 4. 从每个网格单元中随机选择一个点
    unique_grids = np.unique(grid_ids)
    sampled_indices = []

    for grid_id in unique_grids:
        cell_indices = np.where(grid_ids == grid_id)[0]
        if len(cell_indices) > 0:
            sampled_indices.append(np.random.choice(cell_indices))

    return np.array(sampled_indices)


def adaptive_grid_downsample(coords, target_size):
    """使用自适应网格进行下采样"""
    # 1. 计算初始网格大小
    n_cells = coords.shape[0]
    grid_size = int(np.sqrt(target_size))

    # 2. 创建初始网格
    min_vals = np.min(coords, axis=0)
    max_vals = np.max(coords, axis=0)

    # 3. 递归细分高密度区域
    sampled_indices = []
    queue = [(min_vals, max_vals)]

    while queue and len(sampled_indices) < target_size:
        current_min, current_max = queue.pop(0)

        # 在当前区域内选择点
        in_region = np.all((coords >= current_min) & (coords <= current_max), axis=1)
        region_indices = np.where(in_region)[0]

        if len(region_indices) == 0:
            continue

        # 如果区域内点数少，直接随机选一个
        if len(region_indices) <= max(1, target_size // (grid_size * 2)):
            sampled_indices.append(np.random.choice(region_indices))
            continue

        # 否则细分区域
        mid_x = (current_min[0] + current_max[0]) / 2
        mid_y = (current_min[1] + current_max[1]) / 2

        # 创建四个子区域
        sub_regions = [
            (current_min, [mid_x, mid_y]),
            ([current_min[0], mid_y], [mid_x, current_max[1]]),
            ([mid_x, current_min[1]], [current_max[0], mid_y]),
            ([mid_x, mid_y], current_max)
        ]

        # 打乱子区域顺序以避免偏向特定区域
        np.random.shuffle(sub_regions)
        queue.extend(sub_regions)

    return np.array(sampled_indices)


# 假设 clean_adata 函数已定义
# from anndata import AnnData
# def clean_adata(adata: AnnData):
#     # 这是一个示例，您可能需要根据实际情况调整
#     if 'highly_variable' in adata.var:
#         del adata.var['highly_variable']
#     return adata

def concat_adata(adata_list):
    """
    An optimized version to address the O(n²) performance issue.
    """
    if not adata_list:
        raise ValueError("The input adata_list cannot be empty.")

    if len(adata_list) == 1:
        return adata_list[0].copy()

    print("Starting data concatenation...")

    # --- Optimization 1: Quickly determine common genes ---
    common_genes = set(adata_list[0].var_names)
    for adata in adata_list[1:]:
        common_genes.intersection_update(adata.var_names)

    print(f"Found {len(common_genes)} common genes")

    # --- Optimization 2: Use dictionary mapping to avoid O(n²) lookup ---
    # Build a gene name to index map for the reference adata
    reference_adata = adata_list[0]
    gene_to_index_map = {gene: idx for idx, gene in enumerate(reference_adata.var_names)}

    # Keep only the common genes, preserving the original order
    ordered_common_genes = [gene for gene in reference_adata.var_names if gene in common_genes]

    # Pre-compute indices (O(n) time complexity)
    reference_indices = [gene_to_index_map[gene] for gene in ordered_common_genes]

    # --- Optimization 3: Process datasets in a loop with O(n) slicing ---
    filtered_adata_list = []

    for i, adata in enumerate(adata_list):
        print(f"Processing dataset {i + 1}/{len(adata_list)}...")

        # Build a map for the current dataset
        current_gene_map = {gene: idx for idx, gene in enumerate(adata.var_names)}

        # Quickly get indices (O(n) instead of O(n²))
        gene_indices = [current_gene_map[gene] for gene in ordered_common_genes]

        # Slice and maintain gene order
        filtered_adata = adata[:, gene_indices].copy()
        filtered_adata.var_names = ordered_common_genes  # Ensure consistent order

        filtered_adata_list.append(filtered_adata)

    # --- Optimization 4: Use a more efficient concatenation method ---
    print("Starting dataset concatenation...")

    # Method 1: Use anndata.concat (recommended)
    try:
        import anndata
        combined_adata = anndata.concat(
            filtered_adata_list,
            join='inner',
            label='batch',
            keys=[f"batch_{i}" for i in range(len(filtered_adata_list))],
            index_unique='-'
        )
    except Exception:
        # Method 2: Fallback to the original concatenate method
        if len(filtered_adata_list) > 1:
            combined_adata = filtered_adata_list[0].concatenate(
                *filtered_adata_list[1:],
                join='inner',
                index_unique='-',
                batch_categories=[f"batch_{i}" for i in range(len(filtered_adata_list))]
            )
        else:
            combined_adata = filtered_adata_list[0]

    print("Data concatenation complete!")
    return combined_adata


def concat_adata_img(adata_list):
    """
    Concatenation function optimized specifically for IMG data.
    """
    if not adata_list:
        raise ValueError("The input adata_list cannot be empty.")

    if len(adata_list) == 1:
        return adata_list[0].copy()

    print("Concatenating IMG data...")

    # For IMG data, the gene order is usually identical across all datasets.
    # We validate this first and perform a fast concatenation.
    first_genes = adata_list[0].var_names
    first_shape = adata_list[0].shape

    # Quickly check if all datasets have the same genes and shape
    compatible = True
    for i, adata in enumerate(adata_list[1:], 1):
        if not np.array_equal(adata.var_names, first_genes) or adata.shape[1] != first_shape[1]:
            compatible = False
            print(f"Warning: Dataset {i} has incompatible genes. Using standard concatenation.")
            break

    if compatible:
        print("All IMG datasets are compatible. Using fast concatenation...")
        try:
            # Use anndata.concat for fast concatenation
            import anndata
            return anndata.concat(
                adata_list,
                join='inner',
                label='batch',
                keys=[f"batch_{i}" for i in range(len(adata_list))],
                index_unique='-'
            )
        except Exception as e:
            print(f"anndata.concat failed: {e}. Using fallback method.")

    # If not compatible, use the standard, optimized method
    return concat_adata(adata_list)

def clean_adata(adata):
    """清理可能导致合并问题的数据结构"""
    # 删除所有邻居图信息
    for key in list(adata.obsp.keys()):
        del adata.obsp[key]
    for key in list(adata.uns.keys()):
        del adata.uns[key]

    # 检查并清理可能的问题维度
    for key in list(adata.obsm.keys()):
        if len(adata.obsm[key].shape) < 2:
            print(f"Removing problematic obsm['{key}'] with shape {adata.obsm[key].shape}")
            del adata.obsm[key]

    # 清理 layers
    for key in list(adata.layers.keys()):
        if len(adata.layers[key].shape) < 2:
            print(f"Removing problematic layers['{key}'] with shape {adata.layers[key].shape}")
            del adata.layers[key]

    return adata


import pandas as pd  # 需要导入 pandas


def get_concatenated_tensor(adata_list, slices, dtype, data_extractor):
    """
    一个辅助函数，用于高效地从 adata_list 中提取数据并拼接成一个 Tensor。
    此版本已更新，可以自动处理 pandas.Series 类型。

    参数:
        data_extractor: 一个lambda函数，定义如何从单个adata对象中提取数据, e.g.,
                        lambda adata: adata.obsm['spatial']
    """
    tensor_list = []
    for s in slices:
        # 从 anndata 对象中提取数据
        extracted_data = data_extractor(adata_list[s])

        # --- 新增的改造逻辑 ---
        # 检查提取出的数据是否为 Series 类型，如果是，则转换为 numpy array
        if isinstance(extracted_data, pd.Series) or isinstance(extracted_data, pd.DataFrame):
            numpy_data = extracted_data.values
        else:
            numpy_data = extracted_data
        tensor_list.append(torch.from_numpy(numpy_data).to(dtype))
    return torch.cat(tensor_list, dim=0)

def auto_batch_size(N_train_all, dim=2):
    if N_train_all <= 1024 * 1:
        batch_size = 128
    elif N_train_all <= 1024 * 2:
        batch_size = 256
    elif N_train_all <= 1024 * 8:
        batch_size = 512
    elif N_train_all <= 2048 * 16 or dim > 2:
        batch_size = 1024
    else:
        batch_size = 2048
    print(f'Batch size: {batch_size}')
    return batch_size

import math
def get_cosine_schedule_with_warmup(current_epoch, warmup_epochs, total_epochs, start_epoch=0, final_lr_scale=0.05):
    """
    创建一个带有预热期的余弦学习率调度函数。
    """
    assert final_lr_scale < 1.0
    current_epoch = current_epoch - start_epoch
    total_epochs = total_epochs - start_epoch
    if current_epoch>total_epochs:
        current_epoch = total_epochs
    if current_epoch < 0:
        return 1.0
    if current_epoch < warmup_epochs:
        # 线性预热
        return float(current_epoch) / float(max(1, warmup_epochs)) * 5
    else:
        # 余弦衰减
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # 从1.0衰减到final_lr_scale
        return final_lr_scale + (1.0 - final_lr_scale) * cosine_decay


def clean_metadata_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    A robust function to clean AnnData's metadata DataFrames (obs or var).
    - Fills all NaN values in numeric columns with 0.
    - Fills all NaN values in non-numeric columns (object, string, categorical) with 'NA'.
    - Handles 'Categorical' data types correctly to avoid errors.

    Args:
        df: A pandas DataFrame, e.g., adata.obs or adata.var.

    Returns:
        The cleaned pandas DataFrame.
    """
    print(f"Starting to clean DataFrame with {df.shape[1]} columns...")

    # Iterate through each column of the DataFrame
    for col in df.columns:
        column_series = df[col]

        # 1. Use pd.api.types.is_numeric_dtype for a robust check of numeric types
        # This covers all numeric types like float64, float32, int64, int32, etc.
        if pd.api.types.is_numeric_dtype(column_series):
            # If it's a numeric type, fill with 0
            df[col] = column_series.fillna(0)
        else:
            # 2. For all non-numeric types

            # 2a. First, handle potential issues with Categorical types
            if pd.api.types.is_categorical_dtype(column_series):
                # If 'NA' is not yet a valid category, add it
                if 'NA' not in column_series.cat.categories:
                    # .cat.add_categories() returns a new Series, so we reassign it
                    df[col] = column_series.cat.add_categories(['NA'])

            # 2b. Now, it's safe to fill all non-numeric types (including the now-prepared Categorical) with 'NA'
            df[col] = df[col].fillna('NA')

    print("Cleaning complete.")
    return df


import scanpy as sc
import pandas as pd
import numpy as np
import re


def split_rna_atac(adata: sc.AnnData, peak_regex: str = r'^(chr)?[\w]+:\d+-\d+$'):
    """
    根据 var_names 将包含 RNA 和 ATAC 数据的 AnnData 对象拆分为两个。

    这个函数假设 ATAC peaks 的名称遵循 'chr:start-end' 格式，而基因名则不遵循。

    Args:
        adata (sc.AnnData): 包含混合数据的 AnnData 对象。
        peak_regex (str): 用于识别 ATAC peak 名称的正则表达式。

    Returns:
        (sc.AnnData, sc.AnnData):
        一个元组，包含两个 AnnData 对象：(adata_rna, adata_atac)。
    """
    print(f"原始 AnnData 对象维度: {adata.shape}")

    # 1. 创建一个布尔掩码来识别 ATAC peaks
    # .str.match() 会对 var_names 中的每个名字应用正则表达式
    # 返回一个布尔值的 Pandas Series
    is_atac_mask = adata.var_names.str.match(peak_regex)

    # 检查是否有任何匹配项
    n_atac_features = np.sum(is_atac_mask)
    if n_atac_features == 0:
        raise ValueError("错误：根据提供的正则表达式，未在 var_names 中找到任何 ATAC peak。请检查您的数据或正则表达式。")

    print(f"识别到 {n_atac_features} 个 ATAC peaks。")
    print(f"识别到 {adata.shape[1] - n_atac_features} 个 RNA 基因。")

    # 2. 使用掩码进行切片
    # anndata[:, is_atac_mask] 选择所有细胞和所有为 True 的特征 (ATAC)
    adata_atac = adata[:, is_atac_mask].copy()

    # anndata[:, ~is_atac_mask] 选择所有细胞和所有为 False 的特征 (RNA)
    # `~` 符号是布尔取反操作
    adata_rna = adata[:, ~is_atac_mask].copy()

    # 3. 为新的 AnnData 对象添加描述信息 (可选但推荐)
    adata_rna.uns['modality'] = 'RNA'
    adata_atac.uns['modality'] = 'ATAC'

    print("-" * 30)
    print(f"拆分后 RNA AnnData 维度: {adata_rna.shape}")
    print(f"拆分后 ATAC AnnData 维度: {adata_atac.shape}")

    return adata_rna, adata_atac


def preprocessing_atac(
        adata,
        min_genes=None,
        min_cells=0.01,
        n_top_genes=30000,
        target_sum=None,
        log=None
):
    """
    preprocessing
    """
    print('Raw dataset shape: {}'.format(adata.shape))
    if log: log.info('Preprocessing')
    adata.X[adata.X > 0] = 1
    if log: log.info('Filtering cells')
    if min_genes:
        sc.pp.filter_cells(adata, min_genes=min_genes)
    if log: log.info('Filtering genes')
    if min_cells:
        if min_cells < 1:
            min_cells = min_cells * adata.shape[0]
        sc.pp.filter_genes(adata, min_cells=min_cells)
    if n_top_genes:
        if log: log.info('Finding variable features')
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, inplace=False, subset=True)
    if log: log.info('Batch specific maxabs scaling')
    print('Processed dataset shape: {}'.format(adata.shape))
    return adata


import numpy as np
import torch
import scanpy as sc
import pandas as pd  # 确保导入 pandas
import numpy as np
import torch
import pandas as pd  # 确保导入 pandas


def filter_expression_data(exp_A, exp_B, X_A, X_B):
    """
    对输入的表达和坐标数据进行严格的预过滤。
    - 细胞过滤: 保留各自样本中表达量总和 > 0 的细胞。
    - 基因过滤: 仅保留在两个样本中表达量总和都 > 0 的共同基因。
    """
    print("Pre-filtering data with strict common gene policy...")

    # 1. 确保所有数据都为 Numpy 数组 (逻辑不变)
    if isinstance(exp_A, torch.Tensor):
        exp_A = exp_A.cpu().detach().numpy()
        X_A = X_A.cpu().detach().numpy()
    if isinstance(exp_B, torch.Tensor):
        exp_B = exp_B.cpu().detach().numpy()
        X_B = X_B.cpu().detach().numpy()

    # 2. 独立过滤每个样本中“表达量总和为0”的细胞 (逻辑不变)
    sums_per_cell_A = exp_A.sum(axis=1)
    sums_per_cell_B = exp_B.sum(axis=1)
    keep_cells_mask_A = sums_per_cell_A > 0
    keep_cells_mask_B = sums_per_cell_B > 0

    n_obs_A_before, n_obs_B_before = exp_A.shape[0], exp_B.shape[0]

    exp_A = exp_A[keep_cells_mask_A, :]
    X_A = X_A[keep_cells_mask_A, :]
    exp_B = exp_B[keep_cells_mask_B, :]
    X_B = X_B[keep_cells_mask_B, :]

    print(f"Filtered cells in A (sum > 0): {n_obs_A_before} -> {exp_A.shape[0]}")
    print(f"Filtered cells in B (sum > 0): {n_obs_B_before} -> {exp_B.shape[0]}")

    # 确保两个矩阵在进行基因过滤前有相同的基因数
    if exp_A.shape[1] != exp_B.shape[1]:
        raise ValueError("Expression matrices must have the same number of genes before gene filtering.")

    n_vars_before = exp_A.shape[1]

    # 3. <<< 核心逻辑修改：寻找共同的优质基因 >>>
    # 分别计算每个样本中每个基因的总和
    sums_per_gene_A = exp_A.sum(axis=0)
    sums_per_gene_B = exp_B.sum(axis=0)

    # 分别创建布尔掩码
    keep_genes_mask_A = sums_per_gene_A > 0
    keep_genes_mask_B = sums_per_gene_B > 0

    # 使用逻辑“与”(&)操作找到必须同时满足两个条件的基因
    # 这就是“交集”操作，确保基因在两个样本中都有表达
    final_keep_genes_mask = keep_genes_mask_A & keep_genes_mask_B

    # 4. 使用共同的基因掩码来过滤两个样本的表达矩阵
    exp_A = exp_A[:, final_keep_genes_mask]
    exp_B = exp_B[:, final_keep_genes_mask]

    print(f"Filtered genes (sum > 0 in BOTH samples): {n_vars_before} -> {exp_A.shape[1]}")

    return exp_A, exp_B, X_A, X_B


import numpy as np
import torch
from typing import Tuple, Union


def flattened_to_simg(
        flattened_data: Union[np.ndarray, torch.Tensor],
        shape: Tuple[int, int, int]
) -> Union[np.ndarray, torch.Tensor]:
    """
    将展平的2D图像数据恢复为4D的simg张量 (N, C, H, W)。

    Args:
        flattened_data (Union[np.ndarray, torch.Tensor]):
            展平的图像数据，形状为 (n_samples, n_features)，其中 n_features = C * H * W。
        shape (Tuple[int, int, int]):
            目标图像的形状，格式为 (C, H, W)，例如 (3, 32, 32)。

    Returns:
        Union[np.ndarray, torch.Tensor]:
            恢复后的4D图像数据，形状为 (n_samples, C, H, W)。

    Raises:
        ValueError: 如果展平数据的特征数量与目标形状不匹配。
    """
    n_features = flattened_data.shape[1]
    expected_features = shape[0] * shape[1] * shape[2]

    if n_features != expected_features:
        raise ValueError(
            f"特征数量不匹配。展平数据有 {n_features} 个特征, "
            f"但目标形状 {shape} 需要 {expected_features} 个特征。"
        )

    # -1 会自动推断样本数量 (n_samples)
    return flattened_data.reshape(-1, *shape)


def simg_to_flattened(
        simg_data: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    将4D的simg图像张量 (N, C, H, W) 展平为2D数据 (N, C*H*W)。

    Args:
        simg_data (Union[np.ndarray, torch.Tensor]):
            4D图像数据，形状为 (n_samples, C, H, W)。

    Returns:
        Union[np.ndarray, torch.Tensor]:
            展平后的2D数据，形状为 (n_samples, C * H * W)。

    Raises:
        ValueError: 如果输入数据的维度不为4。
    """
    if simg_data.ndim != 4:
        raise ValueError(
            f"输入数据必须是4维的 (N, C, H, W)，但收到了 {simg_data.ndim} 维的数据。"
        )

    # simg_data.shape[0] 是样本数量 (N)
    # -1 会自动计算 C * H * W 的乘积
    return simg_data.reshape(simg_data.shape[0], -1)


import pandas as pd
import numpy as np
import anndata as ad
from PIL import Image
from tqdm import tqdm
from typing import Tuple, List, Union

#
# def _crop_and_flatten_patches(
#         full_img: np.ndarray,
#         coords: np.ndarray,
#         patch_size: int
# ) -> Tuple[np.ndarray, Tuple[int, int, int]]:
#     """
#     一个辅助函数，用于从全尺寸图像中裁剪、转置和展平图像块。
#
#     Args:
#         full_img (np.ndarray): 全尺寸组织图像 (H, W, C)。
#         coords (np.ndarray): 细胞/斑点的中心坐标 (N, 2)，格式为 (y, x)。
#         patch_size (int): 每个图像块的正方形边长。
#
#     Returns:
#         Tuple[np.ndarray, Tuple[int, int, int]]:
#         - flattened_patches (np.ndarray): 展平后的图像块数据 (N, C*H*W)。
#         - patch_shape (Tuple[int, int, int]): 单个图像块的形状 (C, H, W)。
#     """
#     if patch_size % 2 != 0:
#         raise ValueError("patch_size 必须是偶数。")
#
#     n_channels = full_img.shape[2]
#     patch_shape = (n_channels, patch_size, patch_size)
#     flattened_dim = n_channels * patch_size * patch_size
#
#     # 为了安全地在图像边缘进行裁剪，我们先对图像进行零填充
#     pad_width = patch_size//2
#     img_padded = np.pad(
#         full_img,
#         pad_width=((patch_size, patch_size), (patch_size, patch_size), (0, 0)),
#         mode='constant',
#         constant_values=0
#     )
#
#     # 调整坐标以适应填充后的图像
#     coords_padded = coords + patch_size
#
#     flattened_patches = np.zeros((len(coords), flattened_dim), dtype=np.float32)
#
#     patch_transposed_list = np.zeros((len(coords), 3, patch_size, patch_size), dtype=np.float32)
#
#     print(f"正在从 {len(coords)} 个坐标点裁剪 {patch_size}x{patch_size} 的图像块...")
#     for i, (y, x) in tqdm(enumerate(coords_padded), total=len(coords)):
#         # 将坐标转换为整数索引
#         y, x = int(y), int(x)
#
#         # 定义裁剪区域
#         y_start, y_end = y - pad_width, y + pad_width
#         x_start, x_end = x - pad_width, x + pad_width
#
#         # 裁剪图像块
#         patch = img_padded[y_start:y_end, x_start:x_end, :]
#
#         # 转置以匹配 (C, H, W) 格式
#         patch_transposed = patch.transpose(2, 0, 1)
#         patch_transposed_list[i] = patch_transposed
#         # 展平并存储
#         flattened_patches[i] = patch_transposed.flatten()
#
#     return flattened_patches, coords_padded, patch_shape,img_padded
#
#
# def create_img_adata_from_data(
#         full_img: np.ndarray,
#         barcodes: np.ndarray,
#         img_coordinates: np.ndarray,
#         spatial_coords: np.ndarray = None,
#         patch_size: int = 32
# ) -> ad.AnnData:
#     """
#     从已经读取到内存的图像和位置数据构建一个以图像特征为X的AnnData对象。
#     Args:
#         full_img (np.ndarray): 全尺寸组织图像的Numpy数组, 形状为 (H, W, C)。
#         positions_df (pd.DataFrame):
#             组织位置的DataFrame，结构应与 'tissue_positions_list.csv' 类似。
#             函数会假定：
#             - 第0列是细胞/斑点的barcode。
#             - 第4列是y坐标 (pixel coordinate)。
#             - 第5列是x坐标 (pixel coordinate)。
#         patch_size (int): 每个图像块的边长。默认为32。
#
#     Returns:
#         ad.AnnData: 构建好的AnnData对象。
#     """
#     assert barcodes.shape[0] == img_coordinates.shape[0]
#     assert img_coordinates.shape[1] == 2
#     if not spatial_coords is None:
#         assert spatial_coords.shape[0] == barcodes.shape[0]
#     print("\n--- 步骤 2/3: 裁剪并展平图像块 ---")
#     # 调用辅助函数获取特征矩阵 X 和图像块的形状
#     feature_matrix, coords_padded, patch_shape, img_padded = _crop_and_flatten_patches(
#         full_img = full_img,
#         coords=img_coordinates,
#         patch_size=patch_size
#     )
#     print("\n--- 步骤 3/3: 构建AnnData对象 ---")
#     # 创建AnnData对象
#     adata = ad.AnnData(X=feature_matrix)
#     # 填充 observation (obs) 信息
#     adata.obs_names = barcodes
#     # 填充 variable (var) 信息
#     adata.var['mode'] = 'IMG'
#     adata.var_names = [f'pixel_{i}' for i in range(feature_matrix.shape[1])]
#     # 填充 obsm 信息
#     if not spatial_coords is None:
#         adata.obsm['spatial'] = spatial_coords
#     adata.obsm['img_coordinates'] = coords_padded
#     adata.obsm['IMG_Shape'] = np.tile(patch_shape, (adata.shape[0], 1))
#     print("AnnData对象构建完成！")
#     adata.layers['rawX'] = adata.X
#     adata.uns['Original_Image'] = img_padded
#     return adata

import numpy as np
import anndata as ad
from tqdm.auto import tqdm
from typing import Tuple

import numpy as np
from typing import Tuple


def _crop_and_flatten_patches_vectorized_corrected(
        full_img: np.ndarray,
        coords: np.ndarray,
        patch_size: int
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], np.ndarray]:
    """
    一个经过修正和优化的辅助函数，使用NumPy向量化操作高效地裁剪图像块，
    能够同时处理奇数和偶数尺寸的patch_size。
    严格按照原始函数的格式返回所有四个必需的值。

    Args:
        full_img (np.ndarray): 全尺寸组织图像 (H, W, C)。
        coords (np.ndarray): 细胞/斑点的中心坐标 (N, 2)，格式为 (y, x)。
        patch_size (int): 每个图像块的正方形边长 (可以是奇数或偶数)。

    Returns:
        Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], np.ndarray]:
        - flattened_patches (np.ndarray): 展平后的图像块数据 (N, C*H*W)。
        - patches_transposed (np.ndarray): 转置后的4D图像块数据 (N, C, H, W)。
        - patch_shape (Tuple[int, int, int]): 单个图像块的形状 (C, H, W)。
        - img_padded (np.ndarray): 填充后的原始图像。
    """
    # --- 主要修改点 ---
    # 移除'patch_size'必须为偶数的限制。
    # 使用更精确的填充逻辑来同时支持奇数和偶数尺寸。
    # 对于奇数尺寸 (例如 7), 7 // 2 = 3。我们需要在中心像素前后各取3个像素 (3+1+3=7)。
    # 对于偶数尺寸 (例如 6), 6 // 2 = 3。我们习惯上在中心前取3个像素，后取2个像素 (-3,...,+2)。
    # 这种非对称性被 np.indices 的索引方式自然处理了。

    n_coords = len(coords)
    n_channels = full_img.shape[2]

    # 1. 计算精确的填充宽度
    # pad_before 定义了中心点之前需要多少像素
    pad_before = patch_size // 2
    # pad_after 定义了中心点之后需要多少像素
    # 对于奇数 (如 7), pad_after = 7 - 1 - 3 = 3
    # 对于偶数 (如 6), pad_after = 6 - 1 - 3 = 2
    pad_after = patch_size - 1 - pad_before

    print(f"Vectorizing and cropping {patch_size}x{patch_size} image patches from {n_coords} coordinate points...")
    # 2. 使用计算出的精确宽度进行零填充，更节省内存
    img_padded = np.pad(
        full_img,
        pad_width=((pad_before, pad_after), (pad_before, pad_after), (0, 0)),
        mode='constant',
        constant_values=0
    )
    # 坐标也相应地根据 'pad_before'进行偏移
    coords_padded = coords + pad_before

    # 3. 创建索引网格以进行向量化切片
    # delta 的计算方式与 'pad_before' 保持一致
    # np.indices(...,) - pad_before 会生成一个从 -pad_before 到 +pad_after 的索引范围
    delta_y, delta_x = np.indices((patch_size, patch_size)) - pad_before
    all_y_indices = coords_padded[:, 0].reshape(-1, 1, 1) + delta_y
    all_x_indices = coords_padded[:, 1].reshape(-1, 1, 1) + delta_x

    # 4. 一次性提取所有图像块，并转置为 (N, C, H, W)
    patches_transposed = img_padded[all_y_indices, all_x_indices].transpose(0, 3, 1, 2)

    # 5. 从4D数组派生出其他需要的返回值
    # 派生出展平后的2D数组
    flattened_patches = patches_transposed.reshape(n_coords, -1)

    # 派生出图像块的形状元组
    patch_shape = patches_transposed.shape[1:]

    # 6. 严格按照原始顺序和类型返回所有四个值
    # 注意：返回的 coords_padded 现在是 coords + pad_before，与填充后的图像精确对应
    return (
        flattened_patches.astype(np.float32),
        coords_padded,
        patch_shape,
        img_padded
    )

# =================================================================================
# 您原来的 create_img_adata_from_data 函数现在可以无缝对接这个优化版本
# 无需做任何修改
# =================================================================================
def create_img_adata_from_data(
        full_img: np.ndarray,
        barcodes: np.ndarray,
        img_coordinates: np.ndarray,
        spatial_coords: np.ndarray = None,
        patch_size: int = 32
) -> ad.AnnData:
    assert barcodes.shape[0] == img_coordinates.shape[0]
    assert img_coordinates.shape[1] == 2
    if not spatial_coords is None:
        assert spatial_coords.shape[0] == barcodes.shape[0]

    # *** 现在调用修正后的优化版本 ***
    # 返回的所有值都能被正确解包
    feature_matrix, coords_padded, patch_shape, img_padded = _crop_and_flatten_patches_vectorized_corrected(
        full_img=full_img,
        coords=img_coordinates,
        patch_size=patch_size
    )

    adata = ad.AnnData(X=feature_matrix)
    adata.obs_names = barcodes
    adata.var['mode'] = 'IMG'
    adata.var_names = [f'pixel_{i}' for i in range(feature_matrix.shape[1])]
    if not spatial_coords is None:
        adata.obsm['spatial'] = spatial_coords
    adata.obsm['img_coordinates'] = coords_padded
    adata.obsm['IMG_Shape'] = np.tile(patch_shape, (adata.shape[0], 1))
    print("AnnData object construction complete!")
    adata.layers['rawX'] = adata.X
    adata.uns['Original_Image'] = img_padded
    return adata

import anndata
import numpy as np
import pandas as pd
from typing import List, Literal, Tuple


import anndata
import numpy as np
import pandas as pd
from typing import List, Literal, Tuple

def match_and_filter_adata_lists(
    adata_list1: List[anndata.AnnData],
    adata_list2: List[anndata.AnnData],
    match_by: Literal['obs_names', 'spatial'] = 'obs_names',
    spatial_key: str = 'spatial',
    reference_list: Literal[1, 2] = 1
) -> Tuple[List[anndata.AnnData], List[anndata.AnnData]]:
    """
    Matches, filters, and synchronizes two lists of anndata objects pairwise.

    This function not only finds and filters common observations based on the
    specified matching criteria, but also synchronizes non-matching attributes
    to ensure the returned pairs are perfectly aligned in both obs_names and
    spatial coordinates.

    Parameters:
    ----------
    adata_list1 : List[anndata.AnnData]
        The first list of anndata objects.

    adata_list2 : List[anndata.AnnData]
        The second list of anndata objects. Must have the same length as adata_list1.

    match_by : Literal['obs_names', 'spatial'], defaults to 'obs_names'
        The matching mode.

    spatial_key : str, defaults to 'spatial'
        The key for spatial coordinates when match_by='spatial'.

    reference_list : Literal[1, 2], defaults to 1
        Specifies which list serves as the "reference" or "standard" for
        synchronizing non-matching attributes.
        - 1: list1 is the reference, list2 aligns with list1.
        - 2: list2 is the reference, list1 aligns with list2.

    Returns:
    -------
    Tuple[List[anndata.AnnData], List[anndata.AnnData]]
        A tuple containing two new lists of anndata objects that have been
        filtered and fully synchronized.
    """
    if len(adata_list1) != len(adata_list2):
        raise ValueError("Input Error: The lengths of the two lists must be equal.")
    if match_by not in ['obs_names', 'spatial']:
        raise ValueError(f"Input Error: `match_by` parameter must be 'obs_names' or 'spatial'.")
    if reference_list not in [1, 2]:
        raise ValueError("Input Error: `reference_list` parameter must be 1 or 2.")

    matched_list1 = []
    matched_list2 = []

    for i, (adata1, adata2) in enumerate(zip(adata_list1, adata_list2)):
        print(f"\n--- Processing pair {i+1} of anndata objects ---")
        print(f"Original sizes: List1 -> {adata1.shape[0]}, List2 -> {adata2.shape[0]}")

        adata1_filtered, adata2_filtered = None, None

        if match_by == 'obs_names':
            common_obs = sorted(list(set(adata1.obs_names) & set(adata2.obs_names)))
            if not common_obs:
                print("Warning: No common obs_names found.")
                adata1_filtered, adata2_filtered = adata1[[],:].copy(), adata2[[],:].copy()
            else:
                adata1_filtered = adata1[common_obs, :].copy()
                adata2_filtered = adata2[common_obs, :].copy()
                
                # New: Synchronize spatial coordinates
                print(f"Matching mode: 'obs_names'. Synchronizing spatial coordinates with reference list {reference_list}...")
                if reference_list == 1:
                    adata2_filtered.obsm[spatial_key] = adata1_filtered.obsm[spatial_key]
                else: # reference_list == 2
                    adata1_filtered.obsm[spatial_key] = adata2_filtered.obsm[spatial_key]

        elif match_by == 'spatial':
            if spatial_key not in adata1.obsm or spatial_key not in adata2.obsm:
                raise KeyError(f"Error: `spatial_key='{spatial_key}'` does not exist in the objects.")
            
            coords1_str = [f"{c[0]:.6f},{c[1]:.6f}" for c in adata1.obsm[spatial_key]]
            coords2_str = [f"{c[0]:.6f},{c[1]:.6f}" for c in adata2.obsm[spatial_key]]
            df1 = pd.DataFrame({'obs_name_1': adata1.obs_names, 'coord_str': coords1_str})
            df2 = pd.DataFrame({'obs_name_2': adata2.obs_names, 'coord_str': coords2_str})
            merged_df = pd.merge(df1, df2, on='coord_str', how='inner')

            if merged_df.empty:
                print("Warning: No common spatial coordinates found.")
                adata1_filtered, adata2_filtered = adata1[[],:].copy(), adata2[[],:].copy()
            else:
                obs_to_keep1 = merged_df['obs_name_1'].values
                obs_to_keep2 = merged_df['obs_name_2'].values
                adata1_filtered = adata1[obs_to_keep1, :].copy()
                adata2_filtered = adata2[obs_to_keep2, :].copy()

                # New: Synchronize obs_names
                print(f"Matching mode: 'spatial'. Synchronizing obs_names with reference list {reference_list}...")
                if reference_list == 1:
                    adata2_filtered.obs_names = adata1_filtered.obs_names
                else: # reference_list == 2
                    adata1_filtered.obs_names = adata2_filtered.obs_names
        
        print(f"Size after matching and synchronization: {adata1_filtered.shape[0]}")
        matched_list1.append(adata1_filtered)
        matched_list2.append(adata2_filtered)

    return matched_list1, matched_list2


import pandas as pd
import anndata as ad
import numpy as np
from natsort import natsorted


def merge_and_rename_peaks_in_adata_list(adata_list):
    """
    在一系列AnnData对象中，合并重叠的peaks，并根据合并后的共识peaks
    直接在原始AnnData对象上进行重命名和聚合（in-place modification）。

    警告：此操作会清空 .varm 和 .varp 中的数据。

    Args:
        adata_list (list): AnnData对象的列表。此列表中的对象将被直接修改。
    """

    # --- 步骤 1, 2, 3: 收集、合并、创建映射（与之前版本完全相同）---
    # (为了简洁，这里省略了这部分代码，它们与上一个版本一致)
    all_peaks_df_list = []
    if not adata_list:
        print("输入的列表为空，无法处理。")
        return
    for i, adata in enumerate(adata_list):
        if adata.n_vars == 0:
            continue
        df = pd.DataFrame({'peak_name': adata.var_names})
        df['source_adata_index'] = i
        all_peaks_df_list.append(df)
    if not all_peaks_df_list:
        print("所有AnnData对象都没有特征（peaks），无法处理。")
        return
    combined_peaks_df = pd.concat(all_peaks_df_list, ignore_index=True)

    def parse_peak_name(peak_name):
        try:
            parts = peak_name.split(':')
            chrom = parts[0]
            start, end = parts[1].split('-')
            return chrom, int(start), int(end)
        except (ValueError, IndexError):
            return None, None, None

    parsed_coords = combined_peaks_df['peak_name'].apply(parse_peak_name)
    combined_peaks_df[['chromosome', 'start', 'end']] = pd.DataFrame(parsed_coords.tolist(),
                                                                     index=combined_peaks_df.index)
    combined_peaks_df.dropna(subset=['chromosome', 'start', 'end'], inplace=True)
    combined_peaks_df['start'] = combined_peaks_df['start'].astype(int)
    combined_peaks_df['end'] = combined_peaks_df['end'].astype(int)
    unique_chroms = combined_peaks_df['chromosome'].unique()
    combined_peaks_df['chromosome'] = pd.Categorical(
        combined_peaks_df['chromosome'], categories=natsorted(unique_chroms), ordered=True
    )
    combined_peaks_df.sort_values(by=['chromosome', 'start'], inplace=True)
    merged_peaks = []
    for chromosome, group in combined_peaks_df.groupby('chromosome', observed=True):
        if group.empty:
            continue
        current_start, current_end = group.iloc[0]['start'], group.iloc[0]['end']
        for i in range(1, len(group)):
            next_peak = group.iloc[i]
            if next_peak['start'] <= current_end:
                current_end = max(current_end, next_peak['end'])
            else:
                merged_peaks.append({'chromosome': str(chromosome), 'start': current_start, 'end': current_end})
                current_start, current_end = next_peak['start'], next_peak['end']
        merged_peaks.append({'chromosome': str(chromosome), 'start': current_start, 'end': current_end})
    consensus_peaks_df = pd.DataFrame(merged_peaks)
    consensus_peaks_df['consensus_peak_name'] = (
            consensus_peaks_df['chromosome'] + ':' + consensus_peaks_df['start'].astype(str) + '-' + consensus_peaks_df[
        'end'].astype(str)
    )
    print(f"原始peaks总数: {len(combined_peaks_df)}")
    print(f"合并后的共识peaks数量: {len(consensus_peaks_df)}")
    peak_to_consensus_map = {}
    consensus_intervals = {chrom: [] for chrom in unique_chroms}
    for _, row in consensus_peaks_df.iterrows():
        consensus_intervals[row['chromosome']].append((row['start'], row['end'], row['consensus_peak_name']))
    for _, original_peak in combined_peaks_df.iterrows():
        chrom, start, end = original_peak['chromosome'], original_peak['start'], original_peak['end']
        if chrom in consensus_intervals:
            for c_start, c_end, c_name in consensus_intervals[str(chrom)]:
                if start < c_end and end > c_start:
                    peak_to_consensus_map[original_peak['peak_name']] = c_name
                    break

    # --- 步骤 4: 在每个原始AnnData对象上进行重命名和聚合 (In-place) ---
    print("\n正在直接修改原始AnnData对象...")
    for adata in adata_list:
        if adata.n_vars == 0:
            continue

        # 1. 准备聚合信息
        new_var_names = [peak_to_consensus_map.get(name, name) for name in adata.var_names]

        # 2. 计算聚合后的 .X 和 .layers 矩阵
        # 创建一个临时列用于 groupby
        adata.var['consensus_peak_name'] = new_var_names

        # 聚合主数据矩阵 .X
        data_df = pd.DataFrame(
            adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
            index=adata.obs_names,
            columns=adata.var['consensus_peak_name']
        )
        aggregated_X_df = data_df.groupby(axis=1, level=0).sum()

        # 聚合 .layers 中的数据
        aggregated_layers = {}
        for layer_name, layer_matrix in adata.layers.items():
            layer_df = pd.DataFrame(
                layer_matrix.toarray() if hasattr(layer_matrix, "toarray") else layer_matrix,
                index=adata.obs_names,
                columns=adata.var['consensus_peak_name']
            )
            aggregated_layer_df = layer_df.groupby(axis=1, level=0).sum()
            aggregated_layers[layer_name] = aggregated_layer_df

        # 3. 创建新的 var DataFrame
        new_var_df = pd.DataFrame(index=aggregated_X_df.columns)

        # 4. **核心：执行原地修改**

        # 首先，清空不兼容的属性
        adata.varm.clear()
        adata.varp.clear()

        # 获取新的变量名列表，确保顺序正确
        final_var_names = new_var_df.index.tolist()

        # 使用 anndata 的内部方法 `._inplace_subset_var` 是最安全的方式
        # 它会正确处理所有相关属性的切片。
        # 我们通过创建一个 "假" 的索引来实现替换。
        # 注意: AnnData 0.8.0 之后，直接赋值更受推荐。为了兼容性和直接性，我们用直接赋值。

        # 直接赋值 (适用于 AnnData >= 0.8.0)
        adata._var = new_var_df
        adata._X = aggregated_X_df[final_var_names].values

        # 更新 layers
        adata.layers.clear()
        for layer_name, agg_layer_df in aggregated_layers.items():
            adata.layers[layer_name] = agg_layer_df[final_var_names].values

        # 清理临时列
        # 因为我们替换了整个 .var，所以这个临时列已经不存在了

    print("\n成功在所有原始AnnData对象上完成了重命名和聚合。")
    print("警告: 每个对象的 .varm 和 .varp 属性已被清空。")
    return adata_list


# import anndata
# import numpy as np
# import pandas as pd
# from typing import List, Literal, Tuple


# def match_and_filter_adata_lists(
#         adata_list1: List[anndata.AnnData],
#         adata_list2: List[anndata.AnnData],
#         match_by: Literal['obs_names', 'spatial'] = 'obs_names',
#         spatial_key: str = 'spatial',
#         reference_list: Literal[1, 2] = 1
# ) -> Tuple[List[anndata.AnnData], List[anndata.AnnData]]:
#     """
#     成对匹配、过滤并同步两个anndata对象列表。

#     此函数不仅会根据指定的匹配标准找到共同的观测值并进行过滤，
#     还会将非匹配属性进行同步，确保返回的匹配对在obs_names和空间坐标上
#     都完全一致。

#     参数:
#     ----------
#     adata_list1 : List[anndata.AnnData]
#         第一个anndata对象列表。

#     adata_list2 : List[anndata.AnnData]
#         第二个anndata对象列表。长度必须与adata_list1相等。

#     match_by : Literal['obs_names', 'spatial'], 默认为'obs_names'
#         匹配模式。

#     spatial_key : str, 默认为'spatial'
#         当match_by='spatial'时，空间坐标的键名。

#     reference_list : Literal[1, 2], 默认为 1
#         指定哪个列表作为同步非匹配属性时的“参考”或“标准”。
#         - 1: list1是参考，list2向list1看齐。
#         - 2: list2是参考，list1向list2看齐。

#     返回:
#     -------
#     Tuple[List[anndata.AnnData], List[anndata.AnnData]]
#         一个元组，包含两个新的列表，其中的anndata对象已被过滤和完全同步。
#     """
#     if len(adata_list1) != len(adata_list2):
#         raise ValueError("输入错误: 两个列表的长度必须相等。")
#     if match_by not in ['obs_names', 'spatial']:
#         raise ValueError(f"输入错误: `match_by` 参数必须是 'obs_names' 或 'spatial'。")
#     if reference_list not in [1, 2]:
#         raise ValueError("输入错误: `reference_list` 参数必须是 1 或 2。")

#     matched_list1 = []
#     matched_list2 = []

#     for i, (adata1, adata2) in enumerate(zip(adata_list1, adata_list2)):
#         print(f"\n--- 正在处理第 {i + 1} 对 anndata 对象 ---")
#         print(f"原始尺寸: List1 -> {adata1.shape[0]}, List2 -> {adata2.shape[0]}")

#         adata1_filtered, adata2_filtered = None, None

#         if match_by == 'obs_names':
#             common_obs = sorted(list(set(adata1.obs_names) & set(adata2.obs_names)))
#             if not common_obs:
#                 print("警告: 未找到共同的 obs_names。")
#                 adata1_filtered, adata2_filtered = adata1[[], :].copy(), adata2[[], :].copy()
#             else:
#                 adata1_filtered = adata1[common_obs, :].copy()
#                 adata2_filtered = adata2[common_obs, :].copy()

#                 # 新增：同步空间坐标
#                 print(f"匹配模式: 'obs_names'。正在将 spatial 坐标与参考列表 {reference_list} 同步...")
#                 if reference_list == 1:
#                     adata2_filtered.obsm[spatial_key] = adata1_filtered.obsm[spatial_key]
#                 else:  # reference_list == 2
#                     adata1_filtered.obsm[spatial_key] = adata2_filtered.obsm[spatial_key]

#         elif match_by == 'spatial':
#             if spatial_key not in adata1.obsm or spatial_key not in adata2.obsm:
#                 raise KeyError(f"错误: `spatial_key='{spatial_key}'` 在对象中不存在。")

#             coords1_str = [f"{c[0]:.6f},{c[1]:.6f}" for c in adata1.obsm[spatial_key]]
#             coords2_str = [f"{c[0]:.6f},{c[1]:.6f}" for c in adata2.obsm[spatial_key]]
#             df1 = pd.DataFrame({'obs_name_1': adata1.obs_names, 'coord_str': coords1_str})
#             df2 = pd.DataFrame({'obs_name_2': adata2.obs_names, 'coord_str': coords2_str})
#             merged_df = pd.merge(df1, df2, on='coord_str', how='inner')

#             if merged_df.empty:
#                 print("警告: 未找到共同的空间坐标。")
#                 adata1_filtered, adata2_filtered = adata1[[], :].copy(), adata2[[], :].copy()
#             else:
#                 obs_to_keep1 = merged_df['obs_name_1'].values
#                 obs_to_keep2 = merged_df['obs_name_2'].values
#                 adata1_filtered = adata1[obs_to_keep1, :].copy()
#                 adata2_filtered = adata2[obs_to_keep2, :].copy()

#                 # 新增：同步 obs_names
#                 print(f"匹配模式: 'spatial'。正在将 obs_names 与参考列表 {reference_list} 同步...")
#                 if reference_list == 1:
#                     adata2_filtered.obs_names = adata1_filtered.obs_names
#                 else:  # reference_list == 2
#                     adata1_filtered.obs_names = adata2_filtered.obs_names

#         print(f"匹配并同步后尺寸: {adata1_filtered.shape[0]}")
#         matched_list1.append(adata1_filtered)
#         matched_list2.append(adata2_filtered)

#     return matched_list1, matched_list2


