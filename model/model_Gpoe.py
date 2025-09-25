import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scanpy as sc
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from scipy.sparse import issparse
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from typing import Literal

# --- 依赖库 (timm是新增的) ---
from timm.models.vision_transformer import VisionTransformer
import timm # *** NEW ***
from timm.layers import DropPath

from safetensors.torch import load_file # *** NEW: 导入safetensors的加载函数 ***
import warnings
from scipy.stats import ConstantInputWarning

# 将 ConstantInputWarning 警告设置为忽略
warnings.filterwarnings("ignore", category=ConstantInputWarning)

local_weights_path = './model.safetensors' 

import torch
from torch import nn
import numpy as np

class FourierFeatureMapping(nn.Module):
    """
    将低维输入坐标映射到高维傅里叶特征空间。
    """
    def __init__(self, input_dim: int, embedding_dim: int, scale: float = 1.0):
        """
        Args:
            input_dim (int): 输入维度，例如 2 (x, y)。
            embedding_dim (int): 输出维度，必须是偶数。
            scale (float): 傅里叶特征的频率尺度。
        """
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be an even number.")
        
        # 创建可学习的傅里叶频率矩阵 B
        self.B = nn.Parameter(
            torch.randn(input_dim, embedding_dim // 2) * scale,
            requires_grad=True # 频率矩阵可学习
        )
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 形状为 (batch_size, input_dim) 的输入坐标。
        
        Returns:
            torch.Tensor: 形状为 (batch_size, embedding_dim) 的傅里叶特征。
        """
        # (x @ self.B) 的形状为 (batch_size, embedding_dim // 2)
        # 乘以 2π，然后计算 sin 和 cos
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# --- VAE 基础模块 (保持不变) ---
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
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

# ======================================================================
# --- 1. 空间门控网络 (全新，实现您的想法) ---
# ======================================================================
class SpatialGatingNetwork(nn.Module):
    """
    空间门控网络，用于评估输入坐标相对于训练集坐标分布的新颖性。
    """
    def __init__(self, latent_dim: int, k: int = 5):
        """
        Args:
            latent_dim (int): 空间编码器输出的特征维度。
            k (int): 计算最近邻时考虑的邻居数量。
        """
        super().__init__()
        self.k = k
        
        # 这个 buffer 用于存储训练集的坐标，它不是模型参数，但会随模型一起保存和移动
        self.register_buffer('training_coords', torch.randn(1, 2)) 
        
        # 一个小型MLP，将距离特征转换为门控权重
        self.gate_mlp = nn.Sequential(
            nn.Linear(latent_dim + 1, 64), # 输入 = 空间特征 + 距离
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def load_training_coords(self, coords: torch.Tensor):
        """在训练开始前，加载所有训练坐标。"""
        print(f"加载 {coords.shape[0]} 个训练坐标到空间门控网络。")
        self.training_coords = coords

    def forward(self, spatial_features: torch.Tensor, current_coords: torch.Tensor):
        """
        Args:
            spatial_features (torch.Tensor): 空间编码器的输出特征 (batch, latent_dim)。
            current_coords (torch.Tensor): 当前批次的原始坐标 (batch, 2)。

        Returns:
            torch.Tensor: 空间模态的门控权重 beta (batch, 1)。
        """
        if self.training and self.training_coords.shape[0] < 2:
             # 在训练的初始阶段或未加载坐标时，默认信任度为1
            return torch.ones(current_coords.shape[0], 1, device=current_coords.device)
        
        # 计算当前坐标与训练集坐标库之间的距离
        # dists: (batch_size, num_training_coords)
        dists = torch.cdist(current_coords, self.training_coords)
        
        # 找到k个最近邻的距离并取平均
        # topk_dists: (batch_size, k)
        topk_dists, _ = torch.topk(dists, self.k, dim=1, largest=False)
        avg_dist = topk_dists.mean(dim=1, keepdim=True) # (batch_size, 1)
        
        # 将距离特征与空间特征拼接
        combined_features = torch.cat([spatial_features, avg_dist], dim=1)
        
        # 通过MLP得到最终的门控权重 beta
        beta = self.gate_mlp(combined_features)
        
        return beta

# ======================================================================
# --- 2. 双向门控融合模块 (终极版) ---
# ======================================================================
class DualGatedPoEFusion(nn.Module):
    """
    双向门控专家乘积 (BG-PoE) 模块。
    包含一个图像门控和一个空间门控。
    """
    def __init__(self, img_feature_dim: int, spatial_latent_dim: int):
        super().__init__()
        
        # 图像门控 (与上一版类似)
        self.image_gating_network = nn.Sequential(
            nn.Linear(img_feature_dim, img_feature_dim // 4), nn.ReLU(),
            nn.Linear(img_feature_dim // 4, 1), nn.Sigmoid()
        )
        
        # 空间门控 (您提出的新想法)
        self.spatial_gating_network = SpatialGatingNetwork(latent_dim=spatial_latent_dim)

    def forward(self, img_features, mu_img, log_var_img, 
                spatial_features, mu_spatial, log_var_spatial, current_coords):
        
        # 1. 计算图像门控权重 alpha
        alpha = self.image_gating_network(img_features)
        
        # 2. 计算空间门控权重 beta
        beta = self.spatial_gating_network(spatial_features, current_coords)
        
        # 3. 双向加权融合
        var_img = torch.exp(log_var_img); var_spatial = torch.exp(log_var_spatial)
        precision_img = (alpha+0.5) / var_img; precision_spatial = (beta+0.5) / var_spatial
        
        # 使用 alpha 和 beta 同时加权
        fused_precision = precision_img + precision_spatial
        
        # 为避免分母为0，加入一个小的epsilon
        fused_var = 1.0 / (fused_precision + 1e-8)
        fused_log_var = torch.log(fused_var)
        
        fused_mu = (mu_img * precision_img) + (mu_spatial * precision_spatial) * fused_var
        
        return fused_mu, fused_log_var, alpha, beta
    
# ======================================================================
# --- 1. 核心创新模块: GatedPoEFusion (全新) ---
# ======================================================================
class GatedPoEFusion(nn.Module):
    """
    门控专家乘积 (Gated Product of Experts) 模块。
    该模块包含一个门控网络，用于动态学习图像模态的可靠性权重 alpha，
    并使用该权重自适应地融合图像和空间两个模态的潜在分布。
    """
    def __init__(self, feature_dim: int, latent_dim: int):
        """
        Args:
            feature_dim (int): 输入特征的维度，通常是Swin Transformer的输出维度。
            latent_dim (int): 潜在空间的维度。
        """
        super().__init__()
        
        # 定义门控网络 Gating Network
        # 输入是图像特征，输出是一个标量权重 alpha
        self.gating_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()  # 使用 Sigmoid 确保 alpha 在 (0, 1) 范围内
        )
        
    def forward(self, img_features, mu_img, log_var_img, mu_spatial, log_var_spatial):
        """
        执行门控融合。

        Args:
            img_features (torch.Tensor): 从图像编码器提取的特征，用于门控网络。
            mu_img, log_var_img (torch.Tensor): 图像模态的潜在分布参数。
            mu_spatial, log_var_spatial (torch.Tensor): 空间模态的潜在分布参数。

        Returns:
            tuple: 包含融合后的均值、对数方差以及计算出的门控权重 alpha。
        """
        # 1. 通过门控网络计算动态权重 alpha
        # alpha 的形状是 (batch_size, 1)
        alpha = self.gating_network(img_features)
        
        # 2. 计算方差和精度 (inverse variance)
        var_img = torch.exp(log_var_img)
        var_spatial = torch.exp(log_var_spatial)
        
        precision_img = 1.0 / var_img
        precision_spatial = 1.0 / var_spatial
        
        # 3. 应用 Gated PoE 公式进行融合
        # 使用 alpha 对精度进行加权
        fused_precision = alpha * precision_img + (1 - alpha) * precision_spatial
        fused_var = 1.0 / fused_precision
        fused_log_var = torch.log(fused_var)
        
        # 同样对均值进行加权
        fused_mu = (alpha * (mu_img * precision_img) + (1 - alpha) * (mu_spatial * precision_spatial)) * fused_var
        
        return fused_mu, fused_log_var, alpha
    

class ConvBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super(ConvBlock, self).__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# --- 负二项分布组件 (保持不变) ---
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
def log_nb_positive(x, mu, theta, eps=1e-6):
    log_theta_mu_eps = torch.log(theta + mu + eps)
    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    return res

class NegativeBinomial(Distribution):
    arg_constraints = {"mu": constraints.greater_than_eq(0), "theta": constraints.greater_than_eq(0)}
    support = constraints.nonnegative_integer
    def __init__(self, mu, theta, validate_args=False):
        self._eps = 1e-8
        self.mu, self.theta = broadcast_all(mu, theta)
        super().__init__(validate_args=validate_args)
    @property
    def mean(self): return self.mu
    def log_prob(self, value): return log_nb_positive(value, self.mu, self.theta, eps=self._eps)

def scellst_nll_loss(dist_params, target_rna):
    mu = dist_params['mu']; theta = dist_params['theta']
    nb_dist = NegativeBinomial(mu=mu, theta=theta)
    return -nb_dist.log_prob(target_rna).sum() / target_rna.shape[0]

# ======================================================================
# --- 1. 核心模型: MultiModalVAE (已升级) ---
# ======================================================================
# ======================================================================
# --- 2. 核心模型: MultiModalVAE (已升级为Gated PoE版) ---
# ======================================================================
class MultiModalVAE(nn.Module):
    def __init__(self, input_channels=3, spatial_dim=2, latent_dim=128, img_size=32, rna_dim=2000, fourier_feature_dim=128, hidden_dims=None):
        super(MultiModalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.rna_dim = rna_dim

        # --- 编码器 (Encoders) ---
        
        # 1.1 图像编码器 (Image Encoder) - 使用 Swin Transformer (保持不变)
        self.image_encoder_swin = timm.create_model(
            'swin_tiny_patch4_window7_224', pretrained=False, num_classes=0, 
            img_size=self.img_size, in_chans=input_channels
        )
        # ... (本地权重加载代码保持不变) ...
        try:
            print(f"正在从本地文件加载预训练权重: {local_weights_path}")
            pretrained_dict = load_file(local_weights_path, device='cpu')
            model_dict = self.image_encoder_swin.state_dict()
            pretrained_dict_filtered = {
                k: v for k, v in pretrained_dict.items() 
                if k in model_dict and model_dict[k].shape == v.shape
            }
            model_dict.update(pretrained_dict_filtered)
            self.image_encoder_swin.load_state_dict(model_dict)
            loaded_keys = len(pretrained_dict_filtered)
            total_keys = len(model_dict)
            print(f"本地权重加载成功！匹配并加载了 {loaded_keys} / {total_keys} 个权重层。")
        except FileNotFoundError:
            print(f"错误：找不到权重文件 {local_weights_path}！模型将使用随机初始化的权重。")

        swin_output_dim = self.image_encoder_swin.num_features
        self.img_fc_mu = nn.Linear(swin_output_dim, latent_dim)
        self.img_fc_log_var = nn.Linear(swin_output_dim, latent_dim)

        # 1.2 空间位置编码器 (Spatial Encoder) (保持不变)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU()
        )
        self.spatial_fc_mu = nn.Linear(128, latent_dim)
        self.spatial_fc_log_var = nn.Linear(128, latent_dim)
        
        # *** NEW: 1.3 实例化门控融合模块 ***
        self.fusion_module = DualGatedPoEFusion(
            img_feature_dim=swin_output_dim, 
            spatial_latent_dim=latent_dim
        )

        # --- 解码器 (Decoders) (保持不变) ---
        self.decoder_start_channels = 256
        self.decoder_start_size = img_size // 8 
        decoder_input_size = self.decoder_start_channels * (self.decoder_start_size ** 2)
        self.img_decoder_input = nn.Linear(latent_dim, decoder_input_size)
        # ... (图像和RNA解码器的其余部分代码保持不变，此处省略以保持简洁) ...
        if hidden_dims is None: hidden_dims = [32, 64, 128, 256]
        img_decoder_dims = hidden_dims[::-1]
        self.img_decoder_stages = nn.ModuleList()
        for i in range(len(img_decoder_dims) - 1):
            stage = nn.Sequential(
                nn.ConvTranspose2d(img_decoder_dims[i], img_decoder_dims[i+1], kernel_size=2, stride=2),
                nn.BatchNorm2d(img_decoder_dims[i+1]), nn.LeakyReLU(inplace=True),
                ConvBlock(dim=img_decoder_dims[i+1], mlp_ratio=4.)
            )
            self.img_decoder_stages.append(stage)
        self.img_final_layer = nn.Sequential(
            nn.Conv2d(img_decoder_dims[-1], input_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        self.rna_decoder_base = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2)
        )
        self.rna_decoder_mu = nn.Sequential(nn.Linear(512, rna_dim), nn.Softplus())
        self.rna_decoder_theta = nn.Sequential(nn.Linear(512, rna_dim), nn.Softplus())

    # *** DEPRECATED: 不再需要旧的静态融合函数 ***
    # def fuse_latents(self, mu1, log_var1, mu2, log_var2): ...
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, image, spatial):
        # 1. 编码
        # 图像编码 (Swin)
        img_x = self.image_encoder_swin(image).detach() # img_x 是门控网络的输入
        mu_img = self.img_fc_mu(img_x)
        log_var_img = self.img_fc_log_var(img_x)
        
        # 空间编码 (MLP)
        spatial_x = self.spatial_encoder(spatial)
        mu_spatial = self.spatial_fc_mu(spatial_x)
        log_var_spatial = self.spatial_fc_log_var(spatial_x)
        
        # *** MODIFIED: 2. 使用门控模块进行动态融合 ***
        # *** MODIFIED: 2. 使用双向门控进行融合 ***
        mu_fused, log_var_fused, alpha, beta = self.fusion_module(
            img_features=img_x,
            mu_img=mu_img, log_var_img=log_var_img,
            spatial_features=spatial_x,
            mu_spatial=mu_spatial, log_var_spatial=log_var_spatial,
            current_coords=spatial # 传入原始坐标给空间门控
        )
        
        # 3. 重参数化 (保持不变)
        z = self.reparameterize(mu_fused, log_var_fused)
        
        # 4. 解码 (保持不变)
        # ... (解码器代码保持不变，此处省略) ...
        img_recon_x = self.img_decoder_input(z)
        img_recon_x = img_recon_x.view(-1, self.decoder_start_channels, self.decoder_start_size, self.decoder_start_size)
        for stage in self.img_decoder_stages:
            img_recon_x = stage(img_recon_x)
        reconstructed_image = self.img_final_layer(img_recon_x)
        if z.shape[0] > 1:
            rna_base = self.rna_decoder_base(z)
        else:
            original_mode = self.rna_decoder_base.training
            self.rna_decoder_base.eval()
            with torch.no_grad():
                rna_base = self.rna_decoder_base(z)
            self.rna_decoder_base.train(original_mode)
        mu_rna = self.rna_decoder_mu(rna_base)
        theta_rna = self.rna_decoder_theta(rna_base)
        reconstructed_rna_params = {"mu": mu_rna, "theta": theta_rna}

        # *** MODIFIED: 返回 alpha 用于监控 ***
        return (reconstructed_image, reconstructed_rna_params, 
                mu_fused, log_var_fused, 
                mu_img, mu_spatial,
                alpha, beta) # 新增返回alpha

    # ... (get_latent_representations 和 Dynomic_kld_weight 函数保持不变) ...
    def get_latent_representations(self, image, spatial):
        with torch.no_grad():
            img_x = self.image_encoder_swin(image).detach()
            mu_img = self.img_fc_mu(img_x)
            log_var_img = self.img_fc_log_var(img_x)
            spatial_x = self.spatial_encoder(spatial)
            mu_spatial = self.spatial_fc_mu(spatial_x)
            log_var_spatial = self.spatial_fc_log_var(spatial_x)
            
            # 使用门控融合，但这里只为了得到mu_fused和log_var_fused
            mu_fused, log_var_fused, _ = self.fusion_module(img_x, mu_img, log_var_img, mu_spatial, log_var_spatial)
            
            z_fused = self.reparameterize(mu_fused, log_var_fused)
            z_img = self.reparameterize(mu_img, log_var_img)
            z_spatial = self.reparameterize(mu_spatial, log_var_spatial)
            return z_fused, z_img, z_spatial
            
    def Dynomic_kld_weight(self, epoch, beta, kl_anneal_epochs=50):
        return beta * min(1.0, epoch / kl_anneal_epochs) if kl_anneal_epochs > 0 else beta
    
# ======================================================================
# --- 2. 混合损失函数 (已升级) ---
# ======================================================================
def multi_modal_vae_loss(recon_img, true_img, recon_rna_params, true_rna, 
                         mu_fused, log_var_fused, mu_img, mu_spatial,
                         kld_weight=1.0, image_weight=1.0, rna_weight=1.0, 
                         alignment_weight=0.1): # *** NEW: 新增对齐权重 ***
    # 图像重建损失 (MSE)
    image_recon_loss = F.mse_loss(recon_img, true_img, reduction='sum') / true_img.shape[0]

    # RNA 重建损失 (NLL)
    rna_recon_loss = scellst_nll_loss(recon_rna_params, true_rna)
    
    # KL 散度损失 (针对融合后的分布)
    kld_loss = -0.5 * torch.sum(1 + log_var_fused - mu_fused.pow(2) - log_var_fused.exp()) / true_img.shape[0]
    
    # *** NEW: 潜在空间对齐损失 (Alignment Loss) ***
    alignment_loss = F.mse_loss(mu_img, mu_spatial, reduction='mean') # / true_img.shape[0]
    
    # 加权总损失
    total_loss = (image_weight * image_recon_loss + 
                  rna_weight * rna_recon_loss + 
                  kld_weight * kld_loss +
                  alignment_weight * alignment_loss) # *** NEW ***
                  
    return total_loss, image_recon_loss, rna_recon_loss, kld_loss, alignment_loss # *** MODIFIED ***

# ======================================================================
# --- 3. 数据加载和评估指标 (保持不变) ---
# ======================================================================
import numpy as np
import torch
from scipy.sparse import issparse
from torch.utils.data import TensorDataset, DataLoader, Subset


def prepare_adata_loader(adata_img, adata_rna, layer_key_img, shape_key_img, layer_key_rna, spatial_key, batch_size, ratio=(0.7, 0.1, 0.2), Train_shuffle=True):
    """
    准备多模态数据加载器。
    此版本使用 numpy.random.shuffle 来确保与另一个模型的数据划分方式完全一致。
    """
    print("--- 开始准备多模态数据 ---")
    
    # --- 数据预处理部分 (无需修改) ---
    if not np.isclose(np.sum(ratio), 1.0):
        raise ValueError(f"The sum of ratios must be 1, but got {np.sum(ratio)}")
    
    # ... (这部分数据准备代码和之前一样，这里省略以保持简洁) ...
    image_data_flat = adata_img.layers[layer_key_img]
    if issparse(image_data_flat): image_data_flat = image_data_flat.toarray()
    if isinstance(adata_img.obsm[shape_key_img], np.ndarray) and adata_img.obsm[shape_key_img].ndim == 2:
        img_shape_original = adata_img.obsm[shape_key_img][0]
    else: img_shape_original = adata_img.obsm[shape_key_img]
    if img_shape_original[2] == 3: h, w, c = img_shape_original
    else: c, h, w = img_shape_original
    image_data_reshaped = np.transpose(image_data_flat.reshape(-1, h, w, c), (0, 3, 1, 2))
    image_data_normalized = (image_data_reshaped.astype(np.float32) / 127.5) - 1.0
    image_tensor = torch.from_numpy(image_data_normalized)
    final_img_shape = (c, h, w)
    spatial_data = adata_img.obsm[spatial_key].astype(np.float32)
    spatial_tensor = torch.from_numpy(spatial_data)
    spatial_dim = spatial_data.shape[1]
    rna_data = adata_rna.layers[layer_key_rna]
    if issparse(rna_data): rna_data = rna_data.toarray()
    rna_tensor = torch.from_numpy(rna_data.astype(np.float32))
    rna_dim = rna_data.shape[1]
    print(f"图像数据: {image_tensor.shape}, 空间数据: {spatial_tensor.shape}, RNA数据: {rna_tensor.shape}")
    dataset = TensorDataset(image_tensor, spatial_tensor, rna_tensor)
    n_samples = len(dataset)
    
    # --- 数据划分 (修改点在这里) ---
    # 1. 计算各个集合的大小 (与目标模型逻辑保持一致)
    train_ratio, val_ratio, test_ratio = ratio
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    test_size = n_samples - train_size - val_size
    
    # 2. 使用 NumPy 的方法进行随机索引划分 (与目标模型完全一致)
    # --- 修改点开始 ---
    indices = list(range(n_samples))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size:]
    # --- 修改点结束 ---
    
    # 3. 使用切分好的索引创建 PyTorch Subset (无需修改)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # --- 创建 DataLoader 部分 (无需修改) ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=Train_shuffle) if len(train_dataset) > 0 else None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if len(val_dataset) > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if len(test_dataset) > 0 else None
    
    print(f"数据准备完成。训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, final_img_shape, rna_dim, spatial_dim,train_indices,val_dataset,test_indices



from sklearn.metrics import mean_squared_error
def calculate_mse_col_normalized(y_pred, y_true):
    if isinstance(y_pred, torch.Tensor): y_pred = y_pred.cpu().numpy()
    if isinstance(y_true, torch.Tensor): y_true = y_true.cpu().numpy()
    y_pred_norm = np.log1p(y_pred); y_true_norm = np.log1p(y_true)
    return mean_squared_error(y_true_norm, y_pred_norm)

def calculate_median_pearson(y_pred, y_true):
    if isinstance(y_pred, torch.Tensor): y_pred = y_pred.cpu().numpy()
    if isinstance(y_true, torch.Tensor): y_true = y_true.cpu().numpy()
    correlations = [pearsonr(y_pred[:, i], y_true[:, i])[0] for i in range(y_pred.shape[1])]
    return np.median(np.nan_to_num(correlations, nan=0.0))

def calculate_spearman_correlation(pred_matrix, raw_matrix, axis=1):
    correlations = []
    if pred_matrix.shape[0] == 0 or raw_matrix.shape[0] == 0: return 0.0
    for i in range(pred_matrix.shape[1]):
        corr, _ = spearmanr(pred_matrix[:, i], raw_matrix[:, i])
        correlations.append(corr)
    return np.nanmedian(np.nan_to_num(np.array(correlations), nan=0.0))

# ======================================================================
# --- 4. 训练流程 (已升级) ---
# ======================================================================
import torch.optim as optim
import numpy as np

# (Assuming other necessary functions like calculate_spearman_correlation, etc., are defined elsewhere)

def train_multi_modal_vae(model, train_loader, val_loader, epochs, learning_rate, device, **loss_weights):
    """
    训练双向门控多模态VAE模型的函数。
    此版本已更新，以处理和报告 alpha (图像门控) 和 beta (空间门控) 权重。
    """
    print("\n--- 开始 Multi-Modal VAE 模型训练 (双向门控 BG-PoE 版) ---")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    best_val_corr = 0.0  # Use a more descriptive name
    best_model_state = None # Store the model state_dict for best performance
    
    beta_kld = loss_weights.get('kld_weight', 1.0) 

    # --- 关键步骤：在训练开始前加载训练坐标到空间门控网络 ---
    print("正在提取并加载训练坐标到空间门控网络...")
    all_train_coords = []
    # It's better to iterate through the train_dataset if available, 
    # but iterating through loader also works.
    for (_, spatial_data, _) in train_loader:
        all_train_coords.append(spatial_data)
    
    if not all_train_coords:
        raise ValueError("训练数据加载器为空，无法提取坐标！")
        
    all_train_coords_tensor = torch.cat(all_train_coords, dim=0).to(device)
    
    # 调用模型内的方法加载坐标
    model.fusion_module.spatial_gating_network.load_training_coords(all_train_coords_tensor)

    for epoch in range(1, epochs + 1):
        model.train()
        
        # *** MODIFIED: 初始化 alpha 和 beta 的累加器 ***
        total_loss, total_img_loss, total_rna_loss, total_kld_loss, total_align_loss = 0, 0, 0, 0, 0
        total_alpha, total_beta = 0, 0 # 新增
        
        current_loss_weights = loss_weights.copy()
        current_loss_weights['kld_weight'] = model.Dynomic_kld_weight(epoch, beta_kld, kl_anneal_epochs=50)

        for (img_data, spatial_data, rna_data) in train_loader:
            img_data, spatial_data, rna_data = img_data.to(device), spatial_data.to(device), rna_data.to(device)
            
            # *** MODIFIED: 接收模型返回的所有值，包括 alpha 和 beta ***
            outputs = model(img_data, spatial_data)
            recon_img, recon_rna_params, mu_fused, log_var_fused, mu_img, mu_spatial, alpha, beta = outputs
            
            # 损失函数调用保持不变，它不需要 alpha 和 beta
            loss, img_loss, rna_loss, kld_loss, align_loss = multi_modal_vae_loss(
                recon_img, img_data, recon_rna_params, rna_data, 
                mu_fused, log_var_fused, mu_img, mu_spatial,
                **current_loss_weights
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_img_loss += img_loss.item()
            total_rna_loss += rna_loss.item()
            total_kld_loss += kld_loss.item()
            total_align_loss += align_loss.item()
            
            # *** NEW: 累加当前批次的平均 alpha 和 beta 值 ***
            total_alpha += alpha.mean().item()
            total_beta += beta.mean().item()

        # --- 验证过程 ---
        model.eval()
        predicted_rna_list, true_rna_list = [], []
        with torch.no_grad():
            for (img_data, spatial_data, rna_data) in val_loader:
                img_data, spatial_data, rna_data = img_data.to(device), spatial_data.to(device), rna_data.to(device)
                
                # *** MODIFIED: 正确解包模型输出，即使我们只用其中一部分 ***
                # 必须解包所有8个返回值以避免错误
                _, recon_rna_params, _, _, _, _, _, _ = model(img_data, spatial_data)
                
                predicted_mu = recon_rna_params['mu']
                predicted_rna_list.append(predicted_mu.cpu().numpy())
                true_rna_list.append(rna_data.cpu().numpy())
        
        predicted_rna_matrix = np.vstack(predicted_rna_list)
        true_rna_matrix = np.vstack(true_rna_list)
        spearman_corr = calculate_spearman_correlation(predicted_rna_matrix, true_rna_matrix)
        pearson_corr = calculate_median_pearson(predicted_rna_matrix, true_rna_matrix)
        mse = calculate_mse_col_normalized(predicted_rna_matrix, true_rna_matrix)
        
        # --- MODIFIED: 更新打印信息以包含 alpha 和 beta ---
        avg_loss = total_loss / len(train_loader)
        avg_img_loss = total_img_loss / len(train_loader)
        avg_rna_loss = total_rna_loss / len(train_loader)
        avg_kld_loss = total_kld_loss / len(train_loader)
        avg_align_loss = total_align_loss / len(train_loader)
        avg_alpha = total_alpha / len(train_loader)
        avg_beta = total_beta / len(train_loader)
        kld_w = current_loss_weights['kld_weight']

        print(f"Epoch: {epoch}/{epochs} | Loss: {avg_loss:.2f} "
              f"kld_w: {kld_w:.2f} "
              f"[Img: {avg_img_loss:.2f}, RNA: {avg_rna_loss:.2f}, KLD: {avg_kld_loss:.2f}, "
              f"Align: {avg_align_loss:.4f}, Alpha: {avg_alpha:.3f}, Beta: {avg_beta:.3f}] | " # 新增 Alpha, Beta
              f"Val PCC: {pearson_corr:.4f} | Val SCC: {spearman_corr:.4f} | Val MSE: {mse:.4f}")

        # 保存最佳模型
        if pearson_corr > best_val_corr:
            best_val_corr = pearson_corr
            best_model_state = model.state_dict()
            print(f"✨ New best model found at epoch {epoch} with Val PCC: {best_val_corr:.4f}")

    print("模型训练完成!")
    # 加载性能最佳的模型权重并返回
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model