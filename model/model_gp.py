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

import gpytorch

class SVGP_SpatialEncoder(gpytorch.models.ApproximateGP):
    """
    使用SVGP进行空间编码。
    我们将为每个潜在维度学习一个独立的GP，通过batch_shape实现。
    """
    def __init__(self, inducing_points, latent_dim=128):
        # inducing_points: 引导点，是GP的稀疏近似的关键。
        # 形状为 [num_inducing, 2]，例如 [100, 2]
        
        # 定义变分分布和策略，这是实现可扩展性的核心
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0),
            batch_shape=torch.Size([latent_dim]) # 为每个潜在维度创建一个独立的GP
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        # 定义GP的均值和协方差函数（核函数）
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([latent_dim]))
        # RBF核是处理平滑空间函数的经典选择
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([latent_dim])),
            batch_shape=torch.Size([latent_dim])
        )
        


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # 返回一个多变量正态分布
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

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
# --- 1. 核心模型: MultiModalVAE (最终修正版) ---
# ======================================================================
class MultiModalVAE(nn.Module):
    def __init__(self, input_channels=3, spatial_dim=2, latent_dim=128, img_size=224, rna_dim=2000, hidden_dims=None, inducing_points=None):
        super(MultiModalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.rna_dim = rna_dim

        # --- 编码器 (Encoders) ---
        
        # 1.1 图像编码器 (Image Encoder) - 使用 Swin Transformer
        self.image_encoder_swin = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            num_classes=0,
            img_size=self.img_size, # *** BUG FIX: 使用传入的 img_size ***
            in_chans=input_channels
        )
        
        # 加载本地预训练权重
        try:
            print(f"正在从本地文件加载预训练权重: {local_weights_path}")
            pretrained_dict = load_file(local_weights_path, device='cpu')
            model_dict = self.image_encoder_swin.state_dict()
            
            # 智能过滤和加载权重 (处理尺寸不匹配问题)
            pretrained_dict_filtered = {
                k: v for k, v in pretrained_dict.items() 
                if k in model_dict and model_dict[k].shape == v.shape and "pos_embed" not in k # 忽略位置编码
            }
            model_dict.update(pretrained_dict_filtered)
            self.image_encoder_swin.load_state_dict(model_dict)
            print(f"本地权重加载成功！匹配并加载了 {len(pretrained_dict_filtered)} / {len(model_dict)} 个权重层。")
        except FileNotFoundError:
            print(f"错误：找不到权重文件 {local_weights_path}！模型将使用随机初始化的权重。")
        
        swin_output_dim = self.image_encoder_swin.num_features
        self.img_fc_mu = nn.Linear(swin_output_dim, latent_dim)
        self.img_fc_log_var = nn.Linear(swin_output_dim, latent_dim)

        # 1.2 空间编码器 (Spatial Encoder) - 使用 SVGP
        if inducing_points is None:
            raise ValueError("SVGP编码器必须提供 'inducing_points'！")
        self.feature_extractor = nn.Sequential(
            nn.Linear(inducing_points.size(-1), 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        self.svgp_spatial_encoder = SVGP_SpatialEncoder(
            inducing_points=inducing_points,
            latent_dim=self.latent_dim
        )

        # --- 解码器 (Decoders) ---
        # 2.1 图像解码器
        self.decoder_start_channels = 256
        # *** BUG FIX: 解码器的起始尺寸应与编码器最终输出的特征图尺寸匹配 ***
        self.decoder_start_size = self.img_size // 32 
        decoder_input_size = self.decoder_start_channels * (self.decoder_start_size ** 2)
        
        self.img_decoder_input = nn.Linear(latent_dim, decoder_input_size)
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        
        img_decoder_dims = hidden_dims[::-1]
        self.img_decoder_stages = nn.ModuleList()
        # 创建5个转置卷积层以从 H/32 恢复到 H
        for i in range(5): 
            in_chan = self.decoder_start_channels // (2**i)
            out_chan = self.decoder_start_channels // (2**(i+1)) if i < 4 else input_channels
            stage = nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2)
            self.img_decoder_stages.append(stage)

        self.img_final_activation = nn.Tanh()

        # 2.2 RNA 解码器
        self.rna_decoder_base = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256), # 使用LayerNorm替代BatchNorm1d
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.LayerNorm(512), # 使用LayerNorm替代BatchNorm1d
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.rna_decoder_mu = nn.Sequential(nn.Linear(512, rna_dim), nn.Softplus())
        self.rna_decoder_theta = nn.Sequential(nn.Linear(512, rna_dim), nn.Softplus())

    def fuse_latents(self, mu1, log_var1, mu2, log_var2):
        var1 = torch.exp(log_var1); var2 = torch.exp(log_var2)
        # 增加一个小的epsilon防止除以0
        var1 = var1 + 1e-8; var2 = var2 + 1e-8
        fused_var = 1 / (1/var1 + 1/var2)
        fused_log_var = torch.log(fused_var)
        fused_mu = (mu1/var1 + mu2/var2) * fused_var
        return fused_mu, fused_log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, image, spatial):
        # 1. 编码
        img_x = self.image_encoder_swin(image)
        mu_img = self.img_fc_mu(img_x)
        log_var_img = self.img_fc_log_var(img_x)
        spatial_feature = self.feature_extractor(spatial)
        spatial_dist = self.svgp_spatial_encoder(spatial)
        mu_spatial = spatial_dist.mean.transpose(-1, -2)
        log_var_spatial = spatial_dist.variance.transpose(-1, -2).log()
        
        # 2. 融合
        mu_fused, log_var_fused = self.fuse_latents(mu_img, log_var_img, mu_spatial, log_var_spatial)
        
        # 3. 重参数化
        z = self.reparameterize(mu_fused, log_var_fused)
        
        # 4. 解码
        # 图像解码
        img_recon_x = self.img_decoder_input(z)
        img_recon_x = img_recon_x.view(-1, self.decoder_start_channels, self.decoder_start_size, self.decoder_start_size)
        for i, stage in enumerate(self.img_decoder_stages):
            img_recon_x = stage(img_recon_x)
            if i < len(self.img_decoder_stages) -1: # 除了最后一层外都使用 LeakyReLU
                 img_recon_x = F.leaky_relu(img_recon_x, 0.2)
        reconstructed_image = self.img_final_activation(img_recon_x)
        
        # RNA解码
        rna_base = self.rna_decoder_base(z)
        mu_rna = self.rna_decoder_mu(rna_base)
        theta_rna = self.rna_decoder_theta(rna_base)
        reconstructed_rna_params = {"mu": mu_rna, "theta": theta_rna}

        return (reconstructed_image, reconstructed_rna_params, 
                mu_fused, log_var_fused, 
                mu_img, log_var_img,
                mu_spatial, spatial_feature, spatial_dist)

    # *** BUG FIX: 修正拼写错误 ***
    def dynamic_kld_weight(self, epoch, beta, kl_anneal_epochs=50):
        kld_weight = beta * min(1.0, epoch / kl_anneal_epochs) if kl_anneal_epochs > 0 else beta
        return kld_weight
        
    # *** BUG FIX: 更新此函数以使用GP编码器 ***
    def get_latent_representations(self, image, spatial):
        with torch.no_grad():
            img_x = self.image_encoder_swin(image)
            mu_img = self.img_fc_mu(img_x)
            log_var_img = self.img_fc_log_var(img_x)
            
            spatial_dist = self.svgp_spatial_encoder(spatial)
            mu_spatial = spatial_dist.mean.transpose(-1, -2)
            log_var_spatial = spatial_dist.variance.transpose(-1, -2).log()

            mu_fused, log_var_fused = self.fuse_latents(mu_img, log_var_img, mu_spatial, log_var_spatial)
            
            z_fused = self.reparameterize(mu_fused, log_var_fused)
            z_img = self.reparameterize(mu_img, log_var_img)
            z_spatial = self.reparameterize(mu_spatial, log_var_spatial)
            
            return z_fused, z_img, z_spatial
    
# ======================================================================
# --- 2. 混合损失函数 (已升级) ---
# ======================================================================
# ======================================================================
# --- 2. 混合损失函数 (已修正) ---
# ======================================================================
import torch
import torch.nn.functional as F

import torch

def kl_divergence_two_gaussians(mu1, log_var1, mu2, log_var2):
    """
    计算两个多元高斯分布（对角协方差）之间的KL散度。
    
    Args:
        mu1 (Tensor): 第一个分布的均值.
        log_var1 (Tensor): 第一个分布的对数方差.
        mu2 (Tensor): 第二个分布的均值.
        log_var2 (Tensor): 第二个分布的对数方差.
        
    Returns:
        Tensor: 两个分布之间的KL散度值。
    """
    var1 = log_var1.exp()
    var2 = log_var2.exp()
    
    kl_div = 0.5 * (log_var2 - log_var1 + (var1 + (mu1 - mu2).pow(2)) / var2 - 1)
    
    # 对最后一个维度求和，然后对批次求均值
    return kl_div.sum(dim=-1).mean()


def multi_modal_vae_loss(
    recon_img, true_img, 
    recon_rna_params, true_rna,
    mu_fused, log_var_fused, 
    mu_img, log_var_img,
    mu_spatial, 
    kld_weight=1.0, 
    image_weight=1.0, 
    rna_weight=1.0,
    alignment_weight=0.1
):
    """
    Revised VAE loss function for multi-modal data.
    
    Args:
        recon_img (Tensor): Reconstructed image tensor.
        true_img (Tensor): Original image tensor.
        recon_rna_params (Tensor): Reconstructed RNA parameters (e.g., logits for scellst_nll_loss).
        true_rna (Tensor): Original RNA tensor.
        mu_fused (Tensor): Mean of the fused latent distribution.
        log_var_fused (Tensor): Log variance of the fused latent distribution.
        mu_img (Tensor): Mean of the image latent distribution.
        log_var_img (Tensor): Log variance of the image latent distribution.
        mu_spatial (Tensor): Mean of the spatial latent distribution.
        log_var_spatial (Tensor): Log variance of the spatial latent distribution.
        kld_weight (float): Weight for the KL divergence term.
        image_weight (float): Weight for the image reconstruction loss.
        rna_weight (float): Weight for the RNA reconstruction loss.
        alignment_weight (float): Weight for the latent alignment loss.

    Returns:
        tuple: A tuple containing total loss and individual loss components.
    """
    
    batch_size = true_img.shape[0]

    # 1. Reconstruction Losses
    # Image Reconstruction Loss (MSE)
    image_recon_loss = F.mse_loss(recon_img, true_img, reduction='sum') / batch_size

    # RNA Reconstruction Loss (Negative Log-Likelihood)
    # The actual function `scellst_nll_loss` is assumed to be defined elsewhere.
    rna_recon_loss = scellst_nll_loss(recon_rna_params, true_rna)

    # 2. KL Divergence Losses
    # KL loss for the Fused Latent Space (against a standard normal prior)
    kld_fused = -0.5 * torch.sum(1 + log_var_fused - mu_fused.pow(2) - log_var_fused.exp()) / batch_size
    
    # Optional: KL losses for individual modalities
    # You may not need these if the fused KL is sufficient, but it's good practice
    # to be explicit about them for modularity.
    # kld_img = -0.5 * torch.sum(1 + log_var_img - mu_img.pow(2) - log_var_img.exp()) / batch_size
    # kld_spatial = -0.5 * torch.sum(1 + log_var_spatial - mu_spatial.pow(2) - log_var_spatial.exp()) / batch_size

    # 3. Alignment Loss
    # We use MSE to bring the mean of the image and spatial latent spaces closer.
    # A symmetric KL divergence (JSD) or an adversarial approach could also be used.
    alignment_loss = F.mse_loss(mu_img, mu_spatial, reduction='mean')
    # alignment_loss = kl_divergence_two_gaussians(
    #     mu_img, log_var_img,
    #     mu_spatial, log_var_spatial
    # )
    # 4. Total VAE Loss
    # We combine all losses with their respective weights.
    total_vae_loss = (
        image_weight * image_recon_loss +
        rna_weight * rna_recon_loss +
        kld_weight * kld_fused + # Use the fused KL loss
        alignment_weight * alignment_loss
    )

    return total_vae_loss, image_recon_loss, rna_recon_loss, kld_fused, alignment_loss

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
# ======================================================================
# --- 4. 训练流程 (已修正) ---
# ======================================================================
def train_multi_modal_vae(model, likelihood, train_loader, val_loader, epochs, learning_rate, device, **loss_weights):
    print("\n--- 开始 Multi-Modal VAE 模型训练 (SVGP Spatial Encoder版) ---")
    model.to(device)
    likelihood.to(device)

    optimizer = optim.AdamW([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=learning_rate)
    
    mll = gpytorch.mlls.VariationalELBO(likelihood, model.svgp_spatial_encoder, num_data=len(train_loader.dataset))

    best_val_corr = 0.0
    best_model_state = None
    beta = loss_weights.get('kld_weight', 1.0)

    for epoch in range(1, epochs + 1):
        model.train()
        likelihood.train()
        
        total_loss, total_img_loss, total_rna_loss, total_kld_loss, total_align_loss, total_gp_loss = 0, 0, 0, 0, 0, 0
        
        current_loss_weights = loss_weights.copy()
        current_loss_weights['kld_weight'] = model.dynamic_kld_weight(epoch, beta, kl_anneal_epochs=50)

        for (img_data, spatial_data, rna_data) in train_loader:
            img_data, spatial_data, rna_data = img_data.to(device), spatial_data.to(device), rna_data.to(device)
            
            optimizer.zero_grad()
            
            # *** MODIFIED: 接收模型返回的所有8个值 ***
            recon_img, recon_rna_params, mu_fused, log_var_fused, mu_img, log_var_img, mu_spatial, spatial_feature, spatial_dist = model(img_data, spatial_data)
            
            # --- 计算损失 ---
            # *** MODIFIED: 1. 从权重字典中安全地分离出 gp_loss_weight ***
            gp_weight = current_loss_weights.pop('gp_loss_weight', 0.1)
            
            # 2. 计算VAE部分损失 (现在 current_loss_weights 不再包含 gp_loss_weight)
            vae_loss, img_loss, rna_loss, kld_loss, align_loss = multi_modal_vae_loss(
                recon_img, img_data, recon_rna_params, rna_data, 
                mu_fused, log_var_fused, mu_img, log_var_img, mu_spatial, # <-- 传入 log_var_img
                **current_loss_weights
            )
            
            # 3. 计算GP的MLL损失
            # gp_loss = -mll(spatial_dist, mu_img.detach().transpose(-1, -2)).sum()
            
            # 4. 合并总损失
            loss = vae_loss

            loss.backward()
            optimizer.step()
            
            # 记录各个损失项...
            total_loss += loss.item(); total_img_loss += img_loss.item(); total_rna_loss += rna_loss.item()
            total_kld_loss += kld_loss.item(); total_align_loss += align_loss.item(); total_gp_loss += gp_loss.item()

        # --- 验证过程 (保持不变, 但注意模型返回值数量) ---
        model.eval()
        likelihood.eval()
        predicted_rna_list, true_rna_list = [], []
        with torch.no_grad():
            for (img_data, spatial_data, rna_data) in val_loader:
                img_data, spatial_data, rna_data = img_data.to(device), spatial_data.to(device), rna_data.to(device)
                
                # *** MODIFIED: 忽略所有不需要的返回值 (现在是8个) ***
                _, recon_rna_params, *_ = model(img_data, spatial_data)
                
                predicted_mu = recon_rna_params['mu']
                predicted_rna_list.append(predicted_mu.cpu().numpy())
                true_rna_list.append(rna_data.cpu().numpy())
        
        # ... (后续的指标计算和打印部分保持不变) ...
        predicted_rna_matrix = np.vstack(predicted_rna_list)
        true_rna_matrix = np.vstack(true_rna_list)
        spearman_corr = calculate_spearman_correlation(predicted_rna_matrix, true_rna_matrix)
        pearson_corr = calculate_median_pearson(predicted_rna_matrix, true_rna_matrix)
        mse = calculate_mse_col_normalized(predicted_rna_matrix, true_rna_matrix)
        
        print(f"Epoch: {epoch}/{epochs} | Loss: {total_loss/len(train_loader):.2f} "
              f"kld_w: {current_loss_weights['kld_weight']:.2f} "
              f"[Img: {total_img_loss/len(train_loader):.2f}, RNA: {total_rna_loss/len(train_loader):.2f}, "
              f"KLD_img: {total_kld_loss/len(train_loader):.2f}, Align: {total_align_loss/len(train_loader):.4f}, " # KLD->KLD_img
              f"GP: {total_gp_loss/len(train_loader):.2f}] | " 
              f"Val Spearman: {spearman_corr:.4f} | Val Pearson: {pearson_corr:.4f} | Val MSE: {mse:.4f}")
        
        if best_val_corr < pearson_corr:
            best_val_corr = pearson_corr
            best_model_state = model.state_dict()
            
    print("模型训练完成!")
    model.load_state_dict(best_model_state)
    return model