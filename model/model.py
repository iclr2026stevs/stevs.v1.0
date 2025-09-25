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
class MultiModalVAE(nn.Module):
    def __init__(self, input_channels=3, spatial_dim=2, latent_dim=128, img_size=32, rna_dim=2000, fourier_feature_dim=128, hidden_dims=None):
        super(MultiModalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.rna_dim = rna_dim

        # --- 编码器 (Encoders) ---
        
        # *** MODIFIED: 1.1 图像编码器 (Image Encoder) - 使用 Swin Transformer ***
        # 加载预训练的、小型的Swin Transformer。
        # num_classes=0 表示我们只想要特征，而不是分类结果。
        self.image_encoder_swin = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False, # 关键：这里必须是 False，因为我们不联网
            num_classes=0,
            img_size=self.img_size,
            in_chans=input_channels
        )

        # 3. 从本地文件加载权重并应用到模型上
        # *** MODIFIED: 使用 safetensors.torch.load_file 来加载权重 ***
        # *** MODIFIED: 使用更灵活的方式加载权重，以处理尺寸不匹配的问题 ***
        try:
            print(f"正在从本地文件加载预训练权重: {local_weights_path}")
            
            # 1. 加载预训练权重
            pretrained_dict = load_file(local_weights_path, device='cpu')
            
            # 2. 获取你当前模型的权重字典
            model_dict = self.image_encoder_swin.state_dict()
            
            # 3. 过滤预训练权重：
            #    - 只保留那些在你当前模型中也存在的层（处理 "Unexpected key" 问题）
            #    - 只保留那些形状完全匹配的层（处理 "size mismatch" 问题）
            pretrained_dict_filtered = {
                k: v for k, v in pretrained_dict.items() 
                if k in model_dict and model_dict[k].shape == v.shape
            }
            
            # 4. 用过滤后的权重更新你当前模型的权重字典
            model_dict.update(pretrained_dict_filtered)
            
            # 5. 将更新后的权重字典加载回模型
            self.image_encoder_swin.load_state_dict(model_dict)
            
            loaded_keys = len(pretrained_dict_filtered)
            total_keys = len(model_dict)
            print(f"本地权重加载成功！匹配并加载了 {loaded_keys} / {total_keys} 个权重层。")

        except FileNotFoundError:
            print(f"错误：找不到权重文件 {local_weights_path}！请确保文件已上传到正确的位置。模型将使用随机初始化的权重。")
        # 获取Swin输出的特征维度
        swin_output_dim = self.image_encoder_swin.num_features
        self.img_fc_mu = nn.Linear(swin_output_dim, latent_dim)
        self.img_fc_log_var = nn.Linear(swin_output_dim, latent_dim)
        # self.fourier_mapper = FourierFeatureMapping(
        #     input_dim=spatial_dim, 
        #     embedding_dim=fourier_feature_dim # 输出维度为128
        # )
        # 1.2 空间位置编码器 (Spatial Encoder) - 保持不变
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU()
        )
        self.spatial_fc_mu = nn.Linear(128, latent_dim)
        self.spatial_fc_log_var = nn.Linear(128, latent_dim)

        # --- 解码器 (Decoders) ---
        
        # *** MODIFIED: 2.1 图像解码器 (Image Decoder) - 适配 Swin Encoder 输出 ***
        # 为了让解码器工作，我们需要定义一个解码器开始时的“特征图”形状
        self.decoder_start_channels = 256
        self.decoder_start_size = img_size // 8 # 例如，从 32x32 -> 4x4 开始
        decoder_input_size = self.decoder_start_channels * (self.decoder_start_size ** 2)
        
        self.img_decoder_input = nn.Linear(latent_dim, decoder_input_size)
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256] # 保留用于解码器结构
        
        img_decoder_dims = hidden_dims[::-1] # [256, 128, 64, 32]
        self.img_decoder_stages = nn.ModuleList()
        for i in range(len(img_decoder_dims) - 1):
            stage = nn.Sequential(
                nn.ConvTranspose2d(img_decoder_dims[i], img_decoder_dims[i+1], kernel_size=2, stride=2),
                nn.BatchNorm2d(img_decoder_dims[i+1]), nn.LeakyReLU(inplace=True),
                ConvBlock(dim=img_decoder_dims[i+1], mlp_ratio=4.)
            )
            self.img_decoder_stages.append(stage)
        self.img_final_layer = nn.Sequential(
            # A 3x3 convolution with padding=1 and stride=1 preserves the HxW dimensions
            nn.Conv2d(img_decoder_dims[-1], input_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        # *** MODIFIED: 2.2 RNA 解码器 (RNA Decoder) - 增强版 ***
        self.rna_decoder_base = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.rna_decoder_mu = nn.Sequential(nn.Linear(512, rna_dim), nn.Softplus())
        self.rna_decoder_theta = nn.Sequential(nn.Linear(512, rna_dim), nn.Softplus())

    def fuse_latents(self, mu1, log_var1, mu2, log_var2):
        var1 = torch.exp(log_var1); var2 = torch.exp(log_var2)
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
        # 图像编码 (Swin)
        img_x = self.image_encoder_swin(image).detach()
        mu_img = self.img_fc_mu(img_x)
        log_var_img = self.img_fc_log_var(img_x)
        
        # 空间编码 (MLP)
        # spatial_fourier = self.fourier_mapper(spatial)
        spatial_x = self.spatial_encoder(spatial)
        mu_spatial = self.spatial_fc_mu(spatial_x)
        log_var_spatial = self.spatial_fc_log_var(spatial_x)
        
        # 2. 融合潜在空间 (PoE)
        mu_fused, log_var_fused = self.fuse_latents(mu_img, log_var_img, mu_spatial, log_var_spatial)
        
        # 3. 重参数化
        z = self.reparameterize(mu_fused, log_var_fused)
        
        # 4. 解码
        # 图像解码
        img_recon_x = self.img_decoder_input(z)
        img_recon_x = img_recon_x.view(-1, self.decoder_start_channels, self.decoder_start_size, self.decoder_start_size)
        for stage in self.img_decoder_stages:
            img_recon_x = stage(img_recon_x)
        reconstructed_image = self.img_final_layer(img_recon_x)
        
        # RNA解码
        # BatchNorm1d 在 batch_size=1 时会报错，需要特殊处理
        if z.shape[0] > 1:
            rna_base = self.rna_decoder_base(z)
        else:
            # 临时切换到eval模式以禁用BN和Dropout
            original_mode = self.rna_decoder_base.training
            self.rna_decoder_base.eval()
            with torch.no_grad():
                rna_base = self.rna_decoder_base(z)
            self.rna_decoder_base.train(original_mode)

        mu_rna = self.rna_decoder_mu(rna_base)
        theta_rna = self.rna_decoder_theta(rna_base)
        reconstructed_rna_params = {"mu": mu_rna, "theta": theta_rna}
        z_img, z_spatial = self.reparameterize(mu_img, log_var_img), self.reparameterize(mu_spatial, log_var_spatial)
        # *** MODIFIED: 返回所有潜在变量用于计算新损失 ***
        return (reconstructed_image, reconstructed_rna_params, 
                mu_fused, log_var_fused, 
                z_img, z_spatial)

    def get_latent_representations(self, image, spatial):
        """
        输入与 forward 函数相同，但只返回不同层级的潜向量。
        这对于潜空间的可视化和分析非常有用。
        
        Args:
            image (torch.Tensor): 图像输入张量。
            spatial (torch.Tensor): 空间坐标输入张量。
            
        Returns:
            tuple: 包含三个潜向量的元组:
                - z_fused (torch.Tensor): 融合后的潜向量。
                - z_img (torch.Tensor): 仅图像模态的潜向量。
                - z_spatial (torch.Tensor): 仅空间模态的潜向量。
        """
        with torch.no_grad(): # 在推理模式下进行，不计算梯度
            # 1. 编码过程（与 forward 函数相同）
            # 图像编码
            img_x = self.image_encoder_swin(image).detach()
            mu_img = self.img_fc_mu(img_x)
            log_var_img = self.img_fc_log_var(img_x)
            
            # 空间编码
            # spatial_fourier = self.fourier_mapper(spatial)
            spatial_x = self.spatial_encoder(spatial)
            mu_spatial = self.spatial_fc_mu(spatial_x)
            log_var_spatial = self.spatial_fc_log_var(spatial_x)
            
            # 2. 融合
            mu_fused, log_var_fused = self.fuse_latents(mu_img, log_var_img, mu_spatial, log_var_spatial)
            
            # 3. 分别从三个分布中进行重参数化采样
            z_fused = self.reparameterize(mu_fused, log_var_fused)
            z_img = self.reparameterize(mu_img, log_var_img)
            z_spatial = self.reparameterize(mu_spatial, log_var_spatial)
            
            return z_fused, z_img, z_spatial
    
    def Dynomic_kld_weight(self, epoch, beta, kl_anneal_epochs=50):
        kld_weight = beta * min(1.0, epoch / kl_anneal_epochs) if kl_anneal_epochs > 0 else beta
        return kld_weight
    
# ======================================================================
# --- 2. 混合损失函数 (已升级) ---
# ======================================================================
def multi_modal_vae_loss(recon_img, true_img, recon_rna_params, true_rna, 
                         mu_fused, log_var_fused, z_img, z_spatial,
                         kld_weight=1.0, image_weight=1.0, rna_weight=1.0, 
                         alignment_weight=0.1): # *** NEW: 新增对齐权重 ***
    # 图像重建损失 (MSE)
    image_recon_loss = F.mse_loss(recon_img, true_img, reduction='sum') / true_img.shape[0]

    # RNA 重建损失 (NLL)
    rna_recon_loss = scellst_nll_loss(recon_rna_params, true_rna)
    
    # KL 散度损失 (针对融合后的分布)
    kld_loss = -0.5 * torch.sum(1 + log_var_fused - mu_fused.pow(2) - log_var_fused.exp()) / true_img.shape[0]
    
    # *** NEW: 潜在空间对齐损失 (Alignment Loss) ***
    alignment_loss = F.mse_loss(z_img, z_spatial, reduction='mean') # / true_img.shape[0]
    
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
def train_multi_modal_vae(model, train_loader, val_loader, epochs, learning_rate, device, **loss_weights):
    print("\n--- 开始 Multi-Modal VAE 模型训练 (Swin Transformer版) ---")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate) # *** MODIFIED: 使用 AdamW ***
    best_val = 0.0
    best_model = model
    beta = loss_weights.get('kld_weight', 1.0) 

    for epoch in range(1, epochs + 1):
        model.train()
        # *** MODIFIED: 增加 alignment_loss 的记录 ***
        total_loss, total_img_loss, total_rna_loss, total_kld_loss, total_align_loss = 0, 0, 0, 0, 0
        
        current_loss_weights = loss_weights.copy()
        current_loss_weights['kld_weight'] = model.Dynomic_kld_weight(epoch, beta, kl_anneal_epochs=50)

        for (img_data, spatial_data, rna_data) in train_loader:
            img_data, spatial_data, rna_data = img_data.to(device), spatial_data.to(device), rna_data.to(device)
            
            # *** MODIFIED: 接收模型返回的所有潜在变量 ***
            recon_img, recon_rna_params, mu_fused, log_var_fused, z_img, z_spatial = model(img_data, spatial_data)
            
            # *** MODIFIED: 将所有变量传递给损失函数 ***
            loss, img_loss, rna_loss, kld_loss, align_loss = multi_modal_vae_loss(
                recon_img, img_data, recon_rna_params, rna_data, 
                mu_fused, log_var_fused, z_img, z_spatial,
                **current_loss_weights
            )
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            total_loss += loss.item(); total_img_loss += img_loss.item()
            total_rna_loss += rna_loss.item(); total_kld_loss += kld_loss.item()
            total_align_loss += align_loss.item() # *** NEW ***

        # --- 验证过程 ---
        model.eval()
        predicted_rna_list, true_rna_list = [], []
        with torch.no_grad():
            for (img_data, spatial_data, rna_data) in val_loader:
                img_data, spatial_data, rna_data = img_data.to(device), spatial_data.to(device), rna_data.to(device)
                # *** MODIFIED: 忽略不需要的返回值 ***
                _, recon_rna_params, _, _, _, _ = model(img_data, spatial_data)
                predicted_mu = recon_rna_params['mu']
                predicted_rna_list.append(predicted_mu.cpu().numpy())
                true_rna_list.append(rna_data.cpu().numpy())
        
        predicted_rna_matrix = np.vstack(predicted_rna_list)
        true_rna_matrix = np.vstack(true_rna_list)
        spearman_corr = calculate_spearman_correlation(predicted_rna_matrix, true_rna_matrix, axis=1)
        pearson_corr = calculate_median_pearson(predicted_rna_matrix, true_rna_matrix)
        mse = calculate_mse_col_normalized(predicted_rna_matrix, true_rna_matrix)
        # *** MODIFIED: 更新打印信息 ***
        print(f"Epoch: {epoch}/{epochs} | Loss: {total_loss/len(train_loader):.2f} "
              f"kld_w: {current_loss_weights['kld_weight']:.2f} "
              f"[Img: {total_img_loss/len(train_loader):.2f}, RNA: {total_rna_loss/len(train_loader):.2f}, "
              f"KLD: {total_kld_loss/len(train_loader):.2f}, Align: {total_align_loss/len(train_loader):.4f}] | "
              f"Val Spearman: {spearman_corr:.4f} | Val Pearson: {pearson_corr:.4f} | Val MSE: {mse:.4f}")
        if best_val < pearson_corr:
            best_val = pearson_corr
            best_model = model
    print("模型训练完成!")
    return best_model