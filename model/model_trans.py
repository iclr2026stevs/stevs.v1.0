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

from timm.models.vision_transformer import VisionTransformer
import timm
from timm.layers import DropPath

from safetensors.torch import load_file
import warnings
from scipy.stats import ConstantInputWarning

warnings.filterwarnings("ignore", category=ConstantInputWarning)

local_weights_path = './model.safetensors'

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

class CrossAttentionFusion(nn.Module):
    def __init__(self, img_dim, spatial_dim, latent_dim, num_heads=4, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        self.latent_dim = latent_dim

        self.img_proj = nn.Linear(img_dim, latent_dim)
        self.spatial_proj = nn.Linear(spatial_dim, latent_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim)
        )

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_log_var = nn.Linear(latent_dim, latent_dim)

    def forward(self, img_features, spatial_features):
        img_features = img_features.unsqueeze(1)
        spatial_features = spatial_features.unsqueeze(1)

        img_proj = self.img_proj(img_features)
        spatial_proj = self.spatial_proj(spatial_features)

        attn_output, _ = self.attention(
            query=spatial_proj,
            key=img_proj,
            value=img_proj
        )

        fused_features = self.norm1(spatial_proj + attn_output)
        fused_features_ff = self.mlp(fused_features)
        fused_features = self.norm2(fused_features + fused_features_ff)

        fused_features = fused_features.squeeze(1)

        mu = self.fc_mu(fused_features)
        log_var = self.fc_log_var(fused_features)

        return mu, log_var

class MultiModalVAE(nn.Module):
    def __init__(self, input_channels=3, spatial_dim=2, latent_dim=128, img_size=32, rna_dim=2000, hidden_dims=None):
        super(MultiModalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.rna_dim = rna_dim

        self.image_encoder_swin = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            num_classes=0,
            img_size=self.img_size,
            in_chans=input_channels
        )

        try:
            print(f"Loading pretrained weights from local file: {local_weights_path}")

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
            print(f"Local weights loaded successfully! Matched and loaded {loaded_keys} / {total_keys} weight layers.")

        except FileNotFoundError:
            print(f"Error: Pretrained weight file {local_weights_path} not found! Please ensure the file is uploaded to the correct location. The model will use randomly initialized weights.")
        swin_output_dim = self.image_encoder_swin.num_features
        self.img_fc_mu = nn.Linear(swin_output_dim, latent_dim)
        self.img_fc_log_var = nn.Linear(swin_output_dim, latent_dim)
        self.fusion_module = CrossAttentionFusion(
            img_dim=swin_output_dim,
            spatial_dim=128,
            latent_dim=latent_dim
        )
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU()
        )
        self.spatial_fc_mu = nn.Linear(128, latent_dim)
        self.spatial_fc_log_var = nn.Linear(128, latent_dim)

        self.decoder_start_channels = 256
        self.decoder_start_size = img_size // 8
        decoder_input_size = self.decoder_start_channels * (self.decoder_start_size ** 2)

        self.img_decoder_input = nn.Linear(latent_dim, decoder_input_size)

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

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
        img_x = self.image_encoder_swin(image)
        mu_img = self.img_fc_mu(img_x)
        log_var_img = self.img_fc_log_var(img_x)

        spatial_x = self.spatial_encoder(spatial)

        mu_fused, log_var_fused = self.fusion_module(img_x, spatial_x)
        mu_img = self.img_fc_mu(img_x) if hasattr(self, 'img_fc_mu') else mu_fused
        mu_spatial = self.spatial_fc_mu(spatial_x) if hasattr(self, 'spatial_fc_mu') else mu_fused

        z = self.reparameterize(mu_fused, log_var_fused)

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

        return (reconstructed_image, reconstructed_rna_params,
                mu_fused, log_var_fused,
                mu_img, mu_spatial)

    def get_latent_representations(self, image, spatial):
        with torch.no_grad():
            img_x = self.image_encoder_swin(image)
            mu_img = self.img_fc_mu(img_x)
            log_var_img = self.img_fc_log_var(img_x)

            spatial_x = self.spatial_encoder(spatial)
            mu_spatial = self.spatial_fc_mu(spatial_x)
            log_var_spatial = self.spatial_fc_log_var(spatial_x)

            mu_fused, log_var_fused = self.fuse_latents(mu_img, log_var_img, mu_spatial, log_var_spatial)

            z_fused = self.reparameterize(mu_fused, log_var_fused)
            z_img = self.reparameterize(mu_img, log_var_img)
            z_spatial = self.reparameterize(mu_spatial, log_var_spatial)

            return z_fused, z_img, z_spatial

    def Dynomic_kld_weight(self, epoch, beta, kl_anneal_epochs=50):
        kld_weight = beta * min(1.0, epoch / kl_anneal_epochs) if kl_anneal_epochs > 0 else beta
        return kld_weight

def multi_modal_vae_loss(recon_img, true_img, recon_rna_params, true_rna,
                         mu_fused, log_var_fused, mu_img, mu_spatial,
                         kld_weight=1.0, image_weight=1.0, rna_weight=1.0,
                         alignment_weight=0.1):
    image_recon_loss = F.mse_loss(recon_img, true_img, reduction='sum') / true_img.shape[0]

    rna_recon_loss = scellst_nll_loss(recon_rna_params, true_rna)

    kld_loss = -0.5 * torch.sum(1 + log_var_fused - mu_fused.pow(2) - log_var_fused.exp()) / true_img.shape[0]

    alignment_loss = F.mse_loss(mu_img, mu_spatial, reduction='mean')

    total_loss = (image_weight * image_recon_loss +
                  rna_weight * rna_recon_loss +
                  kld_weight * kld_loss +
                  alignment_weight * alignment_loss)

    return total_loss, image_recon_loss, rna_recon_loss, kld_loss, alignment_loss

import numpy as np
import torch
from scipy.sparse import issparse
from torch.utils.data import TensorDataset, DataLoader, Subset

from torchvision import transforms

def prepare_adata_loader(adata_img, adata_rna, layer_key_img, shape_key_img, layer_key_rna, spatial_key, batch_size, ratio=(0.7, 0.1, 0.2), Train_shuffle=True):
    print("--- Starting to prepare multi-modal data ---")

    if not np.isclose(np.sum(ratio), 1.0):
        raise ValueError(f"The sum of ratios must be 1, but got {np.sum(ratio)}")

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
    print(f"Image data: {image_tensor.shape}, Spatial data: {spatial_tensor.shape}, RNA data: {rna_tensor.shape}")
    dataset = TensorDataset(image_tensor, spatial_tensor, rna_tensor)
    n_samples = len(dataset)

    train_ratio, val_ratio, test_ratio = ratio
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    test_size = n_samples - train_size - val_size

    indices = list(range(n_samples))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=Train_shuffle) if len(train_dataset) > 0 else None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if len(val_dataset) > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if len(test_dataset) > 0 else None

    print(f"Data preparation complete. Training set: {len(train_dataset)}, Validation set: {len(val_dataset)}, Test set: {len(test_dataset)}")

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

def train_multi_modal_vae(model, train_loader, val_loader, epochs, learning_rate, device, **loss_weights):
    print("\n--- Starting Multi-Modal VAE Model Training (Swin Transformer version) ---")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    best_val = 0.0
    best_model = model
    beta = loss_weights.get('kld_weight', 1.0)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_img_loss, total_rna_loss, total_kld_loss, total_align_loss = 0, 0, 0, 0, 0

        current_loss_weights = loss_weights.copy()
        current_loss_weights['kld_weight'] = model.Dynomic_kld_weight(epoch, beta, kl_anneal_epochs=50)

        for (img_data, spatial_data, rna_data) in train_loader:
            img_data, spatial_data, rna_data = img_data.to(device), spatial_data.to(device), rna_data.to(device)

            recon_img, recon_rna_params, mu_fused, log_var_fused, mu_img, mu_spatial = model(img_data, spatial_data)

            loss, img_loss, rna_loss, kld_loss, align_loss = multi_modal_vae_loss(
                recon_img, img_data, recon_rna_params, rna_data,
                mu_fused, log_var_fused, mu_img, mu_spatial,
                **current_loss_weights
            )

            optimizer.zero_grad(); loss.backward(); optimizer.step()

            total_loss += loss.item(); total_img_loss += img_loss.item()
            total_rna_loss += rna_loss.item(); total_kld_loss += kld_loss.item()
            total_align_loss += align_loss.item()

        model.eval()
        predicted_rna_list, true_rna_list = [], []
        with torch.no_grad():
            for (img_data, spatial_data, rna_data) in val_loader:
                img_data, spatial_data, rna_data = img_data.to(device), spatial_data.to(device), rna_data.to(device)
                _, recon_rna_params, _, _, _, _ = model(img_data, spatial_data)
                predicted_mu = recon_rna_params['mu']
                predicted_rna_list.append(predicted_mu.cpu().numpy())
                true_rna_list.append(rna_data.cpu().numpy())

        predicted_rna_matrix = np.vstack(predicted_rna_list)
        true_rna_matrix = np.vstack(true_rna_list)
        spearman_corr = calculate_spearman_correlation(predicted_rna_matrix, true_rna_matrix, axis=1)
        pearson_corr = calculate_median_pearson(predicted_rna_matrix, true_rna_matrix)
        mse = calculate_mse_col_normalized(predicted_rna_matrix, true_rna_matrix)

        print(f"Epoch: {epoch}/{epochs} | Loss: {total_loss/len(train_loader):.2f} "
              f"kld_w: {current_loss_weights['kld_weight']:.2f} "
              f"[Img: {total_img_loss/len(train_loader):.2f}, RNA: {total_rna_loss/len(train_loader):.2f}, "
              f"KLD: {total_kld_loss/len(train_loader):.2f}, Align: {total_align_loss/len(train_loader):.4f}] | "
              f"Val Spearman: {spearman_corr:.4f} | Val Pearson: {pearson_corr:.4f} | Val MSE: {mse:.4f}")

        if best_val < pearson_corr:
            best_val = pearson_corr
            best_model = model

    print("Model training complete!")
    return best_model
