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

from safetensors.torch import load_file # *** NEW: import safetensors load function ***
import warnings
from scipy.stats import ConstantInputWarning

# Ignore ConstantInputWarning
warnings.filterwarnings("ignore", category=ConstantInputWarning)

local_weights_path = './model.safetensors'

import torch
from torch import nn
import numpy as np

class FourierFeatureMapping(nn.Module):
    """
    Maps low-dimensional input coordinates to a high-dimensional Fourier feature space.
    """
    def __init__(self, input_dim: int, embedding_dim: int, scale: float = 1.0):
        """
        Args:
            input_dim (int): Input dimension, e.g., 2 for (x, y).
            embedding_dim (int): Output dimension, must be an even number.
            scale (float): Frequency scale of the Fourier features.
        """
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be an even number.")

        # Create a learnable Fourier frequency matrix B
        self.B = nn.Parameter(
            torch.randn(input_dim, embedding_dim // 2) * scale,
            requires_grad=True # The frequency matrix is learnable
        )
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input coordinates with shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Fourier features with shape (batch_size, embedding_dim).
        """
        # (x @ self.B) has shape (batch_size, embedding_dim // 2)
        # Multiply by 2π and compute sin and cos
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# --- VAE base modules (unchanged) ---
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

# --- Negative Binomial distribution components (unchanged) ---
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


class MultiModalVAE(nn.Module):
    def __init__(self, input_channels=3, spatial_dim=2, latent_dim=128, img_size=32, rna_dim=2000, fourier_feature_dim=128, hidden_dims=None):
        super(MultiModalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.rna_dim = rna_dim

        # --- Encoders ---

        # *** MODIFIED: 1.1 Image Encoder - Using Swin Transformer ***
        # Load a pretrained, small Swin Transformer.
        # num_classes=0 means we only want features, not classification results.
        self.image_encoder_swin = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False, # Key: Must be False here because we're not connected to the internet
            num_classes=0,
            img_size=self.img_size,
            in_chans=input_channels
        )

        # 3. Load weights from local file and apply to the model
        # *** MODIFIED: Use safetensors.torch.load_file to load weights ***
        # *** MODIFIED: Use a more flexible way to load weights, to handle size mismatch issues ***
        try:
            print(f"Loading pretrained weights from local file: {local_weights_path}")

            # 1. Load pretrained weights
            pretrained_dict = load_file(local_weights_path, device='cpu')

            # 2. Get the state dictionary of your current model
            model_dict = self.image_encoder_swin.state_dict()

            # 3. Filter pretrained weights:
            #    - Only keep layers that also exist in your current model (handles "Unexpected key" issues)
            #    - Only keep layers with an exact shape match (handles "size mismatch" issues)
            pretrained_dict_filtered = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict and model_dict[k].shape == v.shape
            }

            # 4. Update your current model's state dictionary with the filtered weights
            model_dict.update(pretrained_dict_filtered)

            # 5. Load the updated state dictionary back into the model
            self.image_encoder_swin.load_state_dict(model_dict)

            loaded_keys = len(pretrained_dict_filtered)
            total_keys = len(model_dict)
            print(f"Local weights loaded successfully! Matched and loaded {loaded_keys} / {total_keys} weight layers.")

        except FileNotFoundError:
            print(f"Error: Pretrained weight file {local_weights_path} not found! Please ensure the file is uploaded to the correct location. The model will use randomly initialized weights.")
        # Get the feature dimension of the Swin output
        swin_output_dim = self.image_encoder_swin.num_features
        self.img_fc_mu = nn.Linear(swin_output_dim, latent_dim)
        self.img_fc_log_var = nn.Linear(swin_output_dim, latent_dim)
        # self.fourier_mapper = FourierFeatureMapping(
        #     input_dim=spatial_dim,
        #     embedding_dim=fourier_feature_dim # Output dimension is 128
        # )
        # 1.2 Spatial Position Encoder (unchanged)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU()
        )
        self.spatial_fc_mu = nn.Linear(128, latent_dim)
        self.spatial_fc_log_var = nn.Linear(128, latent_dim)

        # --- Decoders ---

        # *** MODIFIED: 2.1 Image Decoder - Adapting to Swin Encoder output ***
        # To make the decoder work, we need to define the "feature map" shape at the beginning of the decoder
        self.decoder_start_channels = 256
        self.decoder_start_size = img_size // 8 # e.g., starting from 32x32 -> 4x4
        decoder_input_size = self.decoder_start_channels * (self.decoder_start_size ** 2)

        self.img_decoder_input = nn.Linear(latent_dim, decoder_input_size)

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256] # Retain for decoder structure

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

        # *** MODIFIED: 2.2 RNA Decoder - Enhanced Version ***
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
        # 1. Encoding
        # Image encoding (Swin)
        img_x = self.image_encoder_swin(image).detach()
        mu_img = self.img_fc_mu(img_x)
        log_var_img = self.img_fc_log_var(img_x)

        # Spatial encoding (MLP)
        # spatial_fourier = self.fourier_mapper(spatial)
        spatial_x = self.spatial_encoder(spatial)
        mu_spatial = self.spatial_fc_mu(spatial_x)
        log_var_spatial = self.spatial_fc_log_var(spatial_x)

        # 2. Fusing latent spaces (PoE)
        mu_fused, log_var_fused = self.fuse_latents(mu_img, log_var_img, mu_spatial, log_var_spatial)

        # 3. Reparameterization
        z = self.reparameterize(mu_fused, log_var_fused)

        # 4. Decoding
        # Image decoding
        img_recon_x = self.img_decoder_input(z)
        img_recon_x = img_recon_x.view(-1, self.decoder_start_channels, self.decoder_start_size, self.decoder_start_size)
        for stage in self.img_decoder_stages:
            img_recon_x = stage(img_recon_x)
        reconstructed_image = self.img_final_layer(img_recon_x)

        # RNA decoding
        # BatchNorm1d can raise an error with batch_size=1, special handling is needed
        if z.shape[0] > 1:
            rna_base = self.rna_decoder_base(z)
        else:
            # Temporarily switch to eval mode to disable BN and Dropout
            original_mode = self.rna_decoder_base.training
            self.rna_decoder_base.eval()
            with torch.no_grad():
                rna_base = self.rna_decoder_base(z)
            self.rna_decoder_base.train(original_mode)

        mu_rna = self.rna_decoder_mu(rna_base)
        theta_rna = self.rna_decoder_theta(rna_base)
        reconstructed_rna_params = {"mu": mu_rna, "theta": theta_rna}
        z_img, z_spatial = self.reparameterize(mu_img, log_var_img), self.reparameterize(mu_spatial, log_var_spatial)
        # *** MODIFIED: Return all latent variables for the new loss calculation ***
        return (reconstructed_image, reconstructed_rna_params,
                mu_fused, log_var_fused,
                z_img, z_spatial)

    def get_latent_representations(self, image, spatial):
        """
        Takes the same inputs as the forward pass but only returns the latent vectors from different levels.
        This is useful for latent space visualization and analysis.

        Args:
            image (torch.Tensor): Input image tensor.
            spatial (torch.Tensor): Input spatial coordinate tensor.

        Returns:
            tuple: A tuple containing three latent vectors:
                - z_fused (torch.Tensor): The fused latent vector.
                - z_img (torch.Tensor): The image-only latent vector.
                - z_spatial (torch.Tensor): The spatial-only latent vector.
        """
        with torch.no_grad(): # Operate in inference mode, no gradient calculation
            # 1. Encoding process (same as the forward pass)
            # Image encoding
            img_x = self.image_encoder_swin(image).detach()
            mu_img = self.img_fc_mu(img_x)
            log_var_img = self.img_fc_log_var(img_x)

            # Spatial encoding
            # spatial_fourier = self.fourier_mapper(spatial)
            spatial_x = self.spatial_encoder(spatial)
            mu_spatial = self.spatial_fc_mu(spatial_x)
            log_var_spatial = self.spatial_fc_log_var(spatial_x)

            # 2. Fusion
            mu_fused, log_var_fused = self.fuse_latents(mu_img, log_var_img, mu_spatial, log_var_spatial)

            # 3. Reparameterization sampling from the three distributions
            z_fused = self.reparameterize(mu_fused, log_var_fused)
            z_img = self.reparameterize(mu_img, log_var_img)
            z_spatial = self.reparameterize(mu_spatial, log_var_spatial)

            return z_fused, z_img, z_spatial

    def Dynomic_kld_weight(self, epoch, beta, kl_anneal_epochs=50):
        kld_weight = beta * min(1.0, epoch / kl_anneal_epochs) if kl_anneal_epochs > 0 else beta
        return kld_weight


def multi_modal_vae_loss(recon_img, true_img, recon_rna_params, true_rna,
                         mu_fused, log_var_fused, z_img, z_spatial,
                         kld_weight=1.0, image_weight=1.0, rna_weight=1.0,
                         alignment_weight=0.1): # *** NEW: Add alignment weight ***
    # Image reconstruction loss (MSE)
    image_recon_loss = F.mse_loss(recon_img, true_img, reduction='sum') / true_img.shape[0]

    # RNA reconstruction loss (NLL)
    rna_recon_loss = scellst_nll_loss(recon_rna_params, true_rna)

    # KL divergence loss (for the fused distribution)
    kld_loss = -0.5 * torch.sum(1 + log_var_fused - mu_fused.pow(2) - log_var_fused.exp()) / true_img.shape[0]

    # *** NEW: Latent space alignment loss ***
    alignment_loss = F.mse_loss(z_img, z_spatial, reduction='mean') # / true_img.shape[0]

    # Weighted total loss
    total_loss = (image_weight * image_recon_loss +
                  rna_weight * rna_recon_loss +
                  kld_weight * kld_loss +
                  alignment_weight * alignment_loss) # *** NEW ***

    return total_loss, image_recon_loss, rna_recon_loss, kld_loss, alignment_loss # *** MODIFIED ***

import numpy as np
import torch
from scipy.sparse import issparse
from torch.utils.data import TensorDataset, DataLoader, Subset


def prepare_adata_loader(adata_img, adata_rna, layer_key_img, shape_key_img, layer_key_rna, spatial_key, batch_size, ratio=(0.7, 0.1, 0.2), Train_shuffle=True):
    """
    Prepares multi-modal data loaders.
    This version uses numpy.random.shuffle to ensure data splitting is exactly consistent with another model.
    """
    print("--- Starting to prepare multi-modal data ---")

    # --- Data preprocessing part (no changes) ---
    if not np.isclose(np.sum(ratio), 1.0):
        raise ValueError(f"The sum of ratios must be 1, but got {np.sum(ratio)}")

    # ... (This part of the data preparation code is the same as before, omitted for brevity) ...
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

    # --- Data splitting (modification here) ---
    # 1. Calculate the size of each set (consistent with the target model's logic)
    train_ratio, val_ratio, test_ratio = ratio
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    test_size = n_samples - train_size - val_size

    # 2. Use NumPy's method for random index splitting (exactly consistent with the target model)
    # --- start of modification ---
    indices = list(range(n_samples))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size:]
    # --- end of modification ---

    # 3. Create PyTorch Subsets using the split indices (no changes)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # --- Create DataLoader part (no changes) ---
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
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate) # *** MODIFIED: Use AdamW ***
    best_val = 0.0
    best_model = model
    beta = loss_weights.get('kld_weight', 1.0)

    for epoch in range(1, epochs + 1):
        model.train()
        # *** MODIFIED: Add logging for alignment_loss ***
        total_loss, total_img_loss, total_rna_loss, total_kld_loss, total_align_loss = 0, 0, 0, 0, 0

        current_loss_weights = loss_weights.copy()
        current_loss_weights['kld_weight'] = model.Dynomic_kld_weight(epoch, beta, kl_anneal_epochs=50)

        for (img_data, spatial_data, rna_data) in train_loader:
            img_data, spatial_data, rna_data = img_data.to(device), spatial_data.to(device), rna_data.to(device)

            # *** MODIFIED: Receive all latent variables returned by the model ***
            recon_img, recon_rna_params, mu_fused, log_var_fused, z_img, z_spatial = model(img_data, spatial_data)

            # *** MODIFIED: Pass all variables to the loss function ***
            loss, img_loss, rna_loss, kld_loss, align_loss = multi_modal_vae_loss(
                recon_img, img_data, recon_rna_params, rna_data,
                mu_fused, log_var_fused, z_img, z_spatial,
                **current_loss_weights
            )

            optimizer.zero_grad(); loss.backward(); optimizer.step()

            total_loss += loss.item(); total_img_loss += img_loss.item()
            total_rna_loss += rna_loss.item(); total_kld_loss += kld_loss.item()
            total_align_loss += align_loss.item() # *** NEW ***

        # --- Validation process ---
        model.eval()
        predicted_rna_list, true_rna_list = [], []
        with torch.no_grad():
            for (img_data, spatial_data, rna_data) in val_loader:
                img_data, spatial_data, rna_data = img_data.to(device), spatial_data.to(device), rna_data.to(device)
                # *** MODIFIED: Ignore unneeded return values ***
                _, recon_rna_params, _, _, _, _ = model(img_data, spatial_data)
                predicted_mu = recon_rna_params['mu']
                predicted_rna_list.append(predicted_mu.cpu().numpy())
                true_rna_list.append(rna_data.cpu().numpy())

        predicted_rna_matrix = np.vstack(predicted_rna_list)
        true_rna_matrix = np.vstack(true_rna_list)
        spearman_corr = calculate_spearman_correlation(predicted_rna_matrix, true_rna_matrix, axis=1)
        pearson_corr = calculate_median_pearson(predicted_rna_matrix, true_rna_matrix)
        mse = calculate_mse_col_normalized(predicted_rna_matrix, true_rna_matrix)
        # *** MODIFIED: Update print information ***
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
