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
import os
# --- 依赖库 (timm是新增的) ---
from timm.models.vision_transformer import VisionTransformer
import timm # *** NEW ***
from timm.layers import DropPath

from safetensors.torch import load_file # *** NEW: 导入safetensors的加载函数 ***


def calculate_median_pearson_evaluate(y_pred, y_true):
    """
    计算所有基因的Pearson相关系数（NaN值替换为0），并返回完整相关系数列表与其中位数
    
    参数:
        y_pred: 预测的RNA矩阵（shape: [n_cells, n_genes]），支持torch.Tensor或numpy.ndarray
        y_true: 真实的RNA矩阵（shape: [n_cells, n_genes]），支持torch.Tensor或numpy.ndarray
    
    返回:
        correlations: 所有基因的Pearson相关系数列表（NaN已替换为0，shape: [n_genes,]）
        median_corr: 相关系数的中位数（基于替换后的数据计算，方便快速评估整体性能）
    """
    # 1. 处理输入格式：若为torch张量，转为CPU上的numpy数组
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    
    # 2. 校验输入矩阵形状（确保细胞数、基因数一致，避免计算错误）
    if y_pred.shape != y_true.shape:
        raise ValueError(f"预测矩阵与真实矩阵形状不匹配！y_pred: {y_pred.shape}, y_true: {y_true.shape}")
    
    # 3. 逐个基因（按列）计算Pearson相关系数
    correlations = []
    n_genes = y_pred.shape[1]  # 基因总数（矩阵列数）
    for i in range(n_genes):
        # 提取第i个基因的预测值和真实值（按列取，对应单个基因的所有细胞表达量）
        pred_gene = y_pred[:, i]
        true_gene = y_true[:, i]
        
        # 计算Pearson相关系数（pearsonr返回值为：(相关系数, p值)，取第0个元素即相关系数）
        corr, _ = pearsonr(pred_gene, true_gene)
        correlations.append(corr)
    
    # 4. 将列表转为numpy数组，并将NaN值替换为0
    correlations = np.array(correlations)
    median_corr = np.median(correlations[~np.isnan(correlations)])
    correlations[np.isnan(correlations)] = 0  # 核心修改：NaN -> 0
    
    # 5. 计算替换后相关系数的中位数（保留原功能，方便快速评估）
    # median_corr = np.median(correlations)
    
    # 6. 返回完整相关系数列表 + 中位数（主需为correlations，中位数作为辅助）
    return correlations, median_corr

import numpy as np
import torch
from scipy.stats import spearmanr  # 确保导入spearmanr
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
import warnings


def calculate_spearman_correlation_evaluate(pred_matrix, raw_matrix):
    """
    计算所有基因的Spearman相关系数（NaN值替换为0），并返回完整系数列表与其中位数
    
    参数:
        pred_matrix: 预测的RNA矩阵（shape: [n_cells, n_genes]），支持torch.Tensor或numpy.ndarray
        raw_matrix: 真实的RNA矩阵（shape: [n_cells, n_genes]），支持torch.Tensor或numpy.ndarray
    
    返回:
        all_correlations: 所有基因的Spearman相关系数列表（NaN已替换为0，shape: [n_genes,]）
        median_corr: 相关系数的中位数（基于替换后的数据计算，反映整体相关性水平）
    """
    # 1. 处理输入格式：torch张量 → CPU上的numpy数组
    if isinstance(pred_matrix, torch.Tensor):
        pred_matrix = pred_matrix.cpu().numpy()
    if isinstance(raw_matrix, torch.Tensor):
        raw_matrix = raw_matrix.cpu().numpy()

    # 2. 校验输入有效性：若矩阵为空（无细胞），返回全0列表和0中位数
    if pred_matrix.shape[0] == 0 or raw_matrix.shape[0] == 0:
        warnings.warn("输入矩阵为空（无细胞），返回全0相关系数")
        # 若矩阵无基因，返回空列表；否则返回与基因数匹配的全0列表
        n_genes = max(pred_matrix.shape[1], raw_matrix.shape[1]) if (pred_matrix.ndim >=2 and raw_matrix.ndim >=2) else 0
        all_correlations = np.zeros(n_genes)
        return all_correlations, 0.0
    
    # 3. 校验矩阵形状：确保细胞数、基因数一致（避免按列计算时错位）
    if pred_matrix.shape != raw_matrix.shape:
        raise ValueError(f"预测矩阵与真实矩阵形状不匹配！pred: {pred_matrix.shape}, raw: {raw_matrix.shape}")

    # 4. 逐个基因（按列）计算Spearman相关系数
    all_correlations = []
    n_genes = pred_matrix.shape[1]  # 基因总数（矩阵列数）
    for i in range(n_genes):
        # 提取第i个基因的所有细胞表达量（预测值+真实值）
        pred_gene = pred_matrix[:, i]
        raw_gene = raw_matrix[:, i]
        
        # 计算Spearman相关系数（返回：(相关系数, p值)，取第0个元素）
        corr, _ = spearmanr(pred_gene, raw_gene)
        all_correlations.append(corr)
    
    # 5. 处理NaN值：将NaN替换为0（如基因表达量全相同导致无法计算相关系数的情况）
    all_correlations = np.array(all_correlations)
    median_corr = np.median(all_correlations[~np.isnan(all_correlations)])
    all_correlations[np.isnan(all_correlations)] = 0  # 核心修改：NaN → 0
    
    
    return all_correlations, median_corr



def calculate_mse_checked_evaluate(y_pred, y_true, target_sum=1e4, is_normalized=False):
    """
    计算预测矩阵与真实矩阵的MSE（支持预处理），返回每个基因的MSE列表和整体平均MSE
    
    参数:
        y_pred: 预测的RNA矩阵（shape: [n_cells, n_genes]），支持torch.Tensor或numpy.ndarray
        y_true: 真实的RNA矩阵（shape: [n_cells, n_genes]），支持torch.Tensor或numpy.ndarray
        target_sum: 归一化后的每行总和（仅当is_normalized=False时生效）
        is_normalized: 是否已对输入矩阵做过归一化+log1p预处理（True则跳过预处理）
    
    返回:
        per_gene_mse: 每个基因的MSE列表（shape: [n_genes,]，反映单个基因的预测误差）
        overall_mse: 整体MSE（所有细胞-基因对的平均误差，即原函数返回值，反映全局误差）
    """
    # 1. 处理输入格式：torch张量 → CPU上的numpy数组（detach避免计算图残留）
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # 2. 校验输入形状：确保预测矩阵与真实矩阵维度一致
    if y_pred.shape != y_true.shape:
        raise ValueError(f"预测矩阵与真实矩阵形状不匹配！y_pred: {y_pred.shape}, y_true: {y_true.shape}")
    
    # 3. 复制数组：避免修改原始输入数据
    y_pred_processed = y_pred.copy()
    y_true_processed = y_true.copy()

    # 4. 预处理（仅当未归一化时执行：L1归一化→缩放至target_sum→log1p转换，减少数值偏差）
    if not is_normalized:
        y_true_processed = np.log1p(y_true_processed)
        y_pred_processed = np.log1p(y_pred_processed)

    # 5. 计算每个基因的MSE（按列计算：每个基因的所有细胞误差）
    per_gene_mse = []
    n_genes = y_pred_processed.shape[1]
    for i in range(n_genes):
        # 提取第i个基因的预处理后表达量（预测值+真实值）
        pred_gene = y_pred_processed[:, i]
        true_gene = y_true_processed[:, i]
        # 计算单个基因的MSE
        gene_mse = mean_squared_error(true_gene, pred_gene)
        per_gene_mse.append(gene_mse)
    overall_mse = mean_squared_error(y_true_processed, y_pred_processed)
    per_gene_mse = np.array(per_gene_mse)
    
    return per_gene_mse,overall_mse


local_weights_path = './model.safetensors' 
from model import *

# ======================================================================
# --- 5. 主执行程序 ---
# ======================================================================
if __name__ == '__main__':
    # --- 超参数 ---
    DEVICE = "cuda:4" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 128 # Swin Transformer较大，如果显存不足可以适当减小
    LATENT_DIM = 128
    LEARNING_RATE = 1e-4
    EPOCHS = 100 #100 
    
    # *** MODIFIED: 损失函数权重，增加了 alignment_weight ***
    LOSS_WEIGHTS = {
        "image_weight": 1.0,      # 图像重建损失权重
        "rna_weight": 10.0,         # RNA重建损失权重 (通常NLL值较大，作为基准)
        "kld_weight": 0.5,         # VAE的正则化项，建议使用KL退火，初始beta可以小一些
        "alignment_weight": 0.5    # 新增的潜在空间对齐损失权重
    }

    print(f"Using device: {DEVICE}")
    
    method = 'STevs'
    # # datasets = [151507, 151508, 151509, 151510, 151669,151670, 151671, 151672, 151673, 151674, 151675, 151676,'a1', 'a2', 'p1','p2']
    # datasets = ['H1', 'H2', 'H3','rep1', 'rep2', 'rep3' ]
    datasets = ['E15', 'E18']

    for dataset in datasets:
        
        H5AD_PATH = f'/home/liuyinbo/qq/data/adata_slice{dataset}.h5ad'
        Himg_PATH = f'/home/liuyinbo/qq/data/adata_h_img_{dataset}.h5ad'
        
        adata_h5 = sc.read_h5ad(H5AD_PATH)
        adata_img = sc.read_h5ad(Himg_PATH)
        adata_rna = adata_h5[:, adata_h5.var['mode'] == 'RNA'].copy()

        # 1. 准备数据
        train_loader, val_loader, test_loader, img_shape, rna_dim, spatial_dim,train_indices,val_dataset,test_indices= prepare_adata_loader(
            adata_img=adata_img, adata_rna=adata_rna,
            layer_key_img='rawX', shape_key_img='IMG_Shape', 
            layer_key_rna='rawX', spatial_key='spatial',
            batch_size=BATCH_SIZE
        )
        
        C, H, W = img_shape

        # 2. 初始化模型
        model = MultiModalVAE(
            input_channels=C, spatial_dim=spatial_dim,
            latent_dim=LATENT_DIM, img_size=H, rna_dim=rna_dim
        ).float()

        # 3. 训练模型
        # 注意: KL退火逻辑已在训练函数内部实现
        trained_model = train_multi_modal_vae(
            model=model, train_loader=train_loader, val_loader=val_loader,
            epochs=EPOCHS, learning_rate=LEARNING_RATE, device=DEVICE, **LOSS_WEIGHTS
        )
        # 4. 在测试集上评估 (代码不变)
        print("\n--- 在测试集上评估RNA预测 ---")
        trained_model.eval()
        predicted_rna_list, true_rna_list = [], []
        with torch.no_grad():
            for (img_data, spatial_data, rna_data) in test_loader:
                img_data, spatial_data = img_data.to(DEVICE), spatial_data.to(DEVICE)
                _, recon_rna_params, _, _, _, _ = trained_model(img_data, spatial_data)
                predicted_rna_list.append(recon_rna_params['mu'].cpu().numpy())
                true_rna_list.append(rna_data.numpy())

        predicted_rna_matrix_inner = np.vstack(predicted_rna_list)
        true_rna_matrix_inner = np.vstack(true_rna_list)
        all_pearson,median_pearson = calculate_median_pearson_evaluate(predicted_rna_matrix_inner, true_rna_matrix_inner)
        all_spearman, median_spearman= calculate_spearman_correlation_evaluate(predicted_rna_matrix_inner, true_rna_matrix_inner)
        all_mse, median_mse = calculate_mse_checked_evaluate(predicted_rna_matrix_inner, true_rna_matrix_inner, is_normalized= True)

        print(f"最终测试集评估结果: \n"
                f"  - 中位 Pearson 相关性: {median_pearson:.4f}\n"
                f"  - 中位 Spearman 相关性: {median_spearman:.4f}")


        adata_inner = sc.AnnData(X=predicted_rna_matrix_inner)

        test_data = adata_rna[test_indices].copy()  
        adata_inner.var = test_data.var
        # 1. 添加真实RNA矩阵到layers
        adata_inner.layers['true_rna'] = true_rna_matrix_inner 
        adata_inner.var['Pearson'] = all_pearson
        adata_inner.var['Spearman'] = all_spearman
        adata_inner.var['MSE'] = all_mse

        adata_inner.obs = test_data.obs[['cell_type']].copy()  

        adata_inner.obsm['spatial'] = test_data.obsm['spatial'].copy()  


        adata_inner.uns['metrics'] = {
        'median_pearson': median_pearson,
        'median_spearman': median_spearman,
        'median_mse': median_mse
        }
        adata_inner.obs_names = test_data.obs_names
        # 保存AnnData对象
        
        save_path = f"../../save_data/{dataset}/{method}/inner/adata_slice_{dataset}.h5ad"
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        adata_inner.write(save_path)
        print(f"AnnData对象已成功保存至{save_path}")




    ========= cross slice=========
    datasets_groups = [
            [151507, 151508, 151509, 151510],  # 第一组
            [151669, 151670, 151671, 151672],  # 第二组
            [151673, 151674, 151675, 151676],   # 第三组
            ['a1', 'a2'],  # 第一组
            ['p1','p2'],  # 第二组
            ["151673_0.0", "151673_0.1", "151673_0.2", "151673_0.3", "151673_0.4", "151673_0.5","151673_0.4", "151673_0.5","151673_0.6", "151673_0.7","151673_0.8", "151673_0.9","151673_1.0"],
            ["151674_0.0", "151674_0.1", "151674_0.2", "151674_0.3", "151674_0.4", "151674_0.5","151674_0.4", "151674_0.5","151674_0.6", "151674_0.7","151674_0.8", "151674_0.9","151674_1.0"]
            ['E15', 'E18'] # # 第三组
        ]
    # # datasets_groups = [
    # #         # ['H1', 'H2', 'H3'],
    # #         # ['rep1', 'rep2', 'rep3']
    #         ["151673_0.0", "151673_0.1", "151673_0.2", "151673_0.3", "151673_0.4", "151673_0.5","151673_0.4", "151673_0.5","151673_0.6", "151673_0.7","151673_0.8", "151673_0.9","151673_1.0"],
    #         ["151674_0.0", "151674_0.1", "151674_0.2", "151674_0.3", "151674_0.4", "151674_0.5","151674_0.4", "151674_0.5","151674_0.6", "151674_0.7","151674_0.8", "151674_0.9","151674_1.0"]
    #         [151673, 151674, 151675, 151676],   # 第三组
    # #     ]
    
  
    h5ad_path_template = '/home/liuyinbo/qq/data/adata_slice{sample_id}.h5ad'  # 假设文件名是 adata_151507.h5ad‘
    h5ad_img_template = '/home/liuyinbo/qq/data/adata_h_img_{sample_id}.h5ad'
    for group_idx, group_samples in enumerate(datasets_groups):
        print(f"开始处理第 {group_idx + 1} 个数据集组：{group_samples}")
        # 内层循环：留一法交叉验证（每个样本轮流作为测试集，其余作为训练集）
        for train_sample in group_samples:
            # if train_sample != group_samples[0]:
            #     break
            H5AD_PAT_train = h5ad_path_template.format(sample_id=train_sample)
            H5AD_img_train = h5ad_img_template.format(sample_id=train_sample)
            adata_h5_train = sc.read_h5ad(H5AD_PAT_train)
            adata_img_train = sc.read_h5ad(H5AD_img_train)
            adata_rna = adata_h5_train[:, adata_h5_train.var['mode'] == 'RNA'].copy()
       
            train_loader, val_loader, test_loader, img_shape, rna_dim, spatial_dim,train_indices,val_dataset,test_indices = prepare_adata_loader(
                adata_img=adata_img_train, adata_rna=adata_rna,
                layer_key_img='rawX', shape_key_img='IMG_Shape', 
                layer_key_rna='rawX', spatial_key='spatial',
                batch_size=BATCH_SIZE,
                ratio=(0.9, 0.1, 0.0)
            )
            C, H, W = img_shape
     
            model = MultiModalVAE(
                input_channels=C, spatial_dim=spatial_dim,
                latent_dim=LATENT_DIM, img_size=H, rna_dim=rna_dim
            ).float()
            trained_model = train_multi_modal_vae(
                model=model, train_loader=train_loader, val_loader=val_loader,
                epochs=EPOCHS, learning_rate=LEARNING_RATE, device=DEVICE, **LOSS_WEIGHTS
            )
            for test_sample in group_samples:
                if test_sample != train_sample:
                    print(f"当前配对：训练集={train_sample} 测试集={test_sample}")

                    H5AD_PAT_test = h5ad_path_template.format(sample_id=test_sample)
                    H5AD_img_test = h5ad_img_template.format(sample_id=test_sample)
                    
                    adata_h5_test = sc.read_h5ad(H5AD_PAT_test)
                    adata_img_test = sc.read_h5ad(H5AD_img_test)
                    

                    adata_rna_other = adata_h5_test[:, adata_h5_test.var['mode'] == 'RNA'].copy()

                    
                    other_data_loader, _, _, _, _, _,train_index, _, _ = prepare_adata_loader(
                        adata_img=adata_img_test, adata_rna=adata_rna_other,
                        layer_key_img='rawX', shape_key_img='IMG_Shape', 
                        layer_key_rna='rawX', spatial_key='spatial',
                        batch_size=BATCH_SIZE, ratio=(1.0, 0.0, 0.0),Train_shuffle=False
                    )
                    adata_rna_other_index = adata_rna_other[train_index]
                    predicted_rna_list, true_rna_list = [], []
                    with torch.no_grad():
                        for (img_data, spatial_data, rna_data) in other_data_loader:
                            img_data, spatial_data = img_data.to(DEVICE), spatial_data.to(DEVICE)
                            _, recon_rna_params, _, _, _, _  = trained_model(img_data, spatial_data)
                            
                            predicted_rna_list.append(recon_rna_params['mu'].cpu().numpy())
                            true_rna_list.append(rna_data.numpy())
                            
                    predicted_rna_matrix_cross = np.vstack(predicted_rna_list)
                    true_rna_matrix_cross = np.vstack(true_rna_list)

    
                    all_pearson,median_pearson = calculate_median_pearson_evaluate(predicted_rna_matrix_cross, true_rna_matrix_cross)
                    all_spearman, median_spearman= calculate_spearman_correlation_evaluate(predicted_rna_matrix_cross, true_rna_matrix_cross)
                    all_mse, median_mse = calculate_mse_checked_evaluate(predicted_rna_matrix_cross, true_rna_matrix_cross, is_normalized= True)

                    print(f"最终测试集评估结果: \n"
                            f"  - 中位 Pearson 相关性: {median_pearson:.4f}\n"
                            f"  - 中位 Spearman 相关性: {median_spearman:.4f}")
                    
                    

                    adata_cross = sc.AnnData(X=predicted_rna_matrix_cross)


                    test_data = adata_rna_other_index.copy()  
                    adata_cross.var = test_data.var
                    # 1. 添加真实RNA矩阵到layers
                    adata_cross.layers['true_rna'] = true_rna_matrix_cross 
                    adata_cross.var['Pearson'] = all_pearson
                    adata_cross.var['Spearman'] = all_spearman
                    adata_cross.var['MSE'] = all_mse

                    adata_cross.obs = test_data.obs[['cell_type']].copy()  

                    adata_cross.obsm['spatial'] = test_data.obsm['spatial'].copy()  

       
                    adata_cross.uns['metrics'] = {
                    'median_pearson': median_pearson,
                    'median_spearman': median_spearman,
                    'median_mse': median_mse
                    }
                    adata_cross.obs_names = test_data.obs_names
                    
                    save_path = f"../../save_data/{train_sample}/{method}/cross/adata_slice_{test_sample}.h5ad"
                    
                    # 6. 自动创建路径（不存在则创建）
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # 7. 保存 AnnData
                    adata_cross.write(save_path)
                    print(f"AnnData 已保存至：{save_path}")
                    
                    # -------------------------- 3.9 清理内存（避免多轮实验内存溢出） --------------------------
               
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  # 清空GPU缓存
                    print(f"--- 实验结束（测试集={test_sample}），内存已清理 ---")




