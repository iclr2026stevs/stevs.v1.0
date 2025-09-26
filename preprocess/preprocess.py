from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import h5py
import numpy as np
import scanpy as sc
import os
import scipy.sparse
import pandas as pd

import anndata
import h5py
import numpy as np
import scipy as sp
import scanpy as sc
import pylab as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from anndata import AnnData
from scipy import sparse
from scipy.sparse import issparse
from scipy.stats import stats
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scanpy as sc
import torch
import os
from sklearn.preprocessing import OneHotEncoder
import subprocess
import re
import os
import setproctitle
from statsmodels.stats.multitest import multipletests
from umap import UMAP
import math
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from VAE_utils import merge_and_rename_peaks_in_adata_list, concat_adata,concat_adata_img, clean_metadata_dataframe,split_rna_atac,create_img_adata_from_data
import pandas as pd
import numpy as np
import scanpy as sc
from typing import List, Dict, Optional

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.system('export HDF5_USE_FILE_LOCKING=FALSE')
print(os.environ["HDF5_USE_FILE_LOCKING"])


class Config:
    def __init__(self, spatial_dim=20):
        self.loc_range = 20
        self.center_slice = 0
        self.spatial_dim = spatial_dim

class AnnDataProcessor:
    def __init__(self,ModelTask=None,config=Config(), **kwargs):
        self.ModelTask = ModelTask
        self._set_config(config)
        if kwargs:
            self.identifier = None
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            #
            try:
                self.handle_slices_data(**kwargs)
            except:
                try:
                    self.handle_slices_list_data(**kwargs)
                except:
                    raise ValueError('参数错误...')

    def _set_config(self, config):
        self.loc_range = config.loc_range
        self.view_idx = None
        self.center_slice = config.center_slice
        self.MinMaxScaler_list = []
        self.time_dict = {}
        self.n_celltype = None
        self.unique_celltype = None
        self.h5data_list = []
        self.config = config

    def handle_slices_data(self, loc_init, x_init, slice_values, celltype_init, obs_names=None, var_names=None,
                           center_slice=0, changed_slice_set=None, batch_tuple=None, select_genes=None, loc_range=20,
                           filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True,use_scale=False):
        """
        Initialize the AnnDataProcessor with the provided data.

        Parameters:
        - loc_init: Array of initial locations.
        - x_init: Array of feature data.
        - slice_values: Array of different slice markers.
        - celltype_init: Ncounts matrix (label data cell_type).
        - changed_slice_set: Set of slices to process.
        - batch_tuple: Tuple of batch markers, defaults to None.
        - rotation: Array of rotation transformations, defaults to None.
        - translation: Array of translation transformations, defaults to None.
        - loc_range: Range for location normalization, defaults to 20.
        """


        self.loc_init = loc_init
        self.x_init = x_init
        self.slice_values = slice_values
        self.celltype_init = celltype_init
        self.changed_slice_set = changed_slice_set or tuple(range(slice_values.shape[-1]))
        self.batch_tuple = batch_tuple if batch_tuple is not None else self.changed_slice_set
        self.select_genes = select_genes
        self.loc_range = loc_range
        self.view_idx = None
        self.center_slice = center_slice
        self.MinMaxScaler_list = []
        self.time_dict = {}
        self.n_celltype = None
        self.unique_celltype = None
        self.h5data_list = []


        self.obs_names = obs_names
        self.var_names = var_names
        self.adata_list = self.preprocess_data(filter_min_counts=filter_min_counts,
                          size_factors=size_factors,
                          normalize_input=normalize_input,
                          logtrans_input=logtrans_input,use_scale=use_scale)


    def handle_slices_list_data(self,adata_list = None,
                                center_slice=0,
                                changed_slice_set=None,
                                batch_tuple=None,
                                select_genes=None,
                                loc_range=20,
                                spatial_name='spatial',
                                celltype_name='cell_type',**kwargs):
        loc = [np.array(slice.obsm[spatial_name]) for slice in adata_list]
        celltype_init = [np.array(slice.obs[celltype_name]) for slice in adata_list]
        x_init = []
        for slice in adata_list:
            if scipy.sparse.issparse(slice.X):  # 如果是稀疏矩阵
                x_init.append(slice.X.toarray())  # 转换为密集矩阵
            else:
                x_init.append(np.array(slice.X))  # 直接转换为NumPy数组
        slice_number = np.concatenate([np.zeros(x.shape[0]) + i for i, x in enumerate(x_init)])

        # OneHot编码
        encoder = OneHotEncoder(sparse_output=False)
        slice_values = encoder.fit_transform(np.array(slice_number).reshape(-1, 1))

        # 获取所有列名的交集
        all_columns = set(adata_list[0].var_names)  # 初始化为第一个slice的列名
        for slice in adata_list[1:]:
            all_columns &= set(slice.var_names)  # 对所有slice进行交集操作
        all_columns = sorted(list(all_columns))  # 生成列名交集，并排序

        # 创建统一的 X DataFrame
        x_full = pd.DataFrame(columns=all_columns)
        for i, slice in enumerate(adata_list):
            # 如果slice.X是numpy数组，则赋予行名和列名
            x = pd.DataFrame(x_init[i], index=slice.obs_names, columns=slice.var_names)
            x_full = pd.concat([x_full, x.loc[:, all_columns]])  # 拼接到统一的 DataFrame

        loc = np.vstack(loc)
        celltype_init = np.concatenate(celltype_init)
        self.handle_slices_data(loc_init=loc,
                                x_init=x_full,
                                slice_values=slice_values,
                                celltype_init=celltype_init,
                                center_slice=center_slice,
                                changed_slice_set=changed_slice_set,
                                batch_tuple=batch_tuple,
                                select_genes=select_genes,
                                loc_range=loc_range,**kwargs)

    def preprocess_data(self, mode_class=['RNA', ], filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True, use_scale=False):
        """Preprocess the data to create AnnData objects."""
        adata_list_temp = []
        y_layer_list = []
        y_labels = []
        X_adata = []
        if not isinstance(self.x_init, pd.DataFrame):
            row_names = [f'cell_{i}' for i in range(self.x_init.shape[0])] if self.obs_names is None else list(
                self.obs_names)
            col_names = [f'gene_{i}' for i in range(self.x_init.shape[1])] if self.var_names is None else list(
                self.var_names)
            self.x_init = pd.DataFrame(self.x_init, index=row_names, columns=col_names)
        NA_idx = np.where((self.celltype_init == b'NA') | (self.celltype_init == 'NA'))[0]
        self.loc_init = np.delete(self.loc_init, NA_idx, axis=0)
        self.x_init = self.x_init.drop(self.x_init.index[NA_idx])  # Drop rows with 'NA'
        self.slice_values = np.delete(self.slice_values, NA_idx, axis=0)
        self.celltype_init = np.delete(self.celltype_init, NA_idx, axis=0)

        if self.select_genes is not None and self.select_genes>0:
            importantGenes = geneSelection(self.x_init, n=self.select_genes, plot=False)
            self.x_init = self.x_init[:, importantGenes]
            np.savetxt("selected_genes.txt", importantGenes, delimiter=",", fmt="%i")

        decoded_y = np.array([item.decode('utf-8') if isinstance(item, bytes) else item for item in self.celltype_init])

        batch_dict = {i: j for j, i in enumerate(set(self.batch_tuple))}
        self.n_batch = len(set(self.batch_tuple))
        center_slice_list = [True if self.center_slice is not None and i == self.center_slice else False for i, _ in
                             enumerate(self.changed_slice_set)]

        view_idx = self.view_idx = [np.where(self.slice_values[:, ii] == 1)[0] for ii in self.changed_slice_set]
        n_samples_list = [len(x) for x in view_idx]
        bv = [np.zeros(i) + batch_dict[self.batch_tuple[j]] for j, i in enumerate(n_samples_list)]
        self.sv = [np.zeros(i) + j for j, i in enumerate(n_samples_list)]

        # Normalize location data
        # Create AnnData object
        adata = sc.AnnData(self.x_init.values, dtype="float64")
        adata.var_names = self.x_init.columns  # Set gene names
        adata.obs_names = self.x_init.index  # Set cell names
        adata = normalize(adata,
                          filter_min_counts=filter_min_counts,
                          size_factors=size_factors,
                          normalize_input=normalize_input,
                          logtrans_input=logtrans_input)

        for i in range(len(view_idx)):
            loc = self.loc_init[view_idx[i], :]
            x = adata.X[view_idx[i], :]
            raw_x = adata.layers['rawX'][view_idx[i], :]
            normalized_x = adata.obsm['normalized_counts'][view_idx[i], :]
            size_factors = adata.obs.size_factors[view_idx[i]]
            y = decoded_y[view_idx[i]]
            layers_dict = {}
            # Create layer dictionary
            for l in set(decoded_y):
                layers_dict[l] = np.where(y == l)[0]

            y_layer_list.append(layers_dict)
            y_labels.append(generate_labels(layers_dict))
            X_adata.append(raw_x)
            # Normalize location information
            scaler = MinMaxScaler()
            if use_scale == True:
                scaler_loc = scaler.fit_transform(loc) * self.loc_range
            else:
                scaler_loc = loc-loc.mean(0)
            self.MinMaxScaler_list.append(scaler)
            adata_slice = sc.AnnData(raw_x, dtype="float64")
            adata_slice.obs_names = adata.obs_names[view_idx[i]]
            adata_slice.var_names = adata.var_names
            adata_slice.raw = copy.deepcopy(adata_slice)
            adata_slice.layers['rawX'] = adata_slice.X.copy()
            adata_slice.X = x
            adata_slice.obs['size_factors'] = size_factors.reset_index(drop=True).values
            adata_slice.obs['layers_label'] = generate_labels(layers_dict).flatten().astype(str)
            adata_slice.obs['cell_type'] = adata_slice.obs['layers_label'].astype(str)
            adata_slice.obs['batch_view'] = bv[i]
            adata_slice.obs['slice_view'] = self.sv[i]
            adata_slice.obs['cell_index'] = np.arange(self.sv[i].shape[0])

            mask_data = pd.DataFrame(np.ones((self.sv[i].shape[0],
                                                             mode_class.__len__())),
                                                            index=adata_slice.obs.index.map(str),
                                                            columns=mode_class)
            adata_slice.obsm['mask'] = mask_data
            adata_slice.obsm['location'] = loc
            adata_slice.obsm['spatial'] = scaler_loc
            adata_slice.obsm['normalized_counts'] = normalized_x

            adata_slice.uns['layers_dict'] = layers_dict
            adata_slice.uns['slice_id'] = self.changed_slice_set[i]
            adata_slice.uns['mode'] = ['RNA']
            adata_slice.uns['center_slice'] = center_slice_list[i]
            adata_slice.var['mode'] = mode_class[0]
            adata_list_temp.append(adata_slice)
            self.h5data_list.append(
                [raw_x, loc, adata_slice.obs['cell_type'], adata_slice.uns['slice_id'], adata_slice.obs_names,
                 adata_slice.var_names])

        cell_type_values = np.concatenate([adata.obs['cell_type'].values for adata in adata_list_temp])
        self.unique_celltype = np.unique(cell_type_values)  # 获取唯一的 cell_type 值
        self.n_celltype = np.unique(cell_type_values).__len__()  # 获取唯一的 cell_type 值的大小
        return adata_list_temp

    def normalize_adata_list(self,
                             adata_list=None,
                             size_factors=True,
                             normalize_input=True,
                             logtrans_input=True,
                             spatial='spatial',
                             use_spatial_Nd=False,
                             center_to_origin = False,
                             spatial_loc=False,
                             scaler_spatial_global=False,
                             range_aixs_change = True,
                             assert_norm=False,
                             loc_range=20,
                             min_genes=10,
                             min_counts=10,
                             n_top_genes=3000,
                             spatial_dim=None,
                             mode='RNA'):
        if mode == 'IMG':
            min_genes = None
            min_counts = None
            n_top_genes = None
        if mode == 'ATAC':
            adata_list = merge_and_rename_peaks_in_adata_list(adata_list)
        if spatial_dim is None:
            spatial_dim = self.config.spatial_dim
        if adata_list is None and hasattr(self, 'adata_list'):
            adata_list = self.adata_list
        if not hasattr(self, 'loc_range'):
            self.loc_range = 20
        if not loc_range is None:
            self.loc_range = loc_range
        for i, adata in enumerate(adata_list):
            if not min_genes is None:
                sc.pp.filter_cells(adata, min_genes=min_genes)
            if not min_counts is None:
                sc.pp.filter_genes(adata, min_counts=min_counts)
            if (not n_top_genes is None) and (adata.n_vars > 1.1 * n_top_genes):
                print(n_top_genes)
                print((adata.n_vars > 1.1 * n_top_genes))
                print((not n_top_genes is None) and (adata.n_vars > 1.1 * n_top_genes))
                adata_for_hvg = adata.copy()
                print('highly_variable...')
                sc.pp.normalize_total(adata_for_hvg, target_sum=1e4)
                sc.pp.log1p(adata_for_hvg)
                sc.pp.highly_variable_genes(adata_for_hvg, n_top_genes=n_top_genes, flavor='seurat_v3')
                adata = adata[:, adata_for_hvg.var['highly_variable']]
                adata_list[i] = adata
            print('adata:', adata.shape)
            if center_to_origin == True:
                adata.obsm[spatial][:,:spatial_dim] = adata.obsm[spatial][:,:spatial_dim] - adata.obsm[spatial][:,:spatial_dim].mean(0)
        if mode=="IMG":
            combined_adata = concat_adata_img(adata_list)
        else:
            combined_adata = concat_adata(adata_list)

        combined_adata.obsm['location'] = combined_adata.obsm[spatial].copy()
        if (use_spatial_Nd or scaler_spatial_global==True) and range_aixs_change == True:
            combined_adata.obsm['spatial'] = (combined_adata.obsm[spatial] - combined_adata.obsm[spatial].min(0))/(combined_adata.obsm[spatial].max(0) - combined_adata.obsm[spatial].min(0)) * loc_range
        # 调用标准化函数（假设 normalize 是一个已定义的函数）
        if (use_spatial_Nd or scaler_spatial_global==True) and range_aixs_change == False:
            # 假设 combined_adata.obsm['spatial'] 已经加载
            coords = combined_adata.obsm[spatial]
            coords = coords - coords.mean(0)
            # 计算每个维度的最小值和最大值
            min_vals = coords.min(axis=0)  # 每个维度的最小值 [min_x, min_y, min_z]
            max_vals = coords.max(axis=0)  # 每个维度的最大值 [max_x, max_y, max_z]
            # 计算全局缩放因子（使用最大范围以保持比例）
            global_scale = (max_vals - min_vals).max()
            # 平移并缩放所有坐标到 [0, 1] 区间
            scaled_coords = (coords - min_vals) / global_scale  * loc_range
            # 将缩放后的数据保存回原始位置
            combined_adata.obsm['spatial'] = scaled_coords
        if mode == 'RNA':
            normalized_adata = normalize(combined_adata,
                                         filter_min_counts=False,
                                         size_factors=size_factors,
                                         normalize_input=normalize_input,
                                         logtrans_input=logtrans_input,
                                         assert_norm=assert_norm)
        elif mode == 'embedding':
            X = combined_adata.X if isinstance(combined_adata.X,
                                               np.ndarray) else combined_adata.X.toarray()  # 如果是稀疏矩阵，转换为密集矩阵
            # 对每行进行归一化
            X_normalized = (X - X.min(axis=1, keepdims=True)) / (
                        X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True))
            # 如果需要，将归一化后的数据重新赋值给 normalized_adata.X
            # 注意：如果 normalized_adata.X 是稀疏矩阵，你可能需要将其转换回稀疏格式
            normalized_adata = combined_adata
            normalized_adata.X = X_normalized
        elif mode == 'ATAC':
            normalized_adata = combined_adata

            # 确保X是一个NumPy数组，以防万一
            X = normalized_adata.X.toarray() if scipy.sparse.issparse(normalized_adata.X) else normalized_adata.X

            # 使用 numpy 的函数进行操作
            X = np.maximum(X, 0)  # 确保所有值非负
            X = np.minimum(X, 1)  # 确保所有值不超过1

            # 对数据进行二值化（所有正数变为1.0，0和负数变为0.0）
            X = (X > 0).astype('float32')

            # 将处理后的数据赋值回去
            normalized_adata.X = X

        elif mode == 'IMG':
            normalized_adata = combined_adata
            X = combined_adata.X

            # 使用原地操作，避免创建临时数组
            if X.dtype != np.float32:
                X = X.astype(np.float32, copy=False)  # 如果可能，避免复制
            else:
                X = np.asarray(X)  # 确保是numpy数组

            # 分步原地操作
            np.divide(X, 127.5, out=X)  # 使用out参数进行原地除法
            np.subtract(X, 1.0, out=X)  # 原地减法

            normalized_adata.X = X

        normalized_adata_list = []
        end = 0
        for i, adata in enumerate(adata_list):
            start = end
            end = start + adata.shape[0]
            normalized_adata_temp = normalized_adata[start:end]
            subset_data = adata[:, combined_adata.var_names]
            for n, v in adata.uns.items():
                normalized_adata_temp.uns[n] = v
            normalized_adata_temp.layers['rawX'] = subset_data.X.toarray() if scipy.sparse.issparse(subset_data.X) else subset_data.X
            normalized_adata_temp.var['mode'] = mode
            normalized_adata_temp.uns['mode'] = [mode]
            normalized_adata_temp.obs['batch_view'] = i
            normalized_adata_temp.obs['slice_view'] = i
            normalized_adata_temp.obs['cell_index'] = np.arange(normalized_adata_temp.shape[0])
            if 'cell_type' not in normalized_adata_temp.obs.columns:
                normalized_adata_temp.obs['cell_type'] = '0'
            normalized_adata_temp.obs['layers_label'] = normalized_adata_temp.obs['cell_type']
            mask_data = pd.DataFrame(np.ones((normalized_adata_temp.shape[0],
                                              1)),
                                     index=normalized_adata_temp.obs.index.map(str),
                                     columns=[mode])
            normalized_adata_temp.obsm['mask'] = mask_data
            if use_spatial_Nd == False and scaler_spatial_global==False:
                if spatial_loc == True and range_aixs_change==False:
                    normalized_adata_temp.obsm['spatial'] = normalized_adata_temp.obsm[spatial] - \
                                                            normalized_adata_temp.obsm[
                                                                spatial].mean(0)
                    normalized_adata_temp.obsm['spatial'] = (normalized_adata_temp.obsm[spatial] -
                                                             normalized_adata_temp.obsm[spatial].min(0)) / (
                                                                    normalized_adata_temp.obsm[spatial].max(0) -
                                                                    normalized_adata_temp.obsm[spatial].min(
                                                                        0)) * loc_range
                elif spatial_loc == True and range_aixs_change == False:
                    if (use_spatial_Nd or scaler_spatial_global == True) and range_aixs_change == False:
                        # 假设 combined_adata.obsm['spatial'] 已经加载
                        coords = normalized_adata_temp.obsm[spatial]  # shape: (M, 3)

                        # 计算每个维度的最小值和最大值
                        min_vals = coords.min(axis=0)  # 每个维度的最小值 [min_x, min_y, min_z]
                        max_vals = coords.max(axis=0)  # 每个维度的最大值 [max_x, max_y, max_z]

                        # 计算全局缩放因子（使用最大范围以保持比例）
                        global_scale = (max_vals - min_vals).max()

                        # 平移并缩放所有坐标到 [0, 1] 区间
                        scaled_coords = (coords - min_vals) / global_scale

                        # 将缩放后的数据保存回原始位置
                        normalized_adata_temp.obsm['spatial'] = scaled_coords

            normalized_adata_temp.uns['slice_id'] = i
            normalized_adata_temp.X = normalized_adata_temp.X.toarray() if scipy.sparse.issparse(
                normalized_adata_temp.X) else normalized_adata_temp.X
            normalized_adata_temp.obs_names = subset_data.obs_names
            normalized_adata_list.append(normalized_adata_temp)
        return normalized_adata_list


    def normalize_spatial(self,
                          adata_list,
                          spatial='spatial',
                          out_spatial=None,
                          loc_range=20,
                          center_to_origin=False,
                          scaler_spatial_global=False,
                          use_spatial_Nd=False,
                          range_aixs_change=True,
                          spatial_loc=False,
                          spatial_dim=None
                          ):
        """
        专门处理adata_list中空间坐标的归一化、缩放和居中。
        该函数通过仅拼接空间坐标数组来优化性能，避免了拼接整个adata对象。

        参数:
            adata_list (List[AnnData]): AnnData对象列表。
            ... (其他空间相关的参数与原函数相同) ...

        返回:
            List[AnnData]: 空间坐标被更新后的AnnData对象列表。
        """
        # 创建一个副本以避免修改原始传入的列表
        if out_spatial is None:
            out_spatial = 'spatial'
        if spatial_dim is None:
            spatial_dim = self.spatial_dim
        # --- 1. 对每个adata独立进行的操作 (预处理) ---
        for adata in adata_list:
            if center_to_origin:
                coords = adata.obsm[spatial][:, :spatial_dim]
                adata.obsm[spatial][:, :spatial_dim] = coords - coords.mean(0)

        if scaler_spatial_global or use_spatial_Nd:
            print("执行全局空间坐标缩放...")
            # 记录每个adata的原始长度，以便后续拆分
            original_lengths = [adata.shape[0] for adata in adata_list]
            # **核心优化：只提取和拼接空间坐标**
            all_coords = np.vstack([adata.obsm[spatial] for adata in adata_list])
            # 应用全局归一化/缩放逻辑
            if range_aixs_change:
                # 按各轴独立缩放
                min_vals = all_coords.min(0)
                max_vals = all_coords.max(0)
                scaled_coords = (all_coords - min_vals) / (max_vals - min_vals + 1e-8) * loc_range
            else:
                # 保持坐标轴比例进行缩放
                coords_centered = all_coords - all_coords.mean(0)
                min_vals = coords_centered.min(axis=0)
                max_vals = coords_centered.max(axis=0)
                global_scale = (max_vals - min_vals).max()
                scaled_coords = (coords_centered - min_vals) / (global_scale + 1e-8) * loc_range
            # 将缩放后的坐标拆分并更新回每个adata对象
            split_indices = np.cumsum(original_lengths)[:-1]
            scaled_coords_list = np.split(scaled_coords, split_indices)

            for i, adata in enumerate(adata_list):
                adata.obsm[out_spatial] = scaled_coords_list[i]

        # --- 3. 局部(per-slice)缩放逻辑 ---
        elif spatial_loc:
            print("执行局部（per-slice）空间坐标缩放...")
            for adata in adata_list:
                coords = adata.obsm[spatial]
                coords_centered = coords - coords.mean(0)  # 局部居中
                if range_aixs_change:
                    # 按各轴独立缩放
                    min_vals = coords_centered.min(0)
                    max_vals = coords_centered.max(0)
                    adata.obsm[out_spatial] = (coords_centered - min_vals) / (max_vals - min_vals + 1e-8) * loc_range
                else:
                    # 保持坐标轴比例进行缩放
                    min_vals = coords_centered.min(axis=0)
                    max_vals = coords_centered.max(axis=0)
                    local_scale = (max_vals - min_vals).max()
                    adata.obsm[out_spatial] = (coords_centered - min_vals) / (local_scale + 1e-8) * loc_range
        return adata_list

    def reset_z_range_same_interval(self,
                 adata_list=None,
                 spatial='spatial',
                 min_range=1.0,
                 z_dim=-1):
        if adata_list is None and hasattr(self, 'adata_list'):
            adata_list = self.adata_list
        for i, adata in enumerate(adata_list):
            adata.obsm[spatial][:,z_dim] = i * min_range + 1e-8
        return adata_list

    def reset_z_range(self,
                      adata_list=None,
                      spatial='spatial',
                      min_range=1.0,
                      z_dim=-1):
        """
        按照z维度的平均值对adata进行排序，并根据原始z值的差值比例重新设置z值
        """
        if adata_list is None and hasattr(self, 'adata_list'):
            adata_list = self.adata_list

        # 计算每个adata的z_dim平均值
        z_values = []
        for adata in adata_list:
            z_mean = adata.obsm[spatial][:, z_dim].mean()
            z_values.append(z_mean)

        # 按z值从小到大排序
        sorted_indices = np.argsort(z_values)
        sorted_adata_list = [adata_list[i] for i in sorted_indices]
        sorted_z_values = [z_values[i] for i in sorted_indices]

        # 计算相邻z值之间的差值
        z_diffs = np.diff(sorted_z_values)

        # 如果所有z值相同，设置统一间隔
        if np.allclose(z_diffs, 0):
            z_diffs = np.ones_like(z_diffs)

        # 找出最小差值作为基准
        min_diff = np.min(z_diffs)

        # 计算缩放因子，使最小差值为min_range
        scale_factor = min_range / min_diff
        self.scale_factor = scale_factor
        # 根据原始差值比例计算新的z值
        new_z_values = [0.0]  # 第一个z值为0
        for diff in z_diffs:
            new_z = new_z_values[-1] + diff * scale_factor
            new_z_values.append(new_z)

        # 为每个adata设置新的z值，并添加小偏移量
        for i, adata in enumerate(sorted_adata_list):
            adata.obsm[spatial][:, z_dim] = new_z_values[i] + 1e-8
        if hasattr(self.config, 'initial_inducing_latent_layer_GP_points'):
            self.config.initial_inducing_latent_layer_GP_points[:,:-1] = self.config.initial_inducing_latent_layer_GP_points[:,:-1] / self.config.initial_inducing_latent_layer_GP_points[:,:-1].max() * (max(new_z_values) - min(new_z_values))
        return sorted_adata_list

    def set_center_slice(self, center_slice=0):
        if isinstance(center_slice, int):
            # 如果是整数，则将其转换为列表
            center_slice = [center_slice]
        self.center_slice = center_slice


    def simulate_stitching(self, adata, axis=0, from_low=True, threshold=0.5):
        cadata = adata.copy()
        coo = cadata.obsm['spatial']
        scale = np.max(coo[:, axis]) - np.min(coo[:, axis])
        if from_low:
            chosen_indices = coo[:, axis] > (scale * threshold + np.min(coo[:, axis]))
        else:
            chosen_indices = coo[:, axis] < (np.max(coo[:, axis]) - scale * threshold)
        cadata = cadata[chosen_indices, :]
        return cadata


    def add_data_to_slice(self, slice_index, additional_data, additional_data_name):
        """
        Add additional data to a specific slice.

        Parameters:
        - slice_index: Index of the slice to which data will be added.
        - additional_data: Data to be added (np.ndarray).
        - additional_data_name: Name under which to store the additional data.
        """
        if slice_index < len(self.adata_list):
            self.adata_list[slice_index].uns[additional_data_name] = additional_data
        else:
            raise IndexError("Slice index out of range.")


    def add_data_to_adata_list(self, slice_index_list, additional_data_list, additional_data_name):
        """
        Add additional data to multiple slices.

        Parameters:
        - slice_index_list: List of indices of slices to which data will be added.
        - additional_data_list: List of data to be added, corresponding to each slice index.
        - additional_data_name: Name under which to store the additional data.
        """
        if len(slice_index_list) != len(additional_data_list):
            raise ValueError("Length of slice_index_list must match length of additional_data_list.")

        for index, additional_data in zip(slice_index_list, additional_data_list):
            self.add_data_to_slice(index, additional_data, additional_data_name)

    def add_img_to_slice(self, img=None, img_coordinates=None, adata_list=None, xycoords=None, slice_index=0, simg_shape=(3, 32,32)):
        if adata_list is None:
            adata_list = self.adata_list
        if slice_index < len(adata_list):
            adata_list[slice_index].obsm['mask']['IMG'] = np.ones(adata_list[slice_index].shape[0])
            adata_list[slice_index].uns['mode'].append('IMG')
            if xycoords is not None:
                adata_list[slice_index].obsm['mask']['IMG'] = 1
                img_coordinates_result = []
                for i, coord in enumerate(adata_list[slice_index].obsm['location']):
                    index = np.where((xycoords == coord).all(axis=1))[0]  # 根据(x, y)找到行
                    if len(index) > 0:
                        img_coordinates_result.append(img_coordinates[index[0]])
                    else:
                        img_coordinates_result.append(np.array([0, 0]))
                        adata_list[slice_index].obsm['mask']['IMG'].iloc[i] = 0
                img_coordinates = np.array(img_coordinates_result)
                adata_list[slice_index].obsm['img_coordinates'] = img_coordinates
                adata_list[slice_index].uns['simg'] = extract_subimages(img,
                                                                             img_coordinates,
                                                                               h=simg_shape[-1])
            else:
                adata_list[slice_index].obsm['mask']['IMG'] = 0
                adata_list[slice_index].uns['simg'] = np.zeros(shape=(tuple([adata_list[slice_index].obsm['mask'].shape[0]] + list(simg_shape))))
        else:
            raise IndexError("Slice index out of range.")

    def add_imgs_to_adata_list(self,adata_list=None, imgs_list=None, img_coordinates_list=None, xycoords_list=None, slice_index_list=None, simg_shape=(3, 32,32)):
        if adata_list is None:
            adata_list = self.adata_list
        if slice_index_list is None:
            slice_index_list = range(adata_list.__len__())
        if len(slice_index_list) != len(imgs_list):
            raise ValueError("Length of slice_index_list must match length of additional_data_list.")
        for i,k in enumerate(slice_index_list):
            if xycoords_list is not None:
                self.add_img_to_slice(imgs_list[i], img_coordinates_list[i], adata_list=adata_list, xycoords=xycoords_list[i],slice_index=k, simg_shape=simg_shape)
            else:
                self.add_img_to_slice(imgs_list[i], img_coordinates_list[i],adata_list=adata_list, slice_index=k, simg_shape=simg_shape)
        return adata_list



    def create_img_adata_from_data(self,
                                   full_img: np.ndarray,
                                   barcodes: np.ndarray,
                                   img_coordinates: np.ndarray,
                                   spatial_coords: np.ndarray = None,
                                   patch_size: int = 32
                                   ):
        return create_img_adata_from_data(
                full_img,
                barcodes,
                img_coordinates,
                spatial_coords,
                patch_size,
        )

    def create_img_adata_list_from_data(self,
                                        full_img_list: List[np.ndarray],
                                        barcodes_list: List[np.ndarray],
                                        img_coordinates_list: List[np.ndarray],
                                        spatial_coords_list: List[np.ndarray] = None,
                                        patch_size: int = 32
                                        ):
        adata_list = []
        for i, (full_img,
                barcodes,
                img_coordinates) in enumerate(zip(full_img_list,
                                                  barcodes_list,
                                                  img_coordinates_list)):
            if spatial_coords_list is None:
                spatial_coords = None
            else:
                spatial_coords = spatial_coords_list[i]
            adata = self.create_img_adata_from_data(
                full_img,
                barcodes,
                img_coordinates,
                spatial_coords,
                patch_size,
                )
            adata.var_names = adata.var_names.to_numpy().astype(str)
            adata.obs_names = adata.obs_names.astype(str)
            adata_list.append(adata)
        return adata_list


    def add_time_to_adata_list(self, time_list, adata_list=None,spatial='spatial', time_range=10):
        if adata_list is None:
            adata_list = self.adata_list
        for i, time in enumerate(time_list):
            self.time_dict[i] = time
            adata_list[i].obs['time'] = (time - min(time_list)) / (max(time_list)-min(time_list)) * time_range
            adata_list[i].obsm['spatial_time'] = np.concatenate([adata_list[i].obsm[spatial], adata_list[i].obs['time'].values.reshape((-1,1))], axis=1)
        return adata_list

    def add_batch_to_adata_list(self, adata_list, mean_list=None, std_list=None, batch_factor=None, effect='*',
                                seed=42):
        if mean_list is None:
            mean_list = [1 for _ in range(len(adata_list))]
        if std_list is None:
            std_list = [0.1 for _ in range(len(adata_list))]
        adata_batch_list = []
        for adata, mean, std in zip(adata_list, mean_list, std_list):
            if (mean is None or std is None) and batch_factor is None:
                adata.var['batch_factor'] = 1
                adata_batch_list.append(adata)
            else:
                adata_batch_list.append(self.add_X_batch(adata, mean, std, batch_factor, effect, seed))
        return adata_batch_list

    def add_X_batch(self, adata, mean=1, std=0.1, batch_factor=None, effect='*', seed=42):
        """
        为指定的第 i 个 AnnData 对象增加批次效应，只对非零值生效。

        参数:
        - adata: AnnData 对象
        - mean: float，正态分布的均值，用于生成批次效应
        - effect: str, 指定使用那种模式增加批次效应，'*' 乘法， '+' 加法
        - std: float，正态分布的标准差，用于生成批次效应
        """
        # 获取第 i 个 AnnData
        if mean is None or std is None:
            return adata
        if seed == 'auto':
            self.seed += 1
            seed = self.seed

        np.random.seed(seed)
        adata = copy.deepcopy(adata)

        # 获取数据的形状
        num_cells, num_genes = adata.X.shape


        # 为每个基因生成符合正态分布的偏差值
        if batch_factor is None:
            gene_effects = np.random.normal(loc=mean, scale=std, size=num_genes)
            gene_effects = np.where(gene_effects < 0.1, 0.1, gene_effects)
        else:
            gene_effects = batch_factor
        assert gene_effects.shape[0] == num_genes
        # 将批次效应添加到非零值
        for j in range(num_genes):
            # 只对非零值生效
            non_zero_indices = adata.X[:, j] != 0
            if effect == '+':
                adata.X[non_zero_indices, j] += np.round(gene_effects[j])
            elif effect == '*':
                adata.X[non_zero_indices, j] = np.round(adata.X[non_zero_indices, j] * gene_effects[j])
        adata.var['batch_factor'] = gene_effects
        return adata


    def load_adata_dict(self,
                        data_dir_dict,
                        file_names=None,
                        slice_number=1,
                        use_umap=False, norm=False):
        if file_names is None:
            file_names = [f"/adata_slice{i}.h5ad" for i in list(range(slice_number))]
        adata_dict = {}
        for method_name, data_file in data_dir_dict.items():
            # for data_file in
            adata_list = [sc.read_h5ad(data_file + '/' + file) for file in file_names]
            if use_umap == True:
                emb_list = [adata.X for adata in adata_list]
                emb_list = self._umap(emb_list, norm=norm)
                for s, adata in enumerate(adata_list):
                    adata.obsm['umap'] = emb_list[s]
            adata_dict[method_name] = adata_list
        self.adata_dict = adata_dict
        return adata_dict

    def load_h5_list(self,
                              adata_dir,
                              file_names=None,
                              slice_number=1,
                              use_umap=False):
        if file_names is None:
            file_names = [f"adata_slice{i}.h5ad" for i in range(slice_number)]
        adata_list = [sc.read_h5ad(adata_dir + file) for file in file_names]
        if use_umap == True:
            emb_list = [adata.X for adata in adata_list]
            emb_list = self._umap(emb_list)
            for s, adata in enumerate(adata_list):
                adata.obsm['umap'] = emb_list[s]
        self.adata_list = adata_list
        return adata_list

    def load_h5(self,adata_dir, data_name):
        data_mat = h5py.File(f'{adata_dir}/{data_name}.h5', 'r', libver='latest', locking=False)
        x = np.array(data_mat['X']).astype('float64')  # count matrix
        loc = np.array(data_mat['pos']).astype('float64')  # location information
        batch_idx = np.array(data_mat['batch']).astype('float64')
        celltype = np.array(data_mat['Y'])


    def concat(self, adata_list):
        # Ensure all items are AnnData objects
        if not all(isinstance(adata, AnnData) for adata in adata_list):
            raise ValueError("All elements in adata_list must be AnnData objects.")
        # Ensure that each AnnData object in adata_list has unique row names (barcodes)
        for i, adata in enumerate(adata_list):
            # Add a prefix to each row name to ensure uniqueness
            adata.obs_names = [f"{i}_{name}" for name in adata.obs_names]

            # Check for duplicate row names and resolve them
            if adata.obs_names.duplicated().any():
                print(f"Warning: Duplicate obs_names detected in sample {i}. Resolving...")
                adata.obs_names = [f"{name}_{i}" if name in adata.obs_names[:i] else name for i, name in
                                   enumerate(adata.obs_names)]

        # Concatenate the AnnData objects
        return sc.concat(adata_list)

    import copy
    from typing import Optional, List
    import scanpy as sc
    import numpy as np

    def generate_MultiOmic_adata_dict_list(
            self,
            RNA_list: Optional[List[sc.AnnData]] = None,
            ATAC_list: Optional[List[sc.AnnData]] = None,
            embedding_list: Optional[List[sc.AnnData]] = None,
            Img_list: Optional[List[sc.AnnData]] = None,
            simulate_missing: bool = False,
            missing_ratio: float = 0.05
    ):
        """
        (Docstring remains the same)
        - v2 Improvement: Simulates synchronized cell dropout across modes within the same slice.
        - v3 Correction: Simulation now ONLY removes cells (rows), not features (columns).
        """
        # Steps 1, 2, and 3: Prepare the data structure (identical to your version)
        input_modes = {'RNA': RNA_list, 'ATAC': ATAC_list, 'embedding': embedding_list, 'IMG': Img_list}
        active_modes = {k: v for k, v in input_modes.items() if v is not None and len(v) > 0}

        if not active_modes: return [], None
        try:
            num_slices = len(next(iter(active_modes.values())))
        except StopIteration:
            return [], None
        for mode_name, data_list in active_modes.items():
            if len(data_list) != num_slices:
                raise ValueError(
                    f"Inconsistent number of slices. Expected {num_slices}, but mode '{mode_name}' has {len(data_list)}.")

        output_dict_list = []
        for i in range(num_slices):
            slice_dict = {}
            for mode_name, data_list in active_modes.items():
                slice_dict[mode_name] = data_list[i]
            output_dict_list.append(slice_dict)

        # Step 4: Handle simulation logic
        if not simulate_missing:
            return output_dict_list
        else:
            ground_truth_dict_list = copy.deepcopy(output_dict_list)
            simulated_dict_list = copy.deepcopy(output_dict_list)

            print(f"🔬 Simulating missing data with a {missing_ratio:.0%} ratio...")

            # Loop through each slice
            for i in range(num_slices):
                # --- IMPROVEMENT: Synchronized Cell Removal ---
                # 1. Get all unique cell names for the *entire slice*
                all_slice_obs_names = set()
                # We only need to iterate through one mode to get all cells, assuming they are aligned.
                # Let's pick the first available mode in the slice.
                first_mode_adata = next(iter(ground_truth_dict_list[i].values()))
                if first_mode_adata is not None:
                    all_slice_obs_names.update(first_mode_adata.obs_names)

                if not all_slice_obs_names:  # Skip if slice is empty
                    continue

                all_slice_obs_names = list(all_slice_obs_names)
                # 2. Choose which cells to remove for this slice, just ONCE.
                n_obs_to_remove = int(len(all_slice_obs_names) * missing_ratio)
                obs_names_to_remove_for_slice = set(np.random.choice(
                    all_slice_obs_names, size=n_obs_to_remove, replace=False
                ))

                # Now apply this synchronized removal mask to all modes in the slice
                for mode_name in active_modes.keys():
                    if simulated_dict_list[i][mode_name] is None:
                        continue

                    truth_adata = ground_truth_dict_list[i][mode_name]
                    sim_adata = simulated_dict_list[i][mode_name]

                    # --- Handle Cells (Obs) ---
                    # Keep cells that are NOT in the pre-selected removal set
                    obs_to_keep_mask = ~sim_adata.obs_names.isin(obs_names_to_remove_for_slice)
                    removed_obs_names = sim_adata.obs_names[~obs_to_keep_mask].tolist()
                    truth_adata.uns['simulated_missing_obs'] = removed_obs_names  # Record what was removed

                    # --- MODIFICATION: Feature (Var) removal logic is now deleted ---
                    # The code block for removing columns has been removed entirely.

                    # Slice the AnnData with only the synchronized cell mask.
                    # The colon ":" selects all features (columns).
                    simulated_dict_list[i][mode_name] = sim_adata[obs_to_keep_mask, :].copy()

            return simulated_dict_list, ground_truth_dict_list


    def mode_concat(self, adata_list=None, adata_dict=None, adata_dict_list=None, mode_name=None, location='spatial'):
        assert sum([not adata_list is None, not adata_dict is None, not adata_dict_list is None])==1
        _mode_ = None
        if not adata_dict_list is None:
            assert adata_list is None and adata_dict is None
            if mode_name is None:
                mode_name = list(adata_dict_list[0].keys())
            assert not mode_name is None
            _mode_ = 'adata_dict_list'
        if not adata_list is None:
            assert adata_dict is None and adata_dict_list is None
            assert len(adata_list) == len(mode_name)
            assert not mode_name is None
            adata_dict_list = [{m:adata for m,adata in zip(mode_name, adata_list)}]
            _mode_ = 'adata_list'

        if not adata_dict is None:
            assert adata_list is None and adata_dict_list is None
            assert mode_name is None or len(mode_name) > len(adata_dict)
            if mode_name is None:
                mode_name = list(adata_dict.keys())
            adata_dict_list = [adata_dict]
            _mode_ = 'adata_dict'

        assert not mode_name is None and len(mode_name) >= 2  # 确保有至少两个模式
        assert not adata_dict_list is None

        # 初始化存储不同模式的 `obs_names` 和 `var_names` 的字典
        self.data_obs_name = {i: [] for i in range(len(adata_dict_list))}
        self.mode_var_name = {m: [] for m in mode_name}

        # 1. 收集所有模式的 `obs_names` 和 `var_names`
        for m in mode_name:
            for i, adata_dict in enumerate(adata_dict_list):
                if m in adata_dict.keys() and not adata_dict[m] is None:  # 只有当模式在字典中时才添加
                    self.data_obs_name[i].extend(list(adata_dict[m].obs_names))
                    self.mode_var_name[m].extend(list(adata_dict[m].var_names))

            self.mode_var_name[m] = list(set(self.mode_var_name[m]))

        for i in range(len(adata_dict_list)):
            self.data_obs_name[i] = list(set(self.data_obs_name[i]))

        if 'IMG' in self.mode_var_name.keys():
            self.mode_var_name['IMG'] = list(adata_dict_list[0]['IMG'].var_names)

        print(mode_name)
        # 2. 为每个模式处理缺失的 `obs` 和 `var`
        adata_dict_list_cat = []  # 用来存放拼接后的数据集
        for i, adata_dict in enumerate(adata_dict_list):
            adata_cat = None
            for m in mode_name:
                if m not in adata_dict.keys() or adata_dict[m] is None:
                    # 如果模式不存在，填充缺失的 `obs` 和 `var` 信息
                    adata_m = self._create_empty_adata(self.data_obs_name[i], self.mode_var_name[m], mode=m)
                else:
                    # 存在该模式的情况下，检查并填充缺失的 `obs` 和 `var`
                    adata_m = adata_dict[m]
                    adata_m = self._fill_missing_data(adata_m, self.data_obs_name[i], self.mode_var_name[m], mode=m)
                adata_m.var['mode'] = m
                adata_m.obsm['mask'] = pd.DataFrame(adata_m.obs['mask'])
                adata_m.obsm['mask'] = pd.DataFrame(adata_m.obsm['mask'].values, columns=[m], index=adata_m.obsm['mask'].index)
                if 'cell_type' in adata_m.obs.columns:
                    adata_m.obs[m + '_cell_type'] = adata_m.obs['cell_type']
                if adata_cat is None:
                    adata_cat = adata_m
                    continue
                # 3. 合并 `obs` 和 `obsm` 数据
                adata_cat = self._merge_obs_obsm(adata_cat, adata_m)
                print(f'{i}, {m}')
                print(adata_cat.obsm.keys())
            adata_cat = adata_cat[~np.isnan(adata_cat.obsm[location]).any(axis=1)]
            adata_combined = adata_cat
            adata_combined.obs = clean_metadata_dataframe(adata_combined.obs.copy())
            adata_combined.var = clean_metadata_dataframe(adata_combined.var.copy())
                # 对 obsm 中的 NaN 进行填充
            for key in adata_combined.obsm.keys():
                    data = adata_combined.obsm[key]
                    if isinstance(data, pd.DataFrame):
                        # 对于 pandas DataFrame, 使用 fillna 方法处理 NaN
                        adata_combined.obsm[key] = data.fillna(0)
                    elif isinstance(data, np.ndarray):
                        # 对于 numpy 数组，使用 np.nan_to_num 填充 NaN
                        adata_combined.obsm[key] = np.nan_to_num(data)
            print(adata_combined)
            adata_combined.uns['mode'] = [m for m in mode_name]
            adata_combined.uns['slice_id'] = i
            if 'IMG' in adata_combined.uns['mode']:
                if 'Original_Image' in adata_dict['IMG'].uns.keys():
                    adata_combined.uns['Original_Image'] = adata_dict['IMG'].uns['Original_Image']
            adata_dict_list_cat.append(adata_combined)
        if _mode_ in ['adata_dict', 'adata_list']:
            return adata_dict_list_cat[0]
        return adata_dict_list_cat

    # 创建一个空的 `AnnData`，用于填充缺失的 `obs` 和 `var` 数据
    def _create_empty_adata(self, mode_obs, mode_var, mode):
        # 初始化一个全为NaN的adata对象
        data = np.nan * np.ones((len(mode_obs), len(mode_var)))
        adata = sc.AnnData(X=data, obs=pd.DataFrame(index=mode_obs), var=pd.DataFrame(index=mode_var))
        adata.obs_names = mode_obs
        adata.var_names = mode_var

        adata_raw = sc.AnnData(X=data)
        adata_raw.obs_names = mode_obs
        adata_raw.var_names = mode_var
        adata.layers['rawX'] = adata_raw.X
        adata.obs['mask'] = np.zeros((adata.shape[0],1))
        adata.obsm['mask'] = pd.DataFrame(np.zeros((adata.shape[0],1)),index=adata.obs_names, columns=[mode])
        return adata

    # 填充缺失的 `obs` 和 `var` 数据
    def _fill_missing_data(self, adata, data_obs_name, mode_var_name, mode):
        # 检查并填充缺失的 `obs` 和 `var` 信息
        missing_obs_name = set(data_obs_name) - set(adata.obs_names)
        missing_var_name = set(mode_var_name) - set(adata.var_names)
        # 如果没有缺失，只需添加mask并返回
        if not missing_obs_name and not missing_var_name:
            adata.obs['mask'] = 1
            adata.var['mask'] = 1
            return adata
        else:
            # 确保X的数据维度与新的 `obs` 和 `var` 兼容
            new_obs = list(data_obs_name)
            new_var = list(mode_var_name)

            # 为了避免下标对不齐，使用DataFrame重新生成 X
            new_data_df = pd.DataFrame(np.zeros((len(new_obs), len(new_var))), index=new_obs, columns=new_var)
            if hasattr(adata.X, 'toarray'):
                new_data_df.loc[adata.obs_names, adata.var_names] = adata.X.toarray()
            else:
                new_data_df.loc[adata.obs_names, adata.var_names] = adata.X

            # 创建新的 AnnData 对象
            # 为了节省内存，如果原始数据是稀疏的，我们也将新数据转换回稀疏格式
            new_X = new_data_df.values
            newadata = sc.AnnData(X=new_X, obs=pd.DataFrame(index=new_obs), var=pd.DataFrame(index=new_var))

            newadata.obs_names = np.array(new_obs).astype(str)
            newadata.var_names = np.array(new_var).astype(str)
            newadata.uns['mode'] = mode

            # --- 复制和填充元数据 ---
            # 复制原有的 `obs` 和 `var` 信息
            newadata.obs = newadata.obs.reindex(columns=adata.obs.columns)
            newadata.obs.loc[adata.obs_names] = adata.obs
            newadata.var = newadata.var.reindex(columns=adata.var.columns)
            newadata.var.loc[adata.var_names] = adata.var

            # 生成 mask
            original_obs_set = set(adata.obs_names)
            newadata.obs['mask'] = newadata.obs.index.isin(original_obs_set).astype(int)
            original_var_set = set(adata.var_names)
            newadata.var['mask'] = newadata.var.index.isin(original_var_set).astype(int)

            # --- <<< 新增：处理 .layers 属性 >>> ---
            if adata.layers:
                print("Filling .layers attribute...")
                for key, layer_matrix in adata.layers.items():
                    # 每个 layer 的维度与 .X 相同
                    new_layer_df = pd.DataFrame(np.zeros((len(new_obs), len(new_var))), index=new_obs, columns=new_var)

                    layer_dense = layer_matrix.toarray() if hasattr(layer_matrix, 'toarray') else layer_matrix
                    new_layer_df.loc[adata.obs_names, adata.var_names] = layer_dense

                    # 将填充后的 layer 存入 newadata，同样考虑稀疏性
                    new_layer = new_layer_df.values
                    newadata.layers[key] = new_layer

            # --- 填充 .obsm 数据 (逻辑不变) ---
            if adata.obsm:
                print("Filling .obsm attribute...")
                for key, obsm_matrix in adata.obsm.items():
                    if isinstance(obsm_matrix, pd.DataFrame):
                        newadata.obsm[key] = pd.DataFrame(np.nan, index=newadata.obs_names, columns=obsm_matrix.columns)
                        newadata.obsm[key].loc[obsm_matrix.index] = obsm_matrix
                    else:
                        source_obsm_data = obsm_matrix.toarray() if hasattr(obsm_matrix, 'toarray') else obsm_matrix
                        newadata.obsm[key] = np.full((len(newadata.obs_names), source_obsm_data.shape[1]), np.nan)
                        # 高效地找到原始 obs 在新 obs 中的索引位置
                        original_indices = pd.Series(np.arange(len(new_obs)), index=new_obs)
                        indices_to_fill = original_indices[adata.obs_names].values
                        newadata.obsm[key][indices_to_fill, :] = source_obsm_data
            return newadata

    def _merge_obs_obsm(self, adata_cat, adata_m, uns=0):
        adata_m = adata_m[adata_cat.obs_names, :]  # Ensure consistency
        from scipy.sparse import hstack
        try:
            new_X = hstack([adata_cat.X, adata_m.X])
        except:
            new_X = np.concatenate([adata_cat.X, adata_m.X], axis=1)

        if 'rawX' in adata_m.layers.keys():
            rawX_cat = adata_cat.layers['rawX']
        else:
            rawX_cat = adata_cat.X

        if 'rawX' in adata_m.layers.keys():
            rawX_m = adata_m.layers['rawX']
        else:
            rawX_m = adata_m.X

        try:
            new_raw_x = hstack([rawX_cat, rawX_m])
        except:
            new_raw_x = np.concatenate([rawX_cat, rawX_m], axis=1)

        new_obs_names = adata_cat.obs_names
        new_var_names = np.concatenate([adata_cat.var_names, adata_m.var_names], axis=0)
        adata_combined = sc.AnnData(X=new_X, obs=adata_cat.obs, var=pd.DataFrame(index=new_var_names))
        adata_combined.obs_names = new_obs_names
        adata_combined.var_names = new_var_names
        adata_combined.layers['rawX'] = new_raw_x

        if uns == 0:
            adata_combined.uns = adata_cat.uns
        else:
            adata_combined.uns = adata_m.uns

        # Step 1: Perform default merge for all obs columns (prioritizing adata_cat's values)
        adata_combined.obs = adata_cat.obs.combine_first(adata_m.obs)

        # --- Start of new logic: Apply a special merging rule for the 'cell_type' column ---
        print("Applying special merging rules for 'cell_type'...")
        cat_has_cell_type = 'cell_type' in adata_cat.obs.columns
        m_has_cell_type = 'cell_type' in adata_m.obs.columns

        if cat_has_cell_type and m_has_cell_type:
            # If both adata objects contain 'cell_type'
            num_cat_types = adata_cat.obs['cell_type'].nunique()
            num_m_types = adata_m.obs['cell_type'].nunique()
            print(f"  - Number of cell types in adata_cat: {num_cat_types}")
            print(f"  - Number of cell types in adata_m: {num_m_types}")

            if num_m_types > num_cat_types:
                # If adata_m has more cell types, overwrite the default merge result with adata_m's 'cell_type' column
                print("  -> Decision: Keeping 'cell_type' from adata_m.")
                adata_combined.obs['cell_type'] = adata_m.obs['cell_type']
            else:
                # Otherwise, keep adata_cat's 'cell_type' column (which combine_first already does)
                print("  -> Decision: Keeping 'cell_type' from adata_cat.")
                # No code needed here as the default behavior is correct

        elif m_has_cell_type and not cat_has_cell_type:
            # If only adata_m contains 'cell_type'
            print("  -> Decision: 'cell_type' found only in adata_m, keeping it.")
            adata_combined.obs['cell_type'] = adata_m.obs['cell_type']

        elif cat_has_cell_type and not m_has_cell_type:
            # If only adata_cat contains 'cell_type'
            print("  -> Decision: 'cell_type' found only in adata_cat, keeping it.")
            # No code needed here as the default behavior is correct

        else:
            # If neither contains 'cell_type'
            print("  - 'cell_type' not found in either adata object.")
        # --- End of new logic ---

        # Merge 'var': Concatenate 'var' directly, row-wise
        adata_combined.var = pd.concat([adata_cat.var, adata_m.var], axis=0)
        for key in adata_cat.obsm.keys():
            if key in adata_cat.obsm.keys():
                adata_combined.obsm[key] = adata_cat.obsm[key]

        # Fill in missing `obsm` data (logic remains unchanged)
        for key in adata_m.obsm.keys():
            if key in adata_cat.obsm.keys():
                if isinstance(adata_m.obsm[key], pd.DataFrame):
                    adata_combined.obsm[key] = adata_cat.obsm[key].combine_first(adata_m.obsm[key])
                    new_columns_in_m = set(adata_m.obsm[key].columns) - set(adata_cat.obsm[key].columns)
                    for col in new_columns_in_m:
                        adata_combined.obsm[key][col] = adata_m.obsm[key][col]
                else:
                    mask_cat = np.isnan(adata_cat.obsm[key])
                    adata_combined.obsm[key] = np.where(mask_cat, adata_m.obsm[key], adata_cat.obsm[key])
            else:
                adata_combined.obsm[key] = adata_m.obsm[key]

        return adata_combined

    def split_rna_atac(self, adata):
        return split_rna_atac(adata)

    def DEseq2_analysis(self, adata, layer_key="cell_type", raw=False):
        """
        Perform differential expression analysis (DE) on adata using pydeseq2.

        Parameters:
        - adata: AnnData object with expression data in `adata.X` and cell type information in `adata.obs[layer_key]`.
        - layer_key: Key in adata.obs indicating the layer or cell type.

        Returns:
        - results_dict: Dictionary with DE results including log2FoldChange, p-values, and adjusted p-values for each layer.
        """
        # 提取 counts 数据和条件数据
        n_samples = 20000
        if adata.shape[0] > n_samples:
            # 随机采样 n_samples 个细胞
            sampled_cells = np.random.choice(adata.obs_names, size=n_samples, replace=False)
            adata = adata[sampled_cells, :]

        if raw==False:
            counts_df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
                                     index=adata.obs_names, columns=adata.var_names)
        else:
            counts_df = pd.DataFrame(adata.layers['rawX'].toarray() if hasattr(adata.layers['rawX'], "toarray") else adata.layers['rawX'],
                                     index=adata.obs_names, columns=adata.var_names)

        # 创建条件数据 DataFrame，确保索引与 counts_df 保持一致
        condition_df = pd.DataFrame(adata.obs[[layer_key]].copy())  # 复制需要的列
        condition_df.index = adata.obs_names  # 设置索引为 adata.obs_names
        condition_df.columns = ['condition']  # 重命名列为 'condition'
        condition_df['condition'] = condition_df['condition'].astype(str)
        # 获取唯一的层（细胞类型）
        unique_layers = np.unique(condition_df['condition'])

        results_list = []

        for layer in unique_layers:
            print(f'开始 {layer} ...')
            # 为当前层构建条件
            condition_df_temp = condition_df.copy()
            condition_df_temp['condition'] = condition_df['condition'].apply(lambda x: 'A' if x == layer else 'B')
            if condition_df_temp['condition'].nunique() < 2:
                print(f"Skipping layer {layer} because it has less than two unique conditions.")
                continue
            # 创建 DeseqDataSet 对象
            # Initialize DeseqDataSet without the `design` parameter
            metadata_df = condition_df_temp[['condition']]  # This ensures we provide the necessary metadata
            dds = DeseqDataSet(counts=counts_df, metadata=metadata_df)

            # Now set the design formula after initializing
            dds.design = '~condition'  # Set the design formula
            # 运行 DESeq2 分析
            dds.deseq2()

            # 执行统计分析并返回结果
            res = DeseqStats(dds)
            res.summary()

            # 获取分析结果 DataFrame
            res_df = res.results_df
            res_df['-logpadj'] = res_df.padj.apply(lambda x: -math.log10(x) if x > 0 else 0)

            # 根据 log2FoldChange 和 p-value 筛选上下调基因
            res_df.loc[(res_df.log2FoldChange > 1) & (res_df.padj < 0.05), 'type'] = 'up'
            res_df.loc[(res_df.log2FoldChange < -1) & (res_df.padj < 0.05), 'type'] = 'down'
            res_df.loc[(abs(res_df.log2FoldChange) <= 1) | (res_df.padj >= 0.05), 'type'] = 'nosig'

            # 将当前层的结果添加到列表中
            results_list.append((layer, res_df))
            print(f'结束 {layer} ...')

        # 返回每个层级的差异表达分析结果字典
        return {layer: res for layer, res in results_list}


    def load_adata_list(self,
                        adata_dir,
                        file_names=None,
                        slice_number=1,
                        cell_type=None,
                        use_umap=False,
                        max_workers=None):  # 您可以调整并行的线程数
        from concurrent.futures import ThreadPoolExecutor
        if max_workers is None:
            max_workers = max(4, os.cpu_count()-2)
        if file_names is None:
            file_names = [f"adata_slice{i}.h5ad" for i in range(slice_number)]
        filepaths = [f"{adata_dir}/{file}" for file in file_names]
        # 这个辅助函数将在每个线程中并行运行
        def read_single_adata(path):
            return sc.read_h5ad(path)
        print(f"正在使用最多 {max_workers} 个线程并行读取 {len(filepaths)} 个文件...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(read_single_adata, filepaths)
            adata_list = list(results)  # 将 map 迭代器转换为列表
        print("所有文件加载完毕。")
        if use_umap:
            emb_list = [adata.X for adata in adata_list]
            emb_list = self._umap(emb_list)
            for s, adata in enumerate(adata_list):
                adata.obsm['umap'] = emb_list[s]
        if cell_type is not None:
            for adata in adata_list:
                adata.obs['cell_type'] = adata.obs[cell_type]
        return adata_list

    def load_adata_and_separate(self,
                    adata_file,
                    use_umap=False,
                    sample_order='sample_order'):
        adata = sc.read_h5ad(adata_file)
        slice_orders = adata.obs[sample_order].unique()
        adata_list = []
        for i, slice_order in enumerate(slice_orders):
            adata_slice = adata[adata.obs[sample_order] == slice_order]
            adata_list.append(adata_slice.copy())
        if use_umap == True:
            emb_list = [adata.X for adata in adata_list]
            emb_list = self._umap(emb_list)
            for s, adata in enumerate(adata_list):
                adata.obsm['umap'] = emb_list[s]
        return adata_list

    def load_adata(self,
                        adata_file,
                        use_umap=False):
        adata = sc.read_h5ad(adata_file)
        if use_umap == True:
            adata_umap = self._umap([adata])
            adata.obsm['umap'] = adata_umap[0]
        return adata

    def umap_X(self,X_list):
        emb_list = [X for X in X_list]
        return self._umap(emb_list)

    def umap_adata_list_embedding(self, adata_list, embedding=None, batch=False, removalBE=False, Zeros=False,
                                  umap_name='embedding_umap'):
        adata_list = [copy.deepcopy(adata) for adata in adata_list]
        if embedding is not None:
            if isinstance(embedding, list):
                assert embedding.__len__() == adata_list.__len__()
                embedding_list = [adata.obsm[emb] for adata, emb in zip(adata_list, embedding)]
            else:
                embedding_list = [adata.obsm[embedding] for adata in adata_list]
            emb_list = embedding_list
        else:
            gp_list = [adata.obsm['gp_mu'] for adata in adata_list]
            gaussian_list = [adata.obsm['gaussian_mu'] for adata in adata_list]
            if (batch == True or removalBE == True or 'batch_mu' in adata_list[0].obsm) and Zeros == False:
                batch_list = [adata.obsm['batch_mu'] for adata in adata_list]
                if removalBE == True:
                    batch_pm = np.array([_batch.mean(0) for _batch in batch_list])
                    batch_pm_mean = batch_pm.mean(0)
                    dist = ((batch_pm - batch_pm_mean) ** 2).sum(axis=1)
                    batch_pm_slice_number = np.argmin(dist)
                    batch_pm = batch_pm[batch_pm_slice_number]
                    batch_list = [np.zeros(adata.obsm['batch_mu'].shape) + batch_pm for adata in adata_list]
                emb_list = [np.concatenate([gp, gauss, batch], axis=1) for gp, gauss, batch in
                            zip(gp_list, gaussian_list, batch_list)]
            else:
                emb_list = [np.concatenate([gp, gauss], axis=1) for gp, gauss in
                            zip(gp_list, gaussian_list)]

        emb_list_umap = self._umap(emb_list)
        adata_list_pred = []
        for adata, emb, u in zip(adata_list, emb_list, emb_list_umap):
            adata.obsm[umap_name] = u
            adata.obsm[umap_name + '_embedding'] = emb
            adata_list_pred.append(adata)
        return adata_list_pred

    def umap_adata_list(self, adata_list, counts=False, norm=False):
        emb_list = [adata.X for adata in adata_list]
        emb_list = self._umap(emb_list, counts=counts, norm=norm)
        for adata, emb in zip(adata_list, emb_list):
            adata.obsm['umap'] = emb
        return adata_list

    def get_adata_list(self, normalize=True):
        """Return the list of AnnData objects."""
        if normalize == True:
            return self.adata_list
        else:
            raw_adata_list = []
            adata_list = copy.deepcopy(self.adata_list)
            for adata in adata_list:
                adata.X = adata.layers['rawX']
                raw_adata_list.append(adata)
                del adata.obs["size_factors"]
            return raw_adata_list

    def get_raw_adata_list(self):
        raw_adata_list = []
        adata_list = copy.deepcopy(self.adata_list)
        for adata in adata_list:
            adata.X = adata.layers['rawX']
            adata.obsm['spatial'] = adata.obsm['location']
            raw_adata_list.append(adata)
            del adata.obs["size_factors"]
            del adata.raw
        return raw_adata_list

    def get_SupRes_adata(self, adata_list=None, group_size_list=None):
        self.x_init, self.loc_init, self.slice_values, self.celltype_init = self.sub_process_data(
            self.x_init,
            self.loc_init,
            self.slice_values,
            self.celltype_init,
            slices_list=adata_list,
            group_size_list=group_size_list)
        return self.preprocess_data()

    def get_unique_celltype(self, adata_list=None):
        if adata_list is None:
            return list(self.unique_celltype)
        else:
            # 提取所有adata中的.obs['cell_type']，并返回唯一值
            cell_types = []
            for adata in adata_list:
                cell_types.extend(adata.obs['cell_type'].unique())  # 获取每个adata的唯一细胞类型并累加
            return list(set(cell_types))  # 返回唯一的细胞类型集合

    def generate_SupRes_adata(self, adata, group_size=4):
        assert group_size > 0 and (group_size & (group_size - 1)) == 0  # "group_size 必须是 2 的次方"
        x = adata.X
        loc = adata.obsm['spatial']
        celltype = adata.obs['cell_type']
        slices_idx = adata.obs['slice_view'] + 10
        x, loc, slices_idx, celltype = _sub_process_data_(x, loc, slices_idx, celltype, group_size=group_size)
        # final_x, final_loc, final_slices_idx, final_celltype
        adata = sc.AnnData(x)
        adata.obsm['spatial'] = loc
        adata.obs['cell_type'] = celltype

        return adata

    def generate_SupRes_adata_list(self, adata_list, group_size_list=None):
        if group_size_list is None:
            group_size_list = [4 for _ in adata_list]
        re_adata_list = []
        for gs, adata in zip(group_size_list, adata_list):
            adata = self.generate_SupRes_adata(adata, gs)
            re_adata_list.append(adata)
        return re_adata_list

    def get_transform_adata_list(self, adata_list=None, position_name='spatial', scaler_spatial=False, scaler_spatial_global=False, rotation=None,
                                 translation=None, scale=None, keep4overlap=None, changed_slice_set=None, loc_range=None):
        if adata_list is None and hasattr(self, 'adata_list'):
            adata_list = self.adata_list
        if changed_slice_set is None:
            if hasattr(self, 'changed_slice_set'):
                changed_slice_set = self.changed_slice_set[:len(adata_list)]
            else:
                changed_slice_set = list(range(len(adata_list)))
        if not hasattr(self, 'loc_range'):
            self.loc_range = 20
        if not loc_range is  None:
            self.loc_range = loc_range
        self.rotation = [0 for _ in changed_slice_set] if rotation is None or len(rotation) != len(
            changed_slice_set) else rotation
        self.translation = [[0, 0] for _ in changed_slice_set] if translation is None or len(translation) != len(
            changed_slice_set) else translation
        self.scale = [1 for _ in changed_slice_set] if scale is None or len(scale) != len(changed_slice_set) else scale
        self.keep4overlap = [[1, 1] for _ in changed_slice_set] if keep4overlap is None or len(keep4overlap) != len(
            changed_slice_set) else keep4overlap
        self.MinMaxScaler_list = []
        if adata_list is not None:
            self.adata_transform_list = copy.deepcopy(adata_list)
        else:
            self.adata_transform_list = copy.deepcopy(self.adata_list)

        for i, _ in enumerate(self.adata_transform_list):
            spatial_coords = self.adata_transform_list[i].obsm[position_name]  # (N, 2) array of x, y coordinates
            self.adata_transform_list[i].obsm['old_spatial'] = spatial_coords
            keep4overlap = self.keep4overlap[i]  # keep4overlap value for slicing (0 to 1, 1 to 2, etc.)
            if keep4overlap is not None:
                xlap, ylap = keep4overlap  # Unpack the overlap values for x and y

                # 计算 x 轴和 y 轴的分割
                split_x = self._get_split_indices(spatial_coords, xlap, axis=0)
                split_y = self._get_split_indices(spatial_coords, ylap, axis=1)

                split_ = split_x & split_y
                self.adata_transform_list[i] = self.adata_transform_list[i][split_, :]

            # Store the transformed spatial coordinates

            # Apply the transformation functions
            try:
                loc = translated_func(rotation_func(self.adata_transform_list[i].obsm[position_name],
                                                self.rotation[i],
                                                self.scale[i]),
                                                self.translation[i])
            except:
                loc = self.adata_transform_list[i].obsm[position_name]
            self.adata_transform_list[i].obsm['Trans_spatial'] = copy.deepcopy(loc)
            if scaler_spatial == True and not scaler_spatial_global == True:
                scaler = MinMaxScaler()
                loc = scaler.fit_transform(loc) * self.loc_range
                self.MinMaxScaler_list.append(scaler)

            self.adata_transform_list[i].obsm['spatial'] = loc
            y = self.adata_transform_list[i].obs['layers_label']
            # Create layer dictionary
            if hasattr(self, 'layers_dict'):
                layers_dict = {}
                for l in self.adata_transform_list[i].uns['layers_dict'].keys():
                    layers_dict[l] = np.where(y == l)[0]
                self.adata_transform_list[i].uns['layers_dict'] = layers_dict

        if scaler_spatial_global == True:
            loc_data_list = []
            original_shapes = []
            for adata in self.adata_transform_list:
                loc_data = adata.obsm['Trans_spatial']
                loc_data_list.append(loc_data)
                original_shapes.append(loc_data.shape[0])  # Record the number of rows (samples)
            all_loc_data_stacked = np.vstack(loc_data_list)
            scaler = MinMaxScaler()
            scaled_loc_data_stacked = scaler.fit_transform(all_loc_data_stacked) * self.loc_range
            current_index = 0
            scaled_loc_data_list = []
            for shape in original_shapes:
                scaled_loc_data = scaled_loc_data_stacked[current_index:current_index + shape]
                scaled_loc_data_list.append(scaled_loc_data)
                current_index += shape

            # Step 4: Assign the scaled data back to each adata object
            for i, adata in enumerate(self.adata_transform_list):
                adata.obsm['spatial'] = scaled_loc_data_list[i]

        return self.adata_transform_list

    def sub_process_data(self, x, loc, slices_idx, celltype, slices_list=None, group_size_list=None):
        x = np.array(x)
        if slices_list is None:
            slices_list = self.changed_slice_set
        if group_size_list is None:
            group_size_list = self.group_size_list

        assert slices_list.__len__() == group_size_list.__len__()
        for slice, group in zip(slices_list, group_size_list):
            if group == 1 or group == None:
                continue
            x, loc, slices_idx, celltype = _sub_process_data_(x, loc, slices_idx, celltype, slice, group)
        return x, loc, slices_idx, celltype

    def self_add_z_position(self, adata_list=None):
        z = 0
        if adata_list is None:
            adata_list = self.adata_list
        self.z_list = []
        self.real_z_list = []
        for i, adata_slice in enumerate(adata_list):
            assert adata_slice.obsm['spatial'].shape[-1] == 2
            adata_slice.obs['z_position'] = adata_slice.obsm['z_position'] = (z + np.ones(
                adata_slice.obs['cell_index'].shape[0])) / self.adata_list.__len__() * self.loc_range
            self.real_z_list.append(z + 1)
            adata_slice.obsm['spatial'] = np.concatenate([adata_slice.obsm['spatial'], adata_slice.obsm['z_position'].reshape((-1,1))], axis=1)
            z += 1
            self.z_list.append(adata_slice.obs['z_position'])
        return adata_list

    def rec_ztest(self, ztest):
        ztest = ((ztest - 0) + 1) / self.z_list.__len__() * self.loc_range
        return ztest

    def self_add_timestamp(self, time_range=10):
        time_list = list(range(self.adata_list.__len__()))
        self.add_time_to_adata_list(time_list, time_range=time_range)


    def self_mask(self, slice_index, mask_array, mask_mode='RNA'):
        if slice_index < len(self.adata_list):
            if type(mask_array) == int:
                if mask_array == 1:
                    self.adata_list[slice_index].obsm['mask'][mask_mode] = np.ones((self.sv[slice_index].shape[0]))
                elif mask_array == 0:
                    self.adata_list[slice_index].obsm['mask'][mask_mode] = np.zeros((self.sv[slice_index].shape[0]))
            elif type(mask_array) == np.ndarray:
                assert mask_array.shape[0] == self.adata_list[slice_index].obsm['mask'].shape[0]
                self.adata_list[slice_index].obsm['mask'][mask_mode] = mask_array
        else:
            raise IndexError("Slice index out of range.")

    def process_adata(self, adata, batch_name=None, min_genes=200, min_counts=3):
        if not batch_name is None:
            adata.obs["batch"] = batch_name
        sc.pp.filter_cells(adata, min_genes=min_counts)
        sc.pp.filter_genes(adata, min_counts=min_genes)
        if issparse(adata.X):
            adata.X = adata.X.toarray()  # 或者使用 adata.X = adata.X.todense()
        return adata


    def scaler_adata_list_spatial(self, adata_list=None, loc_range=20):
        if adata_list is None:
            adata_list=self.adata_list
        self.MinMaxScaler_list = []
        for adata in adata_list:
            loc = adata.obsm['spatial']
            scaler = MinMaxScaler()
            adata.obsm['spatial'] = scaler.fit_transform(loc) * loc_range
            self.MinMaxScaler_list.append(scaler)
        return adata_list


    def SpaRefine(self, adata, shape="hexagon", celltype='cell_type'):
        """
        对adata对象进行空间优化，基于细胞的空间坐标信息和邻居的聚类标签进行多数表决，以调整聚类结果。

        Args:
        adata: AnnData对象，包含细胞数据和空间坐标信息
        shape: str, 网格形状，可选择'square'或'hexagon'

        Returns:
        adata: 经过空间优化的AnnData对象，更新了聚类标签
        """
        # 获取空间坐标
        spatial_coords = adata.obsm['spatial']

        # 计算距离矩阵
        dis = pairwise_distances(spatial_coords, metric="euclidean", n_jobs=-1).astype(np.double)

        # 获取聚类标签
        pred = adata.obs[celltype].values.astype(int)
        sample_id = np.arange(len(pred))

        # 定义邻居数
        if shape == "hexagon":
            num_nbs = 6
        elif shape == "square":
            num_nbs = 4
        else:
            raise ValueError("Shape not recognized. Use 'hexagon' for Visium data, 'square' for ST data.")

        # 多数表决更新标签
        refined_pred = []
        pred_df = pd.DataFrame({"pred": pred}, index=sample_id)
        dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)

        for i in range(len(sample_id)):
            index = sample_id[i]
            # 找到距离最近的 num_nbs 个邻居，跳过自身
            dis_tmp = dis_df.loc[index, :].sort_values()
            nbs = dis_tmp.iloc[1:(num_nbs + 1)]  # 确保不包括自身的距离
            nbs_pred = pred_df.loc[nbs.index, "pred"]
            self_pred = pred_df.loc[index, "pred"]
            v_c = nbs_pred.value_counts()

            # 判断是否需要更新当前细胞的聚类标签
            if self_pred not in v_c:
                # 如果没有与自身类别相同的邻居，则选择最多的类别
                refined_pred.append(v_c.idxmax())
            elif v_c[self_pred] < num_nbs / 2 and np.max(v_c) > num_nbs / 2:
                # 如果同类别邻居数量不足且存在其他类别超过一半，则更新为多数类别
                refined_pred.append(v_c.idxmax())
            else:
                # 否则保留原类别
                refined_pred.append(self_pred)

        # 更新adata中的聚类标签
        adata.obs['refine_cell_type'] = np.array(refined_pred)

        return adata


    def bbknn_cluster_celltype(self, adata_list, resolution=1, spaRefine=True, rbatch=True):
        """
        拼接数据，执行数据标准化、选择高变基因，批次效应校正，计算bbknn邻域图，返回独立的聚类后的adata列表。
        """
        # 获取批次名称
        batch_names = []
        for i, adata in enumerate(adata_list):
            if 'batch' not in adata.obs or rbatch:
                adata.obs["batch"] = str(i)
                adata_list[i] = adata
            batch_names.append(adata.obs["batch"][0])

        adata_list_init = adata_list
        adata_list = copy.deepcopy(adata_list)

        # 拼接数据
        adata_concat = adata_list[0].concatenate(*adata_list[1:],
                                                 batch_key="batch",
                                                 batch_categories=batch_names,
                                                 index_unique="-")

        # 数据标准化和log1p变换
        sc.pp.normalize_total(adata_concat)
        sc.pp.log1p(adata_concat)

        # 选择高变基因
        sc.pp.highly_variable_genes(adata_concat, batch_key="batch")
        var_select = adata_concat.var.highly_variable_nbatches >= 2
        adata_concat = adata_concat[:, var_select]

        # PCA降维和BBKNN校正
        sc.tl.pca(adata_concat)
        sc.external.pp.bbknn(adata_concat, batch_key="batch")

        # Leiden聚类
        sc.tl.leiden(adata_concat, resolution=resolution)

        # 4. 将拼接后的 AnnData 对象分割回原始的 AnnData 对象列表
        split_list = []
        start_idx = 0
        for adata in adata_list:
            end_idx = start_idx + adata.shape[0]
            split_list.append(adata_concat[start_idx:end_idx, :])  # 分割数据
            start_idx = end_idx  # 更新开始索引
        adata_list = split_list
        if spaRefine == True:
            adata_list = [self.SpaRefine(adata, celltype='leiden') for adata in adata_list]

        for i, (adata_init, adata) in enumerate(zip(adata_list_init, adata_list)):
            adata_init.obs['cell_type'] = adata.obs['refine_cell_type'].values
            adata_list_init[i] = adata_init
        return adata_list_init


    def aggregate_to_adata(self, adata_list, output_dir):
            merge_counts = None
            var_names = None
            for adata in adata_list:
                # 提取 counts 和 layer 信息
                counts = adata.layers['rawX'] if 'rawX' in adata.layers.keys() else adata.X
                layer = adata.obs['layer_guess_reordered'].astype(str)
                # 聚合数据
                counts_agg = pd.DataFrame(counts.T, index=adata.var_names, columns=adata.obs_names).groupby(layer,
                                                                                                            axis=1).sum()

                # 删除 NA 列
                counts_agg = counts_agg.loc[:, ~counts_agg.columns.str.contains('NA')]

                # 保存每个样本的聚合数据
                sampleid = adata.uns['sample_id']
                counts_agg.columns = [f"{sampleid}_{col}" for col in counts_agg.columns]
                counts_agg.to_csv(f"{output_dir}/{sampleid}_agg_counts.csv")

                # 合并所有样本
                if merge_counts is None:
                    merge_counts = counts_agg
                    var_names = adata.var_names
                else:
                    merge_counts = pd.concat([merge_counts, counts_agg], axis=1)

            # 保存合并后的数据
            merge_counts.to_csv(f"{output_dir}/merged_agg_counts.csv")

            # 转换为 AnnData 对象
            merge_adata = sc.AnnData(
                X=csr_matrix(merge_counts.values),
                var=pd.DataFrame(index=var_names),
                obs=pd.DataFrame(index=merge_counts.columns)
            )
            return merge_adata

    def align_adata_list_genes(self, adata_list, process=True, add_batch = False, min_genes=200, min_counts=3):
        """
        拼接多个AnnData对象，按照基因集合对齐，缺失基因用0填充，并在拼接后按原始大小分割成新的AnnData对象列表。

        参数：
        - adata_list: List[AnnData]，包含多个AnnData对象的列表。

        返回：
        - new_adata_list: List[AnnData]，按原始大小分割的AnnData对象列表。
        """
        # 1. 获取所有基因的并集
        all_genes = None

        for i, adata in enumerate(adata_list):
            if process == True:
                if add_batch != False:
                    adata = self.process_adata(adata, str(i), min_genes=min_genes, min_counts=min_counts)
                else:
                    adata = self.process_adata(adata, min_genes=min_genes, min_counts=min_counts)
                adata_list[i] = adata

            if all_genes is None:
                all_genes = set(adata.var_names)
            else:
                all_genes = all_genes.intersection(set(adata.var_names))

        # 2. 对每个 AnnData 对象，按照基因集合进行对齐
        def align_genes(adata):
            adata_new = adata[:, list(all_genes)]
            return adata_new

        # 对每个 AnnData 对象进行基因对齐
        aligned_adata_list = [align_genes(adata) for i, adata in enumerate(adata_list)]

        return aligned_adata_list

    def save_adata_list(self,adata_list=None,save_dir='.', identifier=None,center_slice=None,save_slices=None, compression=None):
        if adata_list is None:
            adata_list = self.adata_list
        if center_slice is None:
            center_slice = self.center_slice
        if identifier is not None:
            self.identifier = identifier
        if hasattr(self,'identifier') and self.identifier is not None:
            f_h5 = f'{save_dir}/{self.identifier}/'
        else:
            f_h5 = f'{save_dir}/'
        os.makedirs(f_h5, exist_ok=True)
        print('训练结束...')
        if save_slices is None:
            save_slices = range(adata_list.__len__())
        for i, adata in zip(save_slices, adata_list):
            print('正在写入adata.h5ad文件...')
            filename = f_h5 + f'adata_slice{i}.h5ad'
            adata.write_h5ad(filename, compression=compression)
            print(f"Saved {filename}")
            print(adata)

    def save_adata_dict(self,adata_dict=None,save_dir='.', identifier=None,center_slice=None):
        save_slices, adata_list = adata_dict.keys(), adata_dict.values()
        self.save_adata_list(adata_list=adata_list,save_dir=save_dir, identifier=identifier,center_slice=center_slice,save_slices=save_slices)

    def save_adata(self,adata,save_dir='.',identifier=None,save_name=None,center_slice=None):
        if center_slice is None:
            center_slice = self.center_slice
        if identifier is not None:
            self.identifier = identifier
        if self.identifier is not None:
            f_h5 = f'{save_dir}/{self.identifier}/'
        else:
            f_h5 = f'{save_dir}/'
        os.makedirs(f_h5, exist_ok=True)
        print('训练结束...')
        print('正在写入adata.h5ad文件...')
        if save_name is None:
            filename = f_h5 + f'adata_slice.h5ad'
        else:
            filename = f_h5 + save_name
        adata.write_h5ad(filename, compression='gzip')
        print(f"Saved {filename}")
        print(adata)

    def save_df2csv(self,df, save_dir='.',identifier=None,save_name=None,center_slice=None):
        if identifier is not None:
            self.identifier = identifier
        if hasattr(self, 'identifier') and self.identifier is not None:
            f_h5 = f'{save_dir}/{self.identifier}/'
        else:
            f_h5 = f'{save_dir}/'
        os.makedirs(f_h5, exist_ok=True)
        if save_name is None:
            filename = f_h5 + f'data.csv'
        else:
            filename = f_h5 + save_name
        df.to_csv(filename)
        print(f'保存至：{filename}')


    def _get_split_indices(self, coords, keep_value, axis=0):
        """
        计算根据 overlap_value 分割坐标的布尔索引。
        如果 overlap_value 在 [0, 1] 之间，排序并根据比例分割坐标。
        """
        if keep_value == 0 or keep_value == 1:
            # No change, keep all coordinates
            return np.ones(coords.shape[0], dtype=bool)

        # 排序坐标，降序排列
        sorted_indices = np.argsort(coords[:, axis])  # 升序排序

        # 计算分割点的索引
        threshold_index = int(coords.shape[0] * (keep_value if keep_value>0 else abs(-1-keep_value)))  # 根据 overlap_value 计算索引
        threshold_value = coords[sorted_indices[threshold_index], axis]  # 获取排序后的阈值

        if keep_value > 0:
            # 如果 overlap_value 是正的，表示选择坐标小于阈值的部分
            return coords[:, axis] <= threshold_value
        else:
            # 如果 overlap_value 是负的，表示选择坐标大于阈值的部分
            return coords[:, axis] >= threshold_value

    def _umap(self, emb_list, counts=False, norm=False):
        emb = np.concatenate(emb_list, axis=0)
        if counts == True:
            adata = sc.AnnData(emb)
            if norm == True:
                sc.pp.normalize_per_cell(adata)
            sc.pp.log1p(adata)
            sc.tl.pca(adata)
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            reduced_data = adata.obsm['X_umap']
        else:
            umap_model = UMAP(n_components=2, random_state=42)
            reduced_data = umap_model.fit_transform(emb)
        # print('')
        slices_len = len(emb_list)
        views = np.concatenate([np.zeros(gm.shape[0]) + j for j, gm in enumerate(emb_list)])
        emb_list = [reduced_data[views == j, :] for j in range(slices_len)]
        return emb_list

    def filter_spatial_region(self, adata, xy=[], save_all_slice=False):
        """
        根据给定的 [xmin, xmax, ymin, ymax] 坐标范围过滤掉 AnnData 对象中的 spatial 数据点，
        并返回一个新的 AnnData 对象。

        参数:
            adata (AnnData): 输入的 AnnData 对象
            xmin (float): x 的最小值
            xmax (float): x 的最大值
            ymin (float): y 的最小值
            ymax (float): y 的最大值

        返回:
            AnnData: 过滤后的 AnnData 对象
        """
        # 获取 spatial 数据 (假设在 obsm 中名为 'spatial')
        spatial = adata.obsm['spatial']
        xmin, xmax, ymin, ymax = xy
        # 筛选出不在指定区域的索引
        mask = ~((spatial[:, 0] >= xmin) & (spatial[:, 0] <= xmax) &
                 (spatial[:, 1] >= ymin) & (spatial[:, 1] <= ymax))

        # 使用筛选后的索引创建新的 AnnData 对象
        if save_all_slice == True:
            adata_new = adata.copy()
        else:
            adata_new = adata[mask].copy()
        adata_mask = adata[~mask].copy()
        adata_new.uns['mask_normalized_counts'] = adata_mask.obsm['normalized_counts']
        adata_new.uns['mask_spatial'] = adata_mask.obsm['spatial']
        spatial = adata_mask.obsm['spatial']
        return adata_new, spatial

    def random_delete_spots(self, adata, rate):
        """
        根据给定的比例随机删除 AnnData 对象中的 spot（obs），并返回一个新的 AnnData 对象。

        参数:
            adata (AnnData): 输入的 AnnData 对象
            rate (float): 删除的比例，取值范围 [0, 1)

        返回:
            AnnData: 经过随机删除 spot 后的新的 AnnData 对象
        """
        if not (0 <= rate < 1):
            raise ValueError("rate 必须在 [0, 1) 范围内")

        # 总的 spot 数量
        n_spots = adata.n_obs

        # 需要删除的 spot 数量
        n_to_delete = int(n_spots * rate)

        # 随机选择要删除的索引
        all_indices = np.arange(n_spots)
        indices_to_delete = np.random.choice(all_indices, size=n_to_delete, replace=False)

        # 创建掩码，标记需要保留的索引
        mask = np.ones(n_spots, dtype=bool)
        mask[indices_to_delete] = False  # 将删除的索引标记为 False

        # 使用掩码创建新的 AnnData 对象
        adata_new = adata[mask].copy()
        adata_mask = adata[~mask].copy()
        spatial = adata_mask.obsm['spatial']
        return adata_new, spatial

    def find_outliers(point_cloud, k=5, distance_sigma=3, plot_points=False):
        """
        使用K近邻算法查找点云中的孤立点，并进行归一化处理。
        自动设置一个合理的阈值。

        参数:
        point_cloud (numpy.ndarray): 输入的点云数据，形状为(n_samples, n_features)。
        k (int): 选择邻居的数量。

        返回:
        outliers (numpy.ndarray): 识别出的孤立点索引。
        """
        # 归一化处理
        scaler = StandardScaler()
        point_cloud_normalized = scaler.fit_transform(point_cloud)

        # 构建K近邻模型
        nbrs = NearestNeighbors(n_neighbors=k).fit(point_cloud_normalized)

        # 计算每个点到其k个最近邻的距离
        distances, _ = nbrs.kneighbors(point_cloud_normalized)

        # 取每个点与其k个最近邻的平均距离
        mean_distances = np.mean(distances, axis=1)

        # 计算平均距离的标准差，用于自动设置阈值
        mean_distance_mean = np.mean(mean_distances)
        mean_distance_std = np.std(mean_distances)

        # 自动设置阈值，例如：平均距离加上两倍的标准差
        threshold = mean_distance_mean + distance_sigma * mean_distance_std

        # 找出超过阈值的点作为孤立点
        outliers = np.where(mean_distances > threshold)[0]

        return outliers

    def find_inner_points(self, point_cloud_A, point_cloud_B, k=5, distance_sigma=3, plt_img=False):
        """
        找到两个点云的内点（交点）。

        参数:
        point_cloud_A (numpy.ndarray): 第一个点云数据，形状为(n_samples_A, n_features)。
        point_cloud_B (numpy.ndarray): 第二个点云数据，形状为(n_samples_B, n_features)。
        distance_threshold (float, optional): 判断内点的距离阈值。如果为None，则自动计算。
        k (int): 在计算最近邻时考虑的邻居数量。

        返回:
        inner_points_A (numpy.ndarray): 点云A中的内点索引。
        inner_points_B (numpy.ndarray): 点云B中的对应内点索引。
        """

        def plot_point_clouds_with_inner_points(point_cloud_A, point_cloud_B, inner_points_A, inner_points_B,
                                                title='Point Clouds with Inner Points', point_size=1, plt_img=False):
            plt.figure(figsize=(8, 8))
            plt.scatter(point_cloud_A[:, 0], point_cloud_A[:, 1], label='Point Cloud A', alpha=0.6, s=point_size)
            plt.scatter(point_cloud_B[:, 0], point_cloud_B[:, 1], label='Point Cloud B', alpha=0.6, s=point_size)
            plt.scatter(point_cloud_A[inner_points_A, 0], point_cloud_A[inner_points_A, 1], label='Inner Points (A)',
                        color='red', s=point_size)
            plt.scatter(point_cloud_B[inner_points_B, 0], point_cloud_B[inner_points_B, 1], label='Inner Points (B)',
                        color='green', s=point_size)
            plt.legend()
            plt.title(title)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True)
            plt.show()

        def find_inner(nbrs_A, nbrs_B, point_cloud_A_normalized, point_cloud_B_normalized):
            # 为点云B中的每个点找到在点云A中的最近邻
            distances_B_to_A, indices_B_to_A = nbrs_A.kneighbors(point_cloud_B_normalized)
            distances_A_to_A, indices_A_to_A = nbrs_A.kneighbors(point_cloud_A_normalized)
            # 如果distance_threshold为None，则自动计算一个合理的阈值
            mean_distance_A_to_A = np.mean(distances_A_to_A[:, 1:])  # 忽略第一个元素，因为它是点到自身的距离
            std_distance_A_to_A = np.std(distances_A_to_A[:, 1:])
            distance_threshold = mean_distance_A_to_A + distance_sigma * std_distance_A_to_A  # 例如，使用两倍标准差作为阈值
            # 找出内点
            inner_points_mask_B = distances_B_to_A[:, 0] <= distance_threshold
            inner_points_B = np.where(inner_points_mask_B)[0]

            return inner_points_B

        # 合并点云进行归一化
        combined_point_cloud = np.vstack((point_cloud_A, point_cloud_B))
        scaler = StandardScaler().fit(combined_point_cloud)
        point_cloud_A_normalized = scaler.transform(point_cloud_A)
        point_cloud_B_normalized = scaler.transform(point_cloud_B)
        # 构建K近邻模型
        nbrs_A = NearestNeighbors(n_neighbors=k).fit(point_cloud_A_normalized)  # 我们只关心最近的邻居
        nbrs_B = NearestNeighbors(n_neighbors=k).fit(point_cloud_B_normalized)  # 但对于B，我们可以考虑k个最近邻来自动确定阈值

        inner_points_B = find_inner(nbrs_A, nbrs_B, point_cloud_A_normalized, point_cloud_B_normalized)
        inner_points_A = find_inner(nbrs_B, nbrs_A, point_cloud_B_normalized, point_cloud_A_normalized)
        if plt_img == True:
            plot_point_clouds_with_inner_points(point_cloud_A, point_cloud_B, inner_points_A, inner_points_B)
        return inner_points_A, inner_points_B


    def filter_outliers_adata_list(self, adata_list, k=10, distance_sigma=3, plt_img=False):
        adata_list_new = []
        def plot_points(point_cloud, outliers, title='Point Cloud with Outliers Identified'):
            """
            可视化点云数据，区分密集点和孤立点。
            参数:
            point_cloud (numpy.ndarray): 输入的点云数据，形状为(n_samples, n_features)。
            outliers (numpy.ndarray): 识别出的孤立点索引。
            title (str): 图表的标题。
            """
            plt.figure(figsize=(8, 8))
            plt.scatter(point_cloud[~np.isin(np.arange(point_cloud.shape[0]), outliers), 0],  # 密集点
                        point_cloud[~np.isin(np.arange(point_cloud.shape[0]), outliers), 1],
                        c='blue', label='Dense Points')
            plt.scatter(point_cloud[np.isin(np.arange(point_cloud.shape[0]), outliers), 0],  # 孤立点
                        point_cloud[np.isin(np.arange(point_cloud.shape[0]), outliers), 1],
                        c='red', marker='x', label='Outliers')
            plt.title(title)
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.legend()
            plt.grid(True)
            plt.show()
        for s, adata in enumerate(adata_list):
            spatial1 = adata.obsm['spatial']
            o1 = self.find_outliers(spatial1, k=k, distance_sigma=distance_sigma)
            if plt_img == True:
                plot_points(spatial1, o1, title=f'Point Cloud with Outliers Identified slice-{s}')
            mask1 = np.zeros(spatial1.shape[0], dtype=bool)
            mask1[o1] = True
            adata = adata[~mask1]
            adata_list_new.append(adata)
        return adata_list_new


    # 其它具有特异性的代码

    def get_DLPFC_adata_list(self, root_file='/home/liuyinbo/whuserver/spaVAE-61server/src/data/2DAlignData/DLPFC',
                             sampleids=["151507", "151508", "151509", "151510", "151669",
                                        "151670", "151671", "151672", "151673", "151674",
                                        "151675", "151676"
                                        ],
                             save_h5=False,
                             ):
        adata_list = []

        for sampleid in sampleids:
            # 构建文件路径
            file_path = f'{root_file}/{sampleid}/{sampleid}_filtered_feature_bc_matrix.h5'
            metadata_path = f'{root_file}/{sampleid}/metadata.tsv'
            csv_file = f'{root_file}/{sampleid}/spatial/tissue_positions_list.csv'

            # 使用 h5py 打开 .h5 文件
            with h5py.File(file_path, 'r', libver='latest', locking=False) as data_mat:
                # 提取矩阵的数据
                data = np.array(data_mat['matrix']['data'])  # 非零数据
                indices = np.array(data_mat['matrix']['indices'])  # 行索引
                indptr = np.array(data_mat['matrix']['indptr'])  # 列指针

                # 获取矩阵的形状
                shape = np.array(data_mat['matrix']['shape'])
                # 创建稀疏矩阵
                sparse_matrix = csr_matrix((data, indices, indptr), shape=shape[::-1])

                # 提取数据
                genename = np.array(data_mat['matrix']['features']['genome']).astype('U26')

                # 将数据转化为 AnnData 对象

                metadata_df = pd.read_csv(metadata_path, sep='\t', index_col=0)
                adata = sc.AnnData(X=sparse_matrix, obs=metadata_df, var={'gene': genename})

                adata.var_names = adata.var['gene']
                pos = pd.read_csv(csv_file, index_col=0, header=None)
                pos = pos.loc[adata.obs['barcode'], :]

                adata.obs['cell_type'] = adata.obs['layer_guess_reordered']
                # 将 AnnData 对象添加到列表中
                adata.obsm['spatial'] = np.array(pos)[:, -4:-2]
                adata.obsm['img_position'] = np.array(pos)[:, -2:]

                adata = adata[~adata.obs['cell_type'].isna(), :]
                adata.var_names_make_unique()
                sc.pp.filter_cells(adata, min_genes=3)
                sc.pp.filter_genes(adata, min_counts=500)
                print(adata)
                adata_list.append(adata)
                if save_h5 == True:
                    h5_save_path = f'{root_file}/{sampleid}/sample_{sampleid}.h5'
                    with h5py.File(h5_save_path, 'w', libver='latest', locking=False) as f:
                        # Create datasets in the .h5 file
                        f.create_dataset('X', data=adata.X.toarray().astype('float64'))  # Count matrix
                        f.create_dataset('pos', data=np.array(adata.obsm['spatial']).astype('float64'))  # Location information
                        f.create_dataset('batch', data=np.arange(len(adata.obs)).astype('float64'))  # Batch information
                        f.create_dataset('gene', data=np.array(adata.var_names).astype('S26'))  # Gene names
                        f.create_dataset('Y', data=np.array(adata.obs['cell_type']).astype('S26'))
        return adata_list


def read_img_DLPFC(slices_name):
    xycoords_list = []
    img_coordinates_list = []
    img_list = []
    for i, sn in enumerate(slices_name):
        data = pd.read_csv(f"../DLPFC/{sn}/spatial/tissue_positions_list.csv", header=None)
        img = np.array(Image.open(f"../DLPFC/{sn}/spatial/{sn}_full_image.tif"))
        x_y_coords = data.iloc[:, -4:-2].values  # 获取倒数第4和第3列（x, y)
        last_two_columns = data.iloc[:, -2:].values  # 获取倒数第2和第1列
        xycoords_list.append(x_y_coords)
        img_coordinates_list.append(last_two_columns)
        img_list.append(img)
    return img_list, img_coordinates_list, xycoords_list


def read_dataset(adata, transpose=False, copy=False):

    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError

    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error

    if adata.X.size < 50e6: # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if transpose: adata = adata.transpose()

    print('### Autoencoder: Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

    return adata


def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True, assert_norm=True):
    if adata.X.dtype.kind == 'i':  # 检查是否为整数类型
        adata.X = adata.X.astype(np.float32)
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
        adata.layers['rawX'] = adata.X.copy()
    else:
        adata.raw = adata
        adata.layers['rawX'] = adata.X.copy()
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obsm['normalized_counts'] = adata.X.copy()
        if assert_norm:
            assert adata.obsm['normalized_counts'].all() >= 0
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if normalize_input:
        sc.pp.scale(adata)
    return adata


def geneSelection(data, threshold=0, atleast=10, 
                  yoffset=.02, xoffset=5, decay=1.5, n=None, 
                  plot=True, markers=None, genes=None, figsize=(6,3.5),
                  markeroffsets=None, labelsize=10, alpha=1, verbose=1):
    """
    Gene selection by mean-variance relationship
    """


    if sparse.issparse(data):
        zeroRate = 1 - np.squeeze(np.array((data>threshold).mean(axis=0)))
        A = data.multiply(data>threshold)
        A.data = np.log2(A.data)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.squeeze(np.array(A[:,detected].mean(axis=0))) / (1-zeroRate[detected])
    else:
        zeroRate = 1 - np.mean(data>threshold, axis=0)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        mask = data[:,detected]>threshold
        logs = np.zeros_like(data[:,detected]) * np.nan
        logs[mask] = np.log2(data[:,detected][mask])
        meanExpr[detected] = np.nanmean(logs, axis=0)

    lowDetection = np.array(np.sum(data>threshold, axis=0)).squeeze() < atleast
    zeroRate[lowDetection] = np.nan
    meanExpr[lowDetection] = np.nan
            
    if n is not None:
        up = 10
        low = 0
        for t in range(100):
            nonan = ~np.isnan(zeroRate)
            selected = np.zeros_like(zeroRate).astype(bool)
            selected[nonan] = zeroRate[nonan] > np.exp(-decay*(meanExpr[nonan] - xoffset)) + yoffset
            if np.sum(selected) == n:
                break
            elif np.sum(selected) < n:
                up = xoffset
                xoffset = (xoffset + low)/2
            else:
                low = xoffset
                xoffset = (xoffset + up)/2
        if verbose>0:
            print('Chosen offset: {:.2f}'.format(xoffset))
    else:
        nonan = ~np.isnan(zeroRate)
        selected = np.zeros_like(zeroRate).astype(bool)
        selected[nonan] = zeroRate[nonan] > np.exp(-decay*(meanExpr[nonan] - xoffset)) + yoffset
                
    if plot:
        if figsize is not None:
            plt.figure(figsize=figsize)
        plt.ylim([0, 1])
        if threshold>0:
            plt.xlim([np.log2(threshold), np.ceil(np.nanmax(meanExpr))])
        else:
            plt.xlim([0, np.ceil(np.nanmax(meanExpr))])
        x = np.arange(plt.xlim()[0], plt.xlim()[1]+.1,.1)
        y = np.exp(-decay*(x - xoffset)) + yoffset
        if decay==1:
            plt.text(.4, 0.2, '{} genes selected\ny = exp(-x+{:.2f})+{:.2f}'.format(np.sum(selected),xoffset, yoffset), 
                     color='k', fontsize=labelsize, transform=plt.gca().transAxes)
        else:
            plt.text(.4, 0.2, '{} genes selected\ny = exp(-{:.1f}*(x-{:.2f}))+{:.2f}'.format(np.sum(selected),decay,xoffset, yoffset), 
                     color='k', fontsize=labelsize, transform=plt.gca().transAxes)

        plt.plot(x, y, color=sns.color_palette()[1], linewidth=2)
        xy = np.concatenate((np.concatenate((x[:,None],y[:,None]),axis=1), np.array([[plt.xlim()[1], 1]])))
        t = plt.matplotlib.patches.Polygon(xy, color=sns.color_palette()[1], alpha=.4)
        plt.gca().add_patch(t)
        
        plt.scatter(meanExpr, zeroRate, s=1, alpha=alpha, rasterized=True)
        if threshold==0:
            plt.xlabel('Mean log2 nonzero expression')
            plt.ylabel('Frequency of zero expression')
        else:
            plt.xlabel('Mean log2 nonzero expression')
            plt.ylabel('Frequency of near-zero expression')
        plt.tight_layout()


        if markers is not None and genes is not None:
            if markeroffsets is None:
                markeroffsets = [(0, 0) for g in markers]
            for num,g in enumerate(markers):
                i = np.where(genes==g)[0]
                plt.scatter(meanExpr[i], zeroRate[i], s=10, color='k')
                dx, dy = markeroffsets[num]
                plt.text(meanExpr[i]+dx+.1, zeroRate[i]+dy, g, color='k', fontsize=labelsize)
    
    return selected




import numpy as np

def extract_subimages(image, coordinates, h):
    """
    从图像中提取以坐标为中心点的子图（NumPy版本）。

    参数:
    image (numpy.ndarray): 输入图像，形状为 (3, 500, 400)。
    coordinates (numpy.ndarray): 坐标列表，形状为 (1000, 2)，每个坐标为子图的中心点。
    h (int): 子图的大小为 (h, h)。

    返回:
    numpy.ndarray: 包含所有子图的数组，形状为 (1000, 3, h, h)。
    """

    if np.argmin(image.shape) == 2:
        image = np.transpose(image, (2, 0, 1))
    subimages = []
    c, height, width = image.shape
    for coord in coordinates:
        x, y = coord.astype(int)  # 坐标是整数

        # 计算提取区域
        x1 = max(x - h // 2, 0)
        x2 = min(x + h // 2, width)
        y1 = max(y - h // 2, 0)
        y2 = min(y + h // 2, height)

        subimage = image[:, y1:y2, x1:x2]

        # 填充不足的部分
        if subimage.shape[1] < h or subimage.shape[2] < h:
            top = h // 2 - (y - y1)
            bottom = (y + h // 2) - y2
            left = h // 2 - (x - x1)
            right = (x + h // 2) - x2

            subimage_padded = np.pad(subimage, ((0, 0), (top, bottom), (left, right)), mode='constant',
                                     constant_values=0)
        else:
            subimage_padded = subimage

        subimages.append(subimage_padded)

    return np.stack(subimages)



import copy
def estimate_size_factors_and_reconstruct_raw(adata):
    # 创建 AnnData 对象
    # 估算 size_factors
    size_factors_estimate = np.median(np.sum(np.expm1(adata.X), axis=1))

    # 反标准化
    if 'scale' in adata.uns:
        scale_params = adata.uns['scale']
        mean = scale_params['mean']
        var = scale_params['var']
        reconstructed_X = (adata.X * np.sqrt(var)) + mean
    else:
        reconstructed_X = adata.X

    # 反对数变换
    reconstructed_X = np.expm1(reconstructed_X)

    # 反归一化（使用估算的 size_factors）
    reconstructed_X = reconstructed_X * size_factors_estimate
    min_values = reconstructed_X.min(axis=1)[:, np.newaxis]
    reconstructed_X = reconstructed_X - min_values
    reconstructed_X = reconstructed_X.astype(np.int32)
    reconstructed_X[reconstructed_X < 0] = 0

    # 创建新的 AnnData 对象保存还原后的数据
    raw_adata = sc.AnnData(reconstructed_X, dtype="float64")

    # 添加 size_factors 到原始 adata
    adata.obs['size_factors'] = size_factors_estimate
    adata.raw = raw_adata

    return adata


def generate_labels(data):
    """
    将给定的数据转化为 shape=(*,1) 的标签数组。

    参数:
    data (dict): 包含标签和对应索引的字典。例如：
                 {
                     'Layer1': np.array([1, 22, 30, 48, 53]),
                     'WM': np.array([2, 6, 10, 20, 25]),
                     'Layer5': np.array([4, 12, 14, 17, 28])
                 }

    返回:
    np.ndarray: shape=(*,1) 的标签数组。
    """
    # 获取所有位置的最大值
    max_index = max(np.concatenate(list(data.values())))

    # 初始化一个空数组，大小为最大值+1
    labels_array = np.empty(max_index + 1, dtype=object)

    # 填充标签
    for label, indices in data.items():
        labels_array[indices] = label

    # 过滤掉空值并转换为 shape 为 (*, 1) 的数组
    labels_array = labels_array[labels_array != None].reshape(-1, 1)

    return labels_array

def rotation_func(pos, rotation=0, scale=1):
    theta = np.deg2rad(rotation)
    _pos = pos - np.mean(pos,axis=0)
    # 创建旋转矩阵
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    _pos = _pos * scale
    rotated_coordinates = np.dot(_pos, rotation_matrix)
    rotated_coordinates = rotated_coordinates + np.mean(pos,axis=0)
    return rotated_coordinates

def translated_func(pos, translation_vector):
    return pos + np.array(translation_vector)



def get_available_gpus(min_memory=100, excluded_gpus=[]):
    """
    Returns a list of available GPU ids that have more than 'min_memory' MiB of free memory.

    Args:
    min_memory (int): Minimum amount of free memory (in MiB) required for a GPU to be considered available.

    Returns:
    list: A list of GPU ids that meet the memory requirement.
    """
    try:
        # Run nvidia-smi to get memory usage
        smi_output = subprocess.check_output(
            'nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits',
            shell=True
        ).decode('utf-8')

        # Parse the output
        available_gpus = []
        for line in smi_output.strip().split('\n'):
            gpu_id, free_memory = line.split(',')
            if int(free_memory.strip()) >= min_memory:
                available_gpus.append(int(gpu_id.strip()))
        available_gpus = [gpu for gpu in available_gpus if gpu not in excluded_gpus]
        return available_gpus

    except subprocess.CalledProcessError as e:
        print(f"Failed to run nvidia-smi: {e}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def get_gpu_memory(gpu_id):
    """Returns the free memory of the specified GPU."""
    try:
        # Run nvidia-smi command to check free memory for the specific GPU
        command = f'nvidia-smi -i {gpu_id} --query-gpu=memory.free --format=csv,nounits,noheader'
        memory_free = int(subprocess.check_output(command, shell=True))
        return memory_free
    except subprocess.CalledProcessError as e:
        print(f"Error checking memory for GPU {gpu_id}: {e}")
        return 0  # Assume no memory is free if there is an error


def set_gpu(gpu_id):
    """
    Set the GPU id for the process.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

from collections import Counter, defaultdict

from scipy.spatial import distance
import numpy as np
from collections import Counter


def _sub_process_data_(x, loc, slices_idx, celltype, slice=None, group_size=4):
    """
    根据动态距离分组处理数据。
    """
    # Step 2: 筛选出目标 slice 的部分
    if slice is not None:
        selected_idx = np.where(np.argmax(slices_idx, axis=1) == slice)[0]
        selected_loc = loc[selected_idx]
        selected_x = x[selected_idx]
        selected_celltype = celltype[selected_idx]
        selected_slices_idx = slices_idx[selected_idx]
    else:
        selected_slices_idx = slices_idx
        selected_loc = loc
        selected_x = x
        selected_celltype = celltype

    # 初始化结果存储
    new_loc = []
    new_x = []
    new_celltype = []
    new_slices_idx = []

    # Step 3: 动态分组
    while len(selected_loc) > 0:
        # 获取当前第一个点
        first_point = selected_loc[0]

        # 计算与其他点的距离
        distances = distance.cdist([first_point], selected_loc, metric='euclidean')[0]

        # 按距离排序，找到最近的 group_size 个点
        group_indices = np.argsort(distances)[:group_size]

        # 提取当前组的信息
        loc_batch = selected_loc[group_indices]
        x_batch = selected_x[group_indices]
        celltype_batch = selected_celltype[group_indices]
        slices_idx_batch = selected_slices_idx[group_indices]

        # 计算平均坐标
        avg_loc = np.mean(loc_batch, axis=0)

        # 计算 x 的和
        summed_x = np.sum(x_batch, axis=0)

        # 计算 celltype 的众数
        most_common_celltype = Counter(celltype_batch).most_common(1)[0][0]

        # 选第一个 batch 的索引作为新的批次标签
        new_batch_idx = slices_idx_batch[0]

        # 将结果存入列表
        new_loc.append(avg_loc)
        new_x.append(summed_x)
        new_celltype.append(most_common_celltype)
        new_slices_idx.append(new_batch_idx)

        # 移除当前组的点
        selected_loc = np.delete(selected_loc, group_indices, axis=0)
        selected_x = np.delete(selected_x, group_indices, axis=0)
        selected_celltype = np.delete(selected_celltype, group_indices, axis=0)
        selected_slices_idx = np.delete(selected_slices_idx, group_indices, axis=0)

    # 将处理后的数据转换为数组
    new_loc = np.array(new_loc)
    new_x = np.array(new_x)
    new_celltype = np.array(new_celltype)
    new_slices_idx = np.array(new_slices_idx)
    if slice is None:
        return new_x, new_loc, new_slices_idx, new_celltype
    else:
        # Step 4: 拼接回原始数据
        remaining_idx = np.setdiff1d(np.arange(len(loc)), selected_idx)
        remaining_loc = loc[remaining_idx]
        remaining_x = x[remaining_idx]
        remaining_slices_idx = slices_idx[remaining_idx]
        remaining_celltype = celltype[remaining_idx]

        # 拼接数据
        final_loc = np.vstack([remaining_loc, new_loc])
        final_x = np.vstack([remaining_x, new_x])
        final_slices_idx = np.vstack([remaining_slices_idx, new_slices_idx])
        final_celltype = np.concatenate([remaining_celltype, new_celltype])

        return final_x, final_loc, final_slices_idx, final_celltype
