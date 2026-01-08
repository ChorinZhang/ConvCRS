# -*- coding: utf-8 -*-
"""
randomly select samples from where mask==1
@author: Zhang Yucong(zhangyucong20@mails.ucas.ac.cn)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random



class PatchDataset(Dataset):
    def __init__(self, big_arrays, mask, halfPatch_size=(12, 12), total_patches=5_000_000):
        """
        big_arrays: list of numpy arrays, each shape (C, H, W)
        crop_size: (crop_h, crop_w)
        total_patches: dataset 总 patch 数=随机选择的总样本数量
        """
        self.big_arrays = big_arrays
        self.halfH, self.halfW = halfPatch_size
        self.total_patches = total_patches

        self.coords = []  # 每个 patch 对应的大图索引 + 中心像元坐标

        index=np.where(mask==1)
        indexRow=index[0]
        indexCol=index[1]
        indexNum=len(index[0])
        
        for _ in range(total_patches):
            img_idx = random.randint(0, len(big_arrays)-1)
            C, H, W = big_arrays[img_idx].shape
            pickNo=random.randint(0,indexNum-1)
            row=indexRow[pickNo]
            col=indexCol[pickNo]
            self.coords.append((img_idx, row, col))
        print("randomly selected sample's coords")

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        img_idx, row, col = self.coords[idx]

        X = self.big_arrays[img_idx][1:]   #[inFeature, mask]
        y = self.big_arrays[img_idx][0:1]

        patch = X[:, row-self.halfH:row+self.halfH+1, col-self.halfW:col+self.halfW+1]   #X
        patch = torch.from_numpy(patch.astype(np.float32))
        
        label = torch.tensor(y[:,row,col], dtype=torch.float32)   #y

        return patch, label


if __name__ == "__main__":
    # 模拟 24 个大图: [0]为y，[1:-maskChannel]为X, [-maskChannel:]为M
    #big_arrays = [np.random.rand(5, 1800, 3600).astype(np.float32) for _ in range(3)]
    big_arrays=[]
    for i in range(3):   #3: batch size
        data=np.zeros((5,1800,3600), dtype=np.float32)+i
        data[0]=data[0]+0.1   #y
        big_arrays.append(data)
      
    H, W = 25, 25
    dataset = PatchDataset(big_arrays, halfPatch_size=(int(H/2), int(W/2)), total_patches=25)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, pin_memory=True)   #num_workers=4,

    for i, (x, y) in enumerate(loader):
        print(f"Batch {i}: x.shape={x.shape}, y.shape={y.shape}")   #x.shape=torch.Size([B, C, 21, 21]), y.shape=torch.Size([B, 1])
        if i >= 1:
            break


