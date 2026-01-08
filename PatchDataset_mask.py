# -*- coding: utf-8 -*-
"""
select all the samples from where mask==1
@author: Zhang Yucong(zhangyucong20@mails.ucas.ac.cn)
"""


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random



class PatchDataset(Dataset):
    def __init__(self, big_arrays, mask, halfPatch_size=(12, 12)):
        """
        big_arrays: list of numpy arrays, each shape (C, H, W)
        mask: get patches from where mask==1
        """
        self.big_arrays = big_arrays
        self.halfH, self.halfW = halfPatch_size
        self.patch_h = 2 * self.halfH + 1
        self.patch_w = 2 * self.halfW + 1
        self.mask = mask
        
        #Build coords list
        self.coords = []   # 每个 patch 对应的大图索引 + 中心像元坐标
        rows, cols = np.where(mask == 1)
        self.valid_coords = list(zip(rows, cols))
        for img_idx in range(len(big_arrays)):
            for r, c in self.valid_coords:
                self.coords.append((img_idx, r, c))

        print(f"PatchDataset: {len(self.coords)} patches")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        img_idx, row, col = self.coords[idx]
        
        X = self.big_arrays[img_idx][1:]
        y = self.big_arrays[img_idx][0:1]
        
        # Label: center of channel 0
        label = y[:, row, col]
        
        # feature channels: [1 :]
        patch = X[:, row-self.halfH:row+self.halfH+1, col-self.halfW:col+self.halfW+1].astype(np.float32)   #X
        
        patch = torch.from_numpy(patch.astype(np.float32))
        label = torch.tensor(label, dtype=torch.float32)
        
        return patch, label


if __name__ == "__main__":
    # 模拟 24 个大图: [0]为y，[1:-4]为X, [-4:]为大气传输距离
    #big_arrays = [np.random.rand(8, 1800, 3600).astype(np.float32) for _ in range(3)]
    big_arrays=[]
    for i in range(3):   #3: batch size
        data=np.zeros((8,1800,3600), dtype=np.float32)+i
        data[0]=data[0]+0.1   #y
        big_arrays.append(data)
      
    H, W = 21, 21
    mask=np.zeros((1800,3600),dtype=np.int8)
    mask[1010:1020,2700:2720]=1
    dataset = PatchDataset(big_arrays, mask=mask, halfPatch_size=(int(H/2), int(W/2)))
    loader = DataLoader(dataset, batch_size=8, shuffle=False, pin_memory=True)   #num_workers=4,

    for i, (x, y) in enumerate(loader):
        print(f"Batch {i}: x.shape={x.shape}, y.shape={y.shape}")
        print(x[0],y[0])
        if i >= 1:
            break


