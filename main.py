# -*- coding: utf-8 -*-
"""
@author: Zhang Yucong(zhangyucong20@mails.ucas.ac.cn)
"""
from torch.utils.data import DataLoader,random_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import multiprocessing as mp
import torch
import torch.nn as nn
import ConvSSR_crs as ConvSS
import PatchDataset_rdm as PatchDataset 
import PatchDataset_mask as PatchDataset 
import time


# 记录开始时间
start_time = time.time()

a=21   #block size
lu=int(a/2)
lv=lu

#feature name
inpath="/root/"
#加载归一化后的数据
dataNameList=[
        "feature1_",   #0
        "feature2_",   #1
        "feature3_",   #2
        "feature4_",   #3
        "feature5_",   #4
        "feature6_",   #5
        "feature7_",   #6
        "feature8_",   #7
        "feature9_",   #8
        "feature10_",   #9
        "feature11_",   #10
        "feature12_",   #11
        "feature13_",   #12
        "feature14_",   #13
        ]

constantNameList=[
    "constantFeature1_",
    "constantFeature2_",
    "constantFeature3_",
    "constantFeature4_",
    "constantFeature5_",
    ]

inName="/root/mask/land1Mask.npy"
mask=np.load(inName)   #0=no data, 1=valid data

nFeature=len(dataNameList)+len(constantNameList)+2
modelID="ACO2_01deg_CRS_ftr"+str(nFeature)+"cst"+str(len(constantNameList))+"_block"+str(a)

def loadData(date):
    globalSample=[]   #(1+feaNum, 1800, 3600) [y,X]

    #读取该月的y
    inName=inpath + "ACO2_monthTotal/EDGAR_2024_GHG_CO2_monTotal_"+date+ ".npy"
    yData=np.load(inName)
    globalSample.append(yData)

    #读取该月的X
    for k in range(0,len(dataNameList)):
        inName= inpath + dataNameList[k] + date + ".npy" 
        XData=np.load(inName)
        XData[np.isnan(XData)]=0   #gaps=0
        globalSample.append(XData)

    #读取常数变量
    for k in range(0,len(constantNameList)): 
        XData=np.load(constantNameList[k])
        globalSample.append(XData)
    
    #生成该月时间编码
    monNum=float(m)
    sin_m = np.sin(2 * np.pi * monNum / 12)
    cos_m = np.cos(2 * np.pi * monNum / 12)
    mSin=np.zeros(yData.shape,dtype=np.float32)+sin_m
    mCos=np.zeros(yData.shape,dtype=np.float32)+cos_m
    
    globalSample.append(mSin)
    globalSample.append(mCos)
    
    globalSample=np.asarray(globalSample, dtype=np.float32)

    return globalSample


if __name__ == '__main__':
    print(modelID)
    
    outpath="/root/opt/"
   
      
    #---读取数据---
    data_all=[]   #(timeNum, feaNum, W, H) 
    
    #生成时间列表
    for y in range(2019,2022):
        #遍历月
        for m in range(1,13):
            if m<10:
                mNo='0'+str(m)
            else:
                mNo=str(m)
            date=str(y)+mNo
            
            data=loadData(date)
            
            data_all.append(data)
        
    #数据加载
    nSample=30_000_000
    dataset = PatchDataset.PatchDataset(data_all, mask=mask, halfPatch_size=(int(a/2), int(a/2)), total_patches=nSample)
    """
    dataset = PatchDataset_all.PatchDataset(data_all, halfPatch_size=(int(a/2), int(a/2)))
    nSample=len(dataset)
    """
    #随机划分
    train_size = int(0.8 * nSample)
    test_size = nSample - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(21))  # 设置随机种子保证可重现
    print(nSample,len(train_dataset),len(test_dataset))
    # 创建 DataLoader
    batch_size=64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)

    #---网络训练---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #加载模型
    config=ConvSS.Config(nFeature)
    net = ConvSS.ConvSS2t1(config)
    
    #加载训练好的权重
    model_weight_path=outpath+'netParam/ACO2_01deg_CRS_ftr21cst5_block21.pth'  
    net.load_state_dict(torch.load(model_weight_path, map_location=device))  
      
    net.to(device)

    #print(model)   # 打印网络结构
    save_path = outpath+'netParam/'+modelID+'.pth'   
    
    #训练
    loss_function = nn.MSELoss()
    
    epochs = 15
    lr=0.061
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)   #余弦退火调度器
    
    trainloss_list = []
    testloss_list = []
    trainloss2_list = []
    lr_list=[]
    
    print("---Training begins---")
    best_loss = float(np.inf)   #全新训练的best_loss取正无穷
    for epoch in range(epochs):
        
        # train
        net.train()
        train_loss = 0.0
        for data in train_loader:
            X, y = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            
            outputs = net(X)
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # test
        net.eval()
        val_loss = 0.0
        train_loss2 = 0.0
        with torch.no_grad():
            for train_data in train_loader:
                train_X, train_y = train_data[0].to(device), train_data[1].to(device)
                outputs = net(train_X)
                loss = loss_function(outputs, train_y)
                train_loss2 += loss.item()
            
            for val_data in test_loader:
                val_X, val_y = val_data[0].to(device), val_data[1].to(device)
                outputs = net(val_X)
                loss = loss_function(outputs, val_y)
                val_loss += loss.item()

        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(test_loader)
        train_loss2_avg = train_loss2 / len(train_loader)
        trainloss_list.append(train_loss_avg)
        testloss_list.append(val_loss_avg)
        trainloss2_list.append(train_loss2_avg)

        if val_loss_avg < best_loss:
            best_loss = val_loss_avg
            torch.save(net.state_dict(), save_path)
        
        if epoch%5==0:
            logname=outpath+'netParam/'+modelID+'_log.txt'
            log=open(logname,'w+')
            print("epoch:{}, val_loss:{}".format(epoch, val_loss_avg),file=log)
            print([outputs, val_y],file=log)
            log.close()

        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        lr_list.append(current_lr)

        print("epoch: %d #train loss: %f #test loss: %f" % (epoch, train_loss2_avg, val_loss_avg))

    print("---Training is over---")

    net.eval()     
    
    #训练集结果
    y_tr_all=[]
    outputAll=[]  
    for data in train_loader:
        X, y = data[0].to(device), data[1].to(device)
        outputs = net(X)
        outputs1 = outputs.cpu().detach().numpy()
        outputAll.extend(outputs1)
        y = y.cpu().detach().numpy()
        y_tr_all.extend(y)
        
    trR2=r2_score(y_tr_all, outputAll)
    trRMSE=mean_squared_error(y_tr_all, outputAll)**0.5
    
    #测试集结果
    #加载训练结果最好的网络
    net.load_state_dict(torch.load(save_path, map_location=device))
    yAll=[]
    outputAll=[]
    for data in test_loader:
        X, y = data[0].to(device), data[1].to(device)
        outputs = net(X)
        outputs1 = outputs.cpu().detach().numpy()
        outputAll.extend(outputs1)
        y = y.cpu().detach().numpy()
        yAll.extend(y)
        
    teR2=r2_score(yAll, outputAll)
    teRMSE=mean_squared_error(yAll, outputAll)**0.5

    #输出测试集数据
    df=pd.DataFrame(columns=["yTruth","pre"])
    df["yTruth"]=yAll
    df["pre"]=outputAll
    outname=outpath+'netParam/'+modelID+'_testSet.csv'
    df.to_csv(outname) 

    #记录模型信息
    outname=outpath+'netParam/'+modelID+'.txt'
    modelRcrd=open(outname,'w+')
    print("* model: \n",file=modelRcrd)
    print(modelID,file=modelRcrd)
    print(net,file=modelRcrd)
    print('* model path: \n{}'.format(save_path),file=modelRcrd)
    print('* input feature: \n{}, trans distance weight'.format(dataNameList),file=modelRcrd)
    print('* samples per month: nSample={}'.format(nSample),file=modelRcrd)
    print('* block='+str(a)+'\n',file=modelRcrd)
    print('* epoch:{}'.format(epochs),file=modelRcrd)
    print('* lr_list:\n{}'.format(lr_list),file=modelRcrd)
    print('* trainloss0:\n{}'.format(trainloss_list),file=modelRcrd)
    print('* testloss:\n{}'.format(testloss_list),file=modelRcrd)
    print('* trainloss2:\n{}'.format(trainloss2_list),file=modelRcrd)
    print("train R2:", trR2, ", RMSE:", trRMSE,file=modelRcrd)
    print("test R2:", teR2, ", RMSE:", teRMSE,file=modelRcrd)
    modelRcrd.close()
    #命令行输出
    print('#modelID:',modelID,'   #featureName:',dataNameList,'   #train R2:', trR2, ', RMSE:', trRMSE,'test R2:', teR2, ', RMSE:', teRMSE)

    # 计算运行时间
    end_time = time.time()
    total_time = end_time - start_time
    print(f"程序运行总时间: {total_time/60:.2f} 分钟")

    

