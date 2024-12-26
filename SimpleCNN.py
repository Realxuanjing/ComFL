import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import os
import glob
from PIL import Image 
import csv
import random
import numpy as np
from torch.utils.data import random_split
import copy
import logging
from tqdm import tqdm
from pathlib import Path
from GetDataSet import GetData,DatasetSplit,LocalUpdate
from utils import seed_everything, set_parameter_requires_grad, self_iid
from Pruning import layer_pruning
from model_merge import update_gNB_model, FedAvg
from model import VGGnet,SimpleCNN,ComplexCNN

# ---------------------设置自定义文件---------------------
logging.basicConfig(filename='output_SimpleCNN.txt', level=logging.INFO, format='%(message)s', filemode='w')
save_path = Path('/home/data1/xxx/dataset/COMFL')
models_dir= save_path / 'models_SimpleCNN'
dataset_path = save_path / 'datasets'/'PetImages'

# -----------------------------------------------------

seed_everything()

# "/home/data1/xxx/dataset/COMFL/datasets/PetImages/"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model=VGGnet().to(device)

#设置参数 
# learning_rate=0.01 #设置学习率
# num_epochs=5   #本地训练次数
# train_batch_size=16
# test_batch_size=16
# fl_epochs=10 #联邦学习次数
# clients_num=20
learning_rate=0.001 #设置学习率
num_epochs=100   #本地训练次数
train_batch_size=32
test_batch_size=32
fl_epochs=100 #联邦学习次数
clients_num=50

#在这里设置德是用户存储模型德文件夹，每个用户一个文件夹，用来存储模型
#大家最开始模型都是一样的VggNet.py
model=SimpleCNN(num_classes=2)
for i in tqdm(range(clients_num), desc="Saving models", unit="client"):
    model_path = models_dir / f'model{i}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # print(f"Saving model to: {model_path / f'model_before_training_client{i}.pth'}")  # 打印保存路径
    # print(f"Model state dict type: {type(model.state_dict())}")  # 打印状态字典类型
    # 保存模型
    torch.save(model.state_dict(), model_path / f'model_before_training_client{i}.pth')
    # tqdm.write(f"client{i} model initialize complete")

#设置优化器，使用CrossEntropyLoss函数
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

#加载数据集和Dataloader
train_data=GetData(dataset_path,224,'train')
test_data=GetData(dataset_path,224,'test')
client_train_dataloader={}
client_test_dataloader={}
user_train_idx={}
user_test_idx={}
user_train_idx=self_iid(train_data,clients_num)
user_test_idx=self_iid(test_data,clients_num)
for i in range(clients_num):
    client_train_dataloader[i]=LocalUpdate(train_data,user_train_idx[i],train_batch_size).return_data(shuffle=True)
    client_test_dataloader[i]=LocalUpdate(test_data,user_test_idx[i],test_batch_size).return_data(shuffle=False)
    
#选择不同的随机用户进行训练
#client_each_epoch=random.sample(1,clients_num/10)
gNB_model=model

gNB_test_dataloader=DataLoader(test_data,batch_size=test_batch_size, shuffle=False)
# pruned_arr = list()
#进行模型聚合
for fl in range(fl_epochs):
    # pruned_arr = []
    #设置联邦学习次数：
    selected_clients=random.sample(range(0,clients_num),min(3, clients_num))
    for client in selected_clients: #每轮设置随机5个用户进行训练
        #每次最多设置10个用户进行训练
        # if fl==0:
        #     pruned_model=gNB_model
        # else:
        #     #现在每个用户剪枝成都不同很合理，因为这里时设置的随机剪掉一定比例的卷积层
        #     pruned_model,pruned_layers = layer_pruning(gNB_model,random.uniform(0.1,0.5))
        #     pruned_arr.append(pruned_layers)
        #不用剪枝
        #只有第一轮联邦学习需要用户本地模型训练，其他时候都是训的聚合模型
        model_to_train=copy.deepcopy(gNB_model)
        model_to_train.to(device)
        if fl==0:
            model_path = models_dir / f'model{client}' / f'model_before_training_client{client}.pth'
            model_to_train.load_state_dict(torch.load(model_path,weights_only=True))
        else:
            model_path=models_dir / f'fl{fl-1}gNB.pth'
            model_to_train.load_state_dict(torch.load(model_path,weights_only=True))
            print("加载GNB下发模型",model_path)
            #r如果需要剪枝，可以放在这里
        print(f"Client {client} model loaded successfully,prepare to train.")
        # pruned_model=gNB_model
        # torch.save(pruned_model.state_dict(),f'model/fl{fl}gNB_Prun.pth')
        # logging.info("model_to_train state_dict: %s", model_to_train.state_dict())
        correct = 0
        total = 0
        model_to_train.eval()
        with torch.no_grad():
            for images, labels in gNB_test_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                output = model_to_train(images)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Before train, fl{}Client{} gNB_test Accuracy  {} %'.format(fl,client,100*(correct/total)))
        logging.info('Before train, fl{}Client{} gNB_test Accuracy  {} %'.format(fl, client, 100 * (correct / total)))
        print("correct and total",correct,total)

        correct = 0
        total = 0
        model_to_train.eval()
        with torch.no_grad():
            for images, labels in client_test_dataloader[client]:
                images = images.to(device)
                labels = labels.to(device)
                output = model_to_train(images)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Before train, fl{}Client{} client_test Accuracy  {} %'.format(fl,client,100*(correct/total)))
        logging.info('Before train, fl{}Client{} client_test Accuracy  {} %'.format(fl, client, 100 * (correct / total)))

        for epoch in range(num_epochs):# 每个用户训练num_epochs次
            total_step=len(client_train_dataloader[client])
            loss_sum=0
            model_to_train.train()
            for i,(img,label) in enumerate(client_train_dataloader[client]):
                #梯度归零，准备开始训练
                optimizer.zero_grad()

                #加标签与图片
                img=img.to(device)
                label=label.to(device)

                #前向计算
                output=model_to_train(img)
                loss=loss_fn(output,label) #前向计算损失函数，但是需要传入output和label两个参数

                #进行优化
                loss.backward()
                optimizer.step()

                loss_sum+=loss.item()

                #print("running Epoch:{}, client_idx:{}, client_round:{},loss is {}".format(epoch,client,i,loss.item()))
                # print("running Epoch:{}, round:{},avg_loss is {}".format(epoch,i,loss_sum/total_step))
                # logging.info("running Epoch:{}, round:{},avg_loss is {}".format(epoch,i,loss_sum/total_step))

        correct = 0
        total = 0
        model_to_train.eval()
        with torch.no_grad():
            for images, labels in client_test_dataloader[client]:
                images = images.to(device)
                labels = labels.to(device)
                output = model_to_train(images)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('fl{}Client{} Test Accuracy  {} %'.format(fl,client,100*(correct/total)))
        logging.info('fl{}Client{} Test Accuracy  {} %'.format(fl, client, 100 * (correct / total)))
        model_name="fl{}client{}.pth".format(fl,client)
        torch.save(obj = model_to_train.state_dict(), f=models_dir / f'model{client}' / model_name)
    
    last_model_path=models_dir / f'fl{fl-1}gNB.pth'
    client_models=[]
    for client in selected_clients:
        model_path= models_dir / f'model{client}/fl{fl}client{client}.pth'
        temp_model = torch.load(model_path,weights_only=True)       
        #对于这个方法，gNB_model的状态信息都变成了temp_model的信息，因为temp_model很可能是剪枝的模型，所以这样为了给gB模型大小保持不变
        if fl==0:
            updated_gNB_model_state_dict = update_gNB_model(copy.deepcopy(gNB_model).state_dict(), temp_model)
        else:
            updated_gNB_model_state_dict = update_gNB_model(torch.load(last_model_path,weights_only=True), temp_model)
        # print('test update_gNB_model',temp_model==updated_gNB_model_state_dict)
        client_models.append(updated_gNB_model_state_dict)
    print("get all clients models ready")
    logging.info("get all clients models ready")

    merged_model=copy.deepcopy(gNB_model)
    sum_module=FedAvg(client_models)
    merged_model.load_state_dict(sum_module)
    # logging.info(f"fl{fl}gNB.pth: {merged_model.state_dict()}")
    print("model prunning done")
    logging.info("model prunning done")
    print("model prunning testing")
    #在这里进行聚合模型性能的测试

    #gNB_test_data=GetData('../dataset/cat_dog_train',224,'test')
    #selected_test_idx=random.sample(range(0,clients_num),1)
    # gNB_test_idx=self_iid(test_data,1)
    # gNB_test_dataloader=LocalUpdate(test_data,gNB_test_idx[0],test_batch_size).return_data()
    
    merged_model.to(device)
    merged_model.eval()
    gNB_correct=0
    gNB_total=0
    with torch.no_grad():
        for images,labels in gNB_test_dataloader:
            images=images.to(device) 
            labels=labels.to(device)
            output=merged_model(images)
            _,predicted=torch.max(output.data,1)
            gNB_total+=labels.size(0)
            gNB_correct+=(predicted==labels).sum().item()
    print('fl{}merged_model Test Accuracy  {} %'.format(fl,100*(gNB_correct/gNB_total)))
    logging.info('fl{}merged_model Test Accuracy  {} %'.format(fl, 100 * (gNB_correct / gNB_total)))
    # print("gnb correct and total",gNB_correct,gNB_total,100 * (gNB_correct / gNB_total),gNB_correct / gNB_total)
    # correct = 0
    # total = 0
    # model_to_train.eval()
    # with torch.no_grad():
    #     for images, labels in gNB_test_dataloader:
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         output = model_to_train(images)
    #         _, predicted = torch.max(output.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    # print('Before train, fl{}Client{} gNB_test Accuracy  {} %'.format(fl,client,100*(correct/total)))
    # logging.info('Before train, fl{}Client{} gNB_test Accuracy  {} %'.format(fl, client, 100 * (correct / total)))

    torch.save(merged_model.state_dict(),models_dir/f'fl{fl}gNB.pth')
    print(f'fl{fl}gNB.pth saved in {models_dir / f"fl{fl}gNB.pth"}')
    # print("FegAvg is done")
    # print("Fl Module prunning")