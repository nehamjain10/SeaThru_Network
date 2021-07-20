from dataset import UnderWater
import torch
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import os
from model import uw_experiment

def mae_loss(output, target):
    loss = torch.sum(torch.abs(output - target)) / output.size(0)
    return loss

def train(lrdp=8,lrdf=0.5):
    
    data_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
        ])

    l1_loss = torch.nn.L1Loss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    lr = 0.0001
    batch_size = 16
    uw_dataset_train = UnderWater(csv_file ="data_train.csv" ,transforms=data_transform)
    dataloader_train = DataLoader(uw_dataset_train, batch_size=batch_size,
                            shuffle=True, num_workers=4)


    uw_dataset_test = UnderWater(csv_file ="data_test.csv" ,transforms=data_transform)
    dataloader_test = DataLoader(uw_dataset_test, batch_size=8,
                            shuffle=True, pin_memory= True,num_workers=4)

    writer = SummaryWriter(comment=f'ALL_DATA_LR_{lr}_BS_{batch_size}')  


    net = uw_experiment()
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, lrdp, gamma=lrdf, last_epoch=-1)
    
    net.to(device=device)

    global_step = 0
    running_loss = 0.0
    for epoch in range(25):  # loop over the dataset multiple times
        for i, batch in enumerate(dataloader_train, 0):
            # get the inputs; data is a list of [inputs, labels]
            input = batch["uw_image"]
            gt_image = batch["gt_image"]
            
            input = input.to(device=device, dtype=torch.float32)
            gt_image = gt_image.to(device=device, dtype=torch.float32)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(input)
            
            loss = l1_loss(outputs, gt_image)
            loss.backward()
            
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:    # every 1000 mini-batches...

                # ...log the running loss
                writer.add_scalar('training loss',
                                loss,
                                global_step)

                # ...log a Matplotlib Figure showing the model's predictions on a
                # random mini-batch

                running_loss = 0.0
            global_step+=1

        if epoch%1==0:
            val_score = validate_model(net,dataloader_test,device,writer,global_step)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('Loss/test', val_score, global_step)

        scheduler.step()

    print('Finished Training')
    if not os.path.exists('models'):
        os.mkdir('models')
    torch.save(net.state_dict(), 'models/' + 'net.pth')
    writer.close()


def validate_model(net,dataloader_test,device,writer,global_step):
    
    net.eval()
    n_val = len(dataloader_test) + 1
    mae = 0
    l1_loss = torch.nn.L1Loss()

    for i,batch in enumerate(dataloader_test):
        input = batch["uw_image"]
        gt_image = batch["gt_image"]
            
        input = input.to(device=device, dtype=torch.float32)
        gt_image = gt_image.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            imgs_pred = net(input)
            mae += l1_loss(imgs_pred, gt_image)
            if i==0:
                writer.add_images('images', input, global_step)
                writer.add_images('result', imgs_pred, global_step)
                writer.add_images('gt_image',gt_image, global_step)
    return mae / n_val


if __name__=="__main__":
    train()
