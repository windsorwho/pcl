# -*- coding: utf-8 -*-

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from torch.utils.tensorboard import SummaryWriter

import pcl_data
import reid_model

FEATURE_FILENAME = '/Users/wenzehu/data/pcl/train/dense_data.npz'
FEATURE_FILENAME = '/data/pcl/train/dense_data.npz'
pcl_dataset = pcl_data.PCLTrainDataset(FEATURE_FILENAME)
dataloaders = {
    x: torch.utils.data.DataLoader(pcl_dataset,
                                   batch_size=512,
                                   shuffle=True,
                                   num_workers=1)
    for x in ['train', 'val']
}
dataset_sizes = {x: pcl_dataset.__len__() for x in ['train', 'val']}
writer = SummaryWriter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# To verify the dataset is loadable.
inputs, classes = next(iter(dataloaders['train']))


def train_model(model,
                criterion,
                optimizer,
                scheduler,
                writer=None,
                num_epochs=25):
    since = time.time()

    #    best_model_wts = copy.deepcopy(model.state_dict())
    #    best_acc = 0.0
    num_itr = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                #if writer is not None:
                if num_itr % 100 == 0 and writer is not None and phase == 'train':
                    writer.add_scalar('LR',
                                      scheduler.get_last_lr()[0], num_itr)
                    writer.add_scalar('Loss/Train',
                                      loss.item() * inputs.size(0), num_itr)
                    writer.add_scalar(
                        'Acc/Train',
                        torch.sum(preds == labels.data) / inputs.size(0),
                        num_itr)
                num_itr += 1
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            writer.add_scalar('Loss/Train_epoc', epoch_loss, num_itr)
            writer.add_scalar('Acc/Train_epoch', epoch_acc, num_itr)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,
                                                       epoch_acc))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrained model and reset final fully connected layer.
#
model_ft = reid_model.ReIDModel(128, 15000)
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
output = model_ft(inputs)
writer.add_graph(model_ft, inputs)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.1, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 15-25 min on CPU. On GPU though, it takes less than a
# minute.
#

model_ft = train_model(model_ft,
                       criterion,
                       optimizer_ft,
                       exp_lr_scheduler,
                       writer=writer,
                       num_epochs=100)
writer.close()
######################################################################
