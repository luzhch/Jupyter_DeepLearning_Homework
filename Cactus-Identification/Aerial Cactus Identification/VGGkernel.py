import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os


def preprocess():
    if os.path.exists('./data') == False:
        os.mkdir('data')
    if os.path.exists('data/train') == False:
        os.mkdir('./data/train')
    if os.path.exists('data/test') == False:
        os.mkdir('./data/test')
    

preprocess()
print(os.listdir('./data'))

import math

class VGG(nn.Module):
    def __init__(self, cfg):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg)
        # linear layer
        self.classifier = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = self.sigmoid(out)
        #out = self.softmax(out)
        return out

    def _make_layers(self, cfg):
        """
        cfg: a list define layers this layer contains
            'M': MaxPool, number: Conv2d(out_channels=number) -> BN -> ReLU
        """
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

vgg_cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
#vggnet = VGG(vgg_cfg['VGG11'])
#print(vggnet)

train_csv = pd.read_csv(open('../input/train.csv'))
#print(train_csv['has_cactus'])
print(len(train_csv))
print(type(train_csv))
print(type(train_csv['id']))
for i in range(10):
    print(type(train_csv.iloc[i]))
for i in train_csv['id'][:10]:
    print(i)

import os
def get_all_file_paths(path):
    files_list = []
    files = os.listdir(path)
    for file in files:
        if not os.path.isdir(path + '/' + file):
            files_list.append(os.path.join(path, file))
        else:
            files_list = files_list + get_all_file_paths(path + '/' + file)
    return files_list

import shutil
import os
def format_image_folder(root,df=None,output_path='./'):
    # image with unknown labels
    if df is None:
        if os.path.exists(os.path.join(output_path, 'nan')) == False:
            os.mkdir(os.path.join(output_path, 'nan'))
        else:
            return 0
        img_paths = get_all_file_paths(root)
        for img_path in img_paths:
            new_path = os.path.join(output_path, 'nan', os.path.basename(img_path))
            shutil.copyfile(img_path, new_path)
        return len(img_paths)
    if os.path.exists(os.path.join(output_path, '0')) == False:
        os.mkdir(os.path.join(output_path, '0'))
    else:
        return 0
    if os.path.exists(os.path.join(output_path, '1')) == False:
        os.mkdir(os.path.join(output_path, '1'))
    else:
        return 0
    for i in range(len(df)):
        row = df.iloc[i]
        img_id = row['id']
        label_ = row['has_cactus']
        img_path = os.path.join(root, img_id)
        if os.path.exists(img_path):
            new_path = os.path.join(output_path, str(label_), img_id)
            shutil.copyfile(img_path, new_path)
    return len(df)

format_image_folder('../input/train/train', train_csv, './data/train')

format_image_folder('../input/test/test', None, './data/test')

files_list = get_all_file_paths('./data/test')
print(len(files_list))
for i in range(10):
    print(files_list[i])

from PIL import Image
import numpy as np

def cal_dataset_mean_std(root):
    imgs_list = get_all_file_paths(root)
    print('images total:', len(imgs_list))
    print('example:')
    img = Image.open(imgs_list[0])
    np_img = np.array(img) / 255
    print('img shape:', np_img.shape)
    print(np_img)
    print('example mean:', np.mean(np_img, axis=(0,1)))
    print('example std:', np.std(np_img, axis=(0,1)))
    
    means = []
    # 计算均值
    for img_path in imgs_list:
        img = Image.open(img_path)
        np_img = np.array(img) / 255
        means.append(np.mean(np_img, axis=(0,1)))
    # means: (n_image, 3)
    mean = np.array(means).mean(axis=(0,))
    
    diff_sqr = []
    for img_path in imgs_list:
        img = Image.open(img_path)
        np_img = np.array(img) / 255
        sqr = (np_img - mean) * (np_img - mean)
        diff_sqr.append(np.mean(sqr, axis=(0,1)))
    std = np.array(diff_sqr).mean(axis=(0,))
    std = np.sqrt(std)
    return mean, std
    
mean, std = cal_dataset_mean_std('./data/train')
print('cal from dataset:')
print('mean:', mean, ' std:', std)

cfg1 = {}
cfg1['batch_size'] = 32
cfg1['lr'] = 0.01
cfg1['weight_decay'] = 0.01
cfg1['momentum'] = 0.9
cfg1['shuffle'] = True
cfg1['loss_func'] = None
cfg1['train_transform'] = transforms.Compose([
    # data augmentation
    #transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    #transforms.RandomCrop(32),

    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])
cfg1['test_transform'] = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])
cfg1['train_set_rate'] = 0.8

vggnet = VGG(vgg_cfg['VGG16'])

data_set = torchvision.datasets.ImageFolder('./data/train', transform=cfg1['train_transform'])
test_set = torchvision.datasets.ImageFolder('./data/test', transform=cfg1['test_transform'])

train_set_size = int(len(data_set) * cfg1['train_set_rate'])
validation_set_size = len(data_set) - train_set_size
[train_set, validation_set] = torch.utils.data.random_split(data_set, [train_set_size, validation_set_size])

data_set_loader = torch.utils.data.DataLoader(data_set, batch_size=cfg1['batch_size'], shuffle=cfg1['shuffle'])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg1['batch_size'], shuffle=cfg1['shuffle'])
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=cfg1['batch_size'], shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg1['batch_size'], shuffle=False)

#optimizer = torch.optim.SGD(vggnet.parameters(), lr=cfg1['lr'], momentum=cfg1['momentum'], weight_decay=cfg1['weight_decay'])
optimizer = torch.optim.Adam(vggnet.parameters(), lr=cfg1['lr'])
#loss_func = nn.CrossEntropyLoss()
loss_func = nn.BCELoss()
device = torch.device('cuda:0')

def predict(outputs, threshold=0.5):
    '''
    outputs: N * 1 (probability of positive)
    return 0-1 labels, N * 1
    '''
    # predicted = torch.Tensor(outputs.float().cpu())
    predicted = outputs.float()
    predicted[predicted>threshold] = 1.
    predicted[predicted<=threshold] = 0.
    return predicted.squeeze()
    

def analyze(predicted, targets):
    correct_predicted = predicted[predicted.eq(targets.float())]
    correct = len(correct_predicted)
    total = targets.size(0)
    TP = len(correct_predicted[correct_predicted==1])
    TN = len(correct_predicted) - TP
    FP = len(predicted[predicted==1]) - TP
    FN = len(predicted[predicted==0]) - TN
    
    return correct, total, TP, TN, FP, FN

def train(model, optimizer, loss_func, train_loader, device):
    # init
    model.train()
    model.to(device)
    loss_func.to(device)
    
    train_loss = 0    # accumulate every batch loss in a epoch
    correct = 0       # count when model' prediction is correct i train set
    total = 0         # total number of prediction in train set
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    # train
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device) # load data to gpu device
        inputs, targets = Variable(inputs), Variable(targets)
        optimizer.zero_grad()            # clear gradients of all optimized torch.Tensors'
        outputs = model(inputs)          # forward propagation return the value of softmax function
        loss = loss_func(outputs, targets.float()) #compute loss
        loss.backward()                  # compute gradient of loss over parameters 
        optimizer.step()                 # update parameters with gradient descent 

        train_loss += loss.item()        # accumulate every batch loss in a epoch
        #_, predicted = outputs.max(1)    # make prediction according to the outputs
        predicted = predict(outputs, 0.5)
        '''
        correct_predicted = predicted[predicted.eq(targets.float())]
        correct += len(correct_predicted)
        total += targets.size(0)
        tp_count = len(correct_predicted[correct_predicted==1])
        TP += tp_count
        tn_count = len(correct_predicted) - tp_count
        TN += tn_count
        FP += len(predicted[predicted==1]) - tp_count
        FN += len(predicted[predicted==0]) - tn_count
        '''
        correct_c, total_c, TP_c, TN_c, FP_c, FN_c = analyze(predicted, targets)
        correct += correct_c
        total += total_c
        TP += TP_c
        TN += TN_c
        FP += FP_c
        FN += FN_c
        
        if (batch_idx+1) % 100 == 0:
            # print loss and acc
            print( 'Train loss: %.3f | Train Acc: %.3f%% (%d/%d) TP: %d TN: %d FP: %d FN: %d'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, TP, TN, FP, FN))
    print( 'Train loss: %.3f | Train Acc: %.3f%% (%d/%d) TP: %d TN: %d FP: %d FN: %d'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, TP, TN, FP, FN))
    return (train_loss,batch_idx+1, correct, total, TP, TN, FP, FN)

def save_model(model, optimizer, path):
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    #torch.save(checkpoint, path)
    return checkpoint

def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

import datetime

def get_cur_time():
    t = datetime.datetime.now()
    t = str(t)
    t = t.replace(' ', '_')
    t = t.replace(':', '_')
    t = t[:19]
    return t

def evaluate(model, test_loader, device):
    model.eval()
    model.to(device)
    
    total = 0         # total number of prediction in train set
    P = 0
    N = 0
    has_cactus = []
    id_ = get_all_file_paths(test_loader.dataset.root)
    id_ = [os.path.basename(p) for p in id_]
    threshold = 0.5
    
    for batch_idx, (inputs, _) in enumerate(test_loader):
        inputs = inputs.to(device) # load data to gpu device
        inputs = Variable(inputs)
        outputs = model(inputs)
        
        #value, predicted = outputs.max(1)    # make prediction according to the outputs
        
        value = outputs.detach().squeeze()
        has_cactus.extend(value.cpu().numpy())
        
        predicted = predict(outputs, 0.5)
        
        if batch_idx == 0:
            print(outputs)
        total += inputs.size(0)
        p_count = len(predicted[predicted==1])
        P += p_count
        N += inputs.size(0) - p_count
    
    df = pd.DataFrame({'id': id_, 'has_cactus': has_cactus})
    #file_name = 'submission_%s_P%d_N%d.csv' % (get_cur_time(), P, N)
    file_name = 'samplesubmission.csv'
    df.to_csv(file_name, index=False, sep=',')
    
    return (total, P, N)

def test(model, loss_func, test_loader, device):
    model.eval()
    model.to(device)
    loss_func.to(device)
    
    test_loss = 0
    correct = 0       # count when model' prediction is correct i train set
    total = 0         # total number of prediction in train set
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device) # load data to gpu device
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)
        loss = loss_func(outputs, targets.float())
        
        test_loss += loss.item()        # accumulate every batch loss in a epoch
        predicted = predict(outputs, 0.5)
        '''
        _, predicted = outputs.max(1)    # make prediction according to the outputs
        total += targets.size(0)
        correct_predicted = predicted.eq(targets.float())
        correct += correct_predicted.sum().item() # count how many predictions is correct
        correct_predicted = predicted[correct_predicted]
        tp_count = len(correct_predicted[correct_predicted==1])
        TP += tp_count
        tn_count = len(correct_predicted) - tp_count
        TN += tn_count
        FP += len(predicted[predicted==1]) - tp_count
        FN += len(predicted[predicted==0]) - tn_count
        '''
        correct_c, total_c, TP_c, TN_c, FP_c, FN_c = analyze(predicted, targets)
        correct += correct_c
        total += total_c
        TP += TP_c
        TN += TN_c
        FP += FP_c
        FN += FN_c
        
    return (test_loss, batch_idx+1, correct, total, TP, TN, FP, FN)

def show_state(epoch=0, state=(0,0,0,0,0,0,0,0), mode=''):
    train_loss, batch_count, correct, total, TP, TN, FP, FN = state
    print( '%s loss: %.5f | %s Acc: %.3f%% (%d/%d) TP: %d TN: %d FP: %d FN: %d'
                % (mode, train_loss/batch_count, mode, 100.*correct/total, correct, total, TP, TN, FP, FN))
    
def run(model, optimizer, loss_func, train_loader, validation_loader, device, num_epochs):
    global best_checkpoint
    model.to(device)
    loss_func.to(device)
    
    # log train loss and test accuracy
    base_epoch = 0
    losses = []
    accs = []
    best_acc = 0
    best_loss = -1
    best_epoch = 0
    best_checkpoint_name = 'BEST_VGG.pth'
    
    for epoch in range (num_epochs):
        print('Epoch: %d/%d' % (epoch + 1, num_epochs))
        state = train(model, optimizer, loss_func, train_loader, device)
        show_state(base_epoch + epoch, state, 'Train')
        print('Test on Validation Set:')
        state = test(model, loss_func, validation_loader, device)
        show_state(base_epoch + epoch, state, 'Test')
        
        # save best model
        loss = state[0]
        correct = state[2]
        total = state[3]
        acc = correct / total
        #if acc > best_acc:
        if best_loss == -1 or loss < best_loss:
            best_acc = acc
            best_loss = loss
            best_epoch = base_epoch + epoch
            best_checkpoint = save_model(model, optimizer, best_checkpoint_name)
            
    # save last model
    correct = state[2]
    total = state[3]
    acc = correct / total
    checkpoint_name = 'checkpoint_%s_Epoch_%d_acc_%.2f' % (get_cur_time(), base_epoch + epoch, acc)
    save_model(model, optimizer, checkpoint_name)
    return best_epoch, best_acc, best_loss

best_checkpoint = None
#best_epoch, best_acc = run(vggnet, optimizer, loss_func, train_loader, validation_loader, device, 10)
best_epoch, best_acc, best_loss = run(vggnet, optimizer, loss_func, data_set_loader, data_set_loader, device, 40)

vggnet.load_state_dict(best_checkpoint['state_dict'])
print('load best epoch: %d acc: %.3f%% best loss: %.4f' % (best_epoch, best_acc * 100, best_loss))
total, P, N = evaluate(vggnet, test_loader, device)
print('Total: %d P: %d N: %d' % (total, P, N))

import shutil
shutil.rmtree('./data')

import matplotlib.pyplot as plt
pred = pd.read_csv('samplesubmission.csv')
print(pred)
plt.scatter(range(4000), pred['has_cactus'])
plt.show()
plt.hist(pred['has_cactus'])

