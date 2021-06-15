import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data as Data
import numpy as np
from collections import Counter
import tensorboardX
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
from resnet18_ import ResNet18
import sys
import os

if not os.path.exists('./results'):
    os.makedirs('./results')

# 导入数据
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

EPOCH = 200
BATCH_SIZE = 128
LR = 0.001



net = ResNet18().to(device)
net.load_state_dict(torch.load('./results/init.pkl'))
# torch.save(net.state_dict(), './results/init.pkl')
opt = optim.Adam(net.parameters(), lr=LR, weight_decay=5e-4)
lossfn = nn.CrossEntropyLoss()

if __name__=='__main__':
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    eve_class_acc = []
    for epoch in range(EPOCH):
        if epoch == 100 or epoch == 150:
            for param_group in opt.param_groups:
                param_group['lr'] *= 0.1

        net.train()
        sum_train_loss = 0.0
        sum_test_loss = 0.0
        sum_train_acc = 0.0
        sum_test_acc = 0.0
        for i_batch, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            outputs = net(inputs)
            loss = lossfn(outputs, labels)
            loss.backward()
            opt.step()
            _, prediction = torch.max(outputs, 1)  # 按行取最大值
            pre_num = prediction.cpu().numpy()
            sum_train_acc += (pre_num == labels.cpu().numpy()).sum() / len(inputs)
            sum_train_loss += loss.item()
        train_loss.append(sum_train_loss / len(trainloader))
        train_acc.append(sum_train_acc / len(trainloader))
        print('Epoch: {}, train_loss: {}, tal_acc: {}'.format(epoch+1, train_loss[-1], train_acc[-1]))

        net.eval()
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        for i_batch, data in enumerate(testloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = lossfn(outputs, labels)
            _, prediction = torch.max(outputs, 1)  # 按行取最大值
            pre_num = prediction.cpu().numpy()
            sum_test_acc += (pre_num == labels.cpu().numpy()).sum() / len(inputs)
            sum_test_loss += loss.item()
            c = (prediction == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            eve_class_acc.append([class_correct[i] / class_total[i] for i in range(10)])
        test_loss.append(sum_test_loss / len(testloader))
        test_acc.append(sum_test_acc / len(testloader))
        print('Epoch: {}, test_loss: {}, tal_acc: {}'.format(epoch+1, test_loss[-1], test_acc[-1]))
        for i in range(10):
            print('The acc of class {} : {} %'.format(classes[i], 100*eve_class_acc[-1][i]))

        torch.save(net.state_dict(), './results/baseline.pkl')
        np.savetxt('./results/train_loss.txt', train_loss)
        np.savetxt('./results/test_loss.txt', test_loss)
        np.savetxt('./results/train_acc.txt', train_acc)
        np.savetxt('./results/test_acc.txt', test_acc)
        np.savetxt('./results/test_acc.txt', test_acc)
        np.savetxt('./results/every_class_acc.txt', eve_class_acc)

    plt.figure()
    x = [i for i in range(EPOCH)]
    plt.plot(x, train_loss)
    plt.plot(x, test_loss)
    plt.legend(['train_loss', 'test_loss'])
    plt.savefig('./results/loss_baseline.png')

    plt.figure()
    plt.plot(x, train_acc)
    plt.plot(x, test_acc)
    plt.legend(['train_acc', 'test_acc'])
    plt.savefig('./results/acc_baseline.png')
    plt.show()

