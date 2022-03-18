import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.optim as optim
from network import *
from ResNet import *
from VGG import *
from LeNet import *
from vgg_OFFICIAL import *
from resnet_official import *
from data_download import dataloader
from tqdm import tqdm
import warnings
from test import *

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def trains(net):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(15): # loop over the dataset multiple times
        net.train()
        total_label_accuracy = 0
        running_loss = 0.0
        n_batch = len(trainloader)
        for i, data in tqdm(enumerate(trainloader, 0),leave=False,total=n_batch,ncols=80):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            total_label_accuracy += (outputs.max(1)[1] == labels).float().mean().item()

        if epoch % 5 == 0:
            net.eval()
            _, testloader, _, _ = dataloader()
            correct = 0
            total = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    # calculate outputs by running images through the network
                    outputs = net(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
        mean_accuracy = total_label_accuracy/n_batch
        loss = running_loss/n_batch
            # if i % 2000 == 1999:    # print every 2000 mini-batches
        print(f'loss: {loss:.3f},'
                   f'accuracy:{mean_accuracy:.4f},'f'epoch:{epoch}')

    print('Finished Training')

    PATH = './resnet.pth'
    torch.save(net.state_dict(), PATH)

if __name__ == '__main__':
    trainloader, testloader, classes,_ = dataloader()
    net = ResNet18().to(device)
    total_num = sum(p.numel() for p in net.parameters())
    print("total paramater:",total_num)
    trains(net)