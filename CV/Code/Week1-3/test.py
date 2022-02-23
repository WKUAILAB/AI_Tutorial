import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from data_download import dataloader
from network import *

path = 'cifar_net.pth'
def test(path):
    PATH = path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = Net().to(device)
    net.load_state_dict(torch.load(PATH))
    _,testloader,_,_ = dataloader()
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

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

def test_for_all(path):
    trainloader, testloader, classes, batch_size = dataloader()
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    PATH = path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = Net().to(device)
    net.load_state_dict(torch.load(PATH))
    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

if __name__ == '__main__':
    test()
    test_for_all()