import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.optim as optim
from network import *
from data_download import dataloader
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(net):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(5):  # loop over the dataset multiple times
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

        mean_accuracy = total_label_accuracy/n_batch
        loss = running_loss/n_batch
            # if i % 2000 == 1999:    # print every 2000 mini-batches
        print(f'loss: {loss:.3f},'
                   f'accuracy:{mean_accuracy:.4f},'f'epoch:{epoch}')

    print('Finished Training')

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

if __name__ == '__main__':
    trainloader, testloader, classes,_ = dataloader()
    net = Net().to(device)
    train(net)