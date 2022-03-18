import torch
import torch.nn as nn
import torchvision

def Conv3x3BNReLU(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

class VGGNet(nn.Module):
    def __init__(self, block_nums,num_classes=10):
        super(VGGNet, self).__init__()

        self.stage1 = self._make_layers(in_channels=3, out_channels=8, block_num=block_nums[0])
        self.stage2 = self._make_layers(in_channels=8, out_channels=16, block_num=block_nums[1])
        self.stage3 = self._make_layers(in_channels=16, out_channels=32, block_num=block_nums[2])
        self.stage4 = self._make_layers(in_channels=32, out_channels=64, block_num=block_nums[3])
        self.stage5 = self._make_layers(in_channels=64, out_channels=64, block_num=block_nums[4])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=64*7*7,out_features=64),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64, out_features=num_classes),
            nn.Softmax(dim=-1)
        )

    def _make_layers(self, in_channels, out_channels, block_num):
        layers = []
        layers.append(Conv3x3BNReLU(in_channels,out_channels))
        for i in range(1,block_num):
            layers.append(Conv3x3BNReLU(out_channels,out_channels))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        # print(x.size())
        x = x.view(x.size(0),-1)
        out = self.classifier(x)
        return out

def VGG16():
    block_nums = [2, 2, 3, 3, 3]
    model = VGGNet(block_nums)
    return model

def VGG19():
    block_nums = [2, 2, 4, 4, 4]
    model = VGGNet(block_nums)
    return model

if __name__ == '__main__':
    model = VGG16() #is VGG7(), the size is too small, we can not apply VGG16
    total_num = sum(p.numel() for p in model.parameters())
    # for p in model.parameters():
        # print(p.numel())
    print("total paramater:", total_num)
    input = torch.randn(1, 3, 32, 32)
    out = model(input)
    print(out.shape)