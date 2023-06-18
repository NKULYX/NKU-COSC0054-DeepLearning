import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):  # SE block
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x
    
class SEBasicBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(SEBasicBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True), #inplace=True表示进行原地操作，一般默认为False，表示新建一个变量存储操作
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        #论文中模型架构的虚线部分，需要下采样
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        self.se = SEModule(outchannel)
 
    def forward(self, x):
        out = self.left(x) #这是由于残差块需要保留原始输入
        out += self.shortcut(x)#这是ResNet的核心，在输出上叠加了输入x
        out = F.relu(out)
        out = self.se(out)
        return out

class SE_ResNet_18(nn.Module):
    def __init__(self, num_classes=10):
        super(SE_ResNet_18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        self.layer4 = self.make_layer(512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)
 
    def make_layer(self, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(SEBasicBlock(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
 
    def forward(self, x):  # 3*32*32
        out = self.conv1(x)  # 64*32*32
        out = self.layer1(out)  # 64*32*32
        out = self.layer2(out)  # 128*16*16
        out = self.layer3(out)  # 256*8*8
        out = self.layer4(out)  # 512*4*4
        out = F.avg_pool2d(out, 4)  # 512*1*1
        out = out.view(out.size(0), -1)  # 512
        out = self.fc(out)
        return out
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    se_resnet = SE_ResNet_18().to(device)
    test_input = torch.rand(1, 3, 32, 32).to(device)
    test_output = se_resnet(test_input)
    print(test_output.shape)