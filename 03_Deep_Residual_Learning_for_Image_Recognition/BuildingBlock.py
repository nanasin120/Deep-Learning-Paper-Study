import torch 
import torch.nn as nn
import torch.nn.functional as F

# 이미지의 크기는 1024, 1024라 가정

class basicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(basicBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        # conv1을 통과하면 채널은 out_channel로 바뀌고 
        # 입력받은 stride가 1이라 가정하면
        # 크기는 (1024 + 2 * 1 - 3) / 1 + 1 = 1024로 유지됨
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, stride=stride, kernel_size=3, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        # conv1을 통과하면 채널은 out_channel로 유지되고 
        # 크기는 (1024 + 2 * 1 - 3) / 1 + 1 = 1024로 유지됨
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, stride=1, kernel_size=3, padding=1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        # shorcut connection을 위한 사전작업
        self.shortcutConnection = nn.Sequential()
        # 만약 stride가 1이 아니면 이미지의 크기가 유지가 안되고
        # 만약 in_channel과 out_channel이 같지 않으면 채널이 유지가 안되니
        # 아래 방식으로 바꿔줘야함
        if in_channel != out_channel or stride != 1:
            self.shortcutConnection = nn.Sequential(
                # stride가 1이라 가정하고
                # 채널은 out_channel로, 이미지 크기는 (1024 + 2*0 - 1)/1 + 1 = 1024
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, stride=stride, kernel_size=1, padding=0, bias = False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        # 나중에 더해줄 지름길을 미리 계산해줌
        identity = self.shortcutConnection(x)

        # 컨볼루션, 정규화, ReLU 적용
        y1 = self.conv1(x)
        y1 = self.bn1(y1)
        y1 = self.relu(y1)

        # 컨볼루션, 정규화 적용
        y2 = self.conv2(y1)
        y2 = self.bn2(y2)
        
        # 정규화후 나온 값 + 지름길을 해주고 ReLU 적용
        output = self.relu(y2 + identity)

        return output
    
# 이미지 크기는 1024,1024라 가정

class bottleNeckBlock(nn.Module):
    def __init__(self, in_channel, middle_channel, out_channel, stride=1):
        super(bottleNeckBlock, self).__init__()
        self.in_channel = in_channel
        self.middle_channel = middle_channel
        self.out_channel = out_channel
        
        # conv1을 통과하면 채널은 middle_channel로 바뀌고 
        # 크기는 (1024 + 2 * 0 - 1) / 1 + 1 = 1024로 유지됨
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=middle_channel, stride=1, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(middle_channel)
        
        # conv2을 통과하면 채널은 middle_channel로 유지되고 
        # stride가 1이라 가정하면
        # 크기는 (1024 + 2 * 1 - 3) / 1 + 1 = 1024로 유지됨
        self.conv2 = nn.Conv2d(in_channels=middle_channel, out_channels=middle_channel, stride=stride, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(middle_channel)

        # conv1을 통과하면 채널은 out_channel 바뀌고 
        # 크기는 (1024 + 2 * 0 - 1) / 1 + 1 = 1024로 유지됨
        self.conv3 = nn.Conv2d(in_channels=middle_channel, out_channels=out_channel, stride=1, kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        # shorcut connection을 위한 사전작업
        self.shortcutConnection = nn.Sequential()
        # 만약 stride가 1이 아니면 이미지의 크기가 유지가 안되고
        # 만약 in_channel과 out_channel이 같지 않으면 채널이 유지가 안되니
        # 아래 방식으로 바꿔줘야함
        if stride != 1 or in_channel != out_channel:
            self.shortcutConnection = nn.Sequential(
                # stride가 1이라 가정하고
                # 채널은 out_channel로, 이미지 크기는 (1024 + 2*0 - 1)/1 + 1 = 1024
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, stride=stride, kernel_size=1, padding=0, bias = False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        # 나중에 더해줄 지름길을 미리 계산해줌
        identity = self.shortcutConnection(x)

        # 컨볼루션, 정규화, ReLU 적용
        y1 = self.conv1(x)
        y1 = self.bn1(y1)
        y1 = self.relu(y1)
        
        # 컨볼루션, 정규화, ReLU 적용
        y2 = self.conv2(y1)
        y2 = self.bn2(y2)
        y2 = self.relu(y2)
        
        # 컨볼루션, 정규화 적용
        y3 = self.conv3(y2)
        y3 = self.bn3(y3)
        
        # 정규화후 나온 값 + 지름길을 해주고 ReLU 적용
        output = self.relu(y3 + identity)

        return output