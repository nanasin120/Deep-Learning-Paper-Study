import torch
import torch.nn as nn
import torch.nn.functional as F
from BuildingBlock import basicBlock, bottleNeckBlock

image = torch.randn(8, 64, 1024, 1024)
basicBlock = basicBlock(64, 32, 2)
bottleNeckBlock = bottleNeckBlock(64, 32, 32, 2)


print(f"입력 이미지의 크기 : {image.shape}")

basicBlock_image = basicBlock(image)
print(f"basicBlock 이후 이미지의 크기 : {basicBlock_image.shape}")

bottleNeckBlock_image = bottleNeckBlock(image)
print(f"bottleNeckBlock 이후 이미지의 크기 : {bottleNeckBlock_image.shape}")
