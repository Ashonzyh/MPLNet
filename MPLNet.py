import torch
import torch.nn as nn
import torchvision.models as M
import torch.nn.functional as F




class CategoryAttentionBlock(nn.Module):
    def __init__(self, channel, classes=5, k=5, dropout_rate=0.5):
        super(CategoryAttentionBlock, self).__init__()
        self.classes = classes
        self.k = k
        self.conv1 = nn.Conv2d(channel, classes * k, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(classes*k)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        
    def forward(self, inputs):
        batch_size, channels, height, width = inputs.size()
        
        F = self.conv1(inputs)
        F = self.bn(F)
        F1 = self.relu(F)


        F2 = self.dropout(F1)
        x = self.maxpool(F2)
        x = x.view(batch_size, self.classes, self.k, 1, 1)
        S = torch.mean(x, dim=-3, keepdim=False)
        
        x = F1.view(batch_size, self.classes, self.k, height, width)
        x = torch.mean(x, dim=-3, keepdim=False)
        x = S * x
        M = torch.mean(x, dim=-3, keepdim=True)
        
        semantic = inputs * M
        
        return semantic


class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output


class MPLNet(nn.Module):
    def __init__(
        self, num_classes: int = 5, init_weights: bool = True, dropout: float = 0.5, adaloss: bool=True
    ) -> None:
        super().__init__()
        features = M.vgg16_bn(pretrained=True).features
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.cha = ChannelAttention(channel=512)
        self.sa = SpatialAttention()
        self.ca = CategoryAttentionBlock(channel=512)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        self.fc_clf = nn.Linear(512, 2)

        if adaloss:
            self.sigma1 = nn.Parameter(torch.zeros(1))
            self.sigma2 = nn.Parameter(torch.zeros(1))
            
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.features[:-11]:
            x = block(x)

        x1 = x*self.cha(x)
        x2 = x*self.sa(x)

        x = torch.mean(torch.stack([x1, x2], dim=-1), dim=-1)

        feat = x
        x_clf = F.adaptive_avg_pool2d(feat, (1,1))
        x_clf = torch.flatten(x_clf, 1)
        out_clf = self.fc_clf(x_clf)

        for block in self.features[-11:]:
            x = block(x)
        
        x1 = x*self.cha(x)
        x2 = x*self.sa(x)
        x3 = self.ca(x)

        x = torch.mean(torch.stack([x1, x2, x3], dim=-1), dim=-1)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return out_clf,x
