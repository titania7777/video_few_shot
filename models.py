import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from torchvision.models.video import r2plus1d_18
# torch.backends.cudnn.enabled = False

class R2Plus1D(nn.Module):
    def __init__(self, way=5, shot=1, query=1):
        super(R2Plus1D, self).__init__()
        self.way = way
        self.shot = shot
        self.query = query

        # encoder(r2plus1d_18)
        self.encoder = r2plus1d_18(pretrained=True)

        # scaler
        self.scaler = nn.Parameter(torch.tensor(5.0))

    def forward(self, shot, query):
        x = torch.cat((shot, query), dim=0)
        b, d, c, h, w = x.shape

        x = x.transpose(1, 2).contiguous() # b, c, d, h, w
        x = self.encoder(x).squeeze()

        # calculate similarity =================
        shot, query = x[:shot.size(0)], x[shot.size(0):]
        # make prototype
        shot = shot.reshape(self.shot, self.way, -1).mean(dim=0)
        shot = F.normalize(shot, dim=-1)
        query = F.normalize(query, dim=-1)
        # cosine similarity
        logits = torch.mm(query, shot.t())
        # calculate similarity =================

        return logits * self.scaler

class Resnet(nn.Module):
    def __init__(self, way=5, shot=1, query=1, hidden_size=1024, num_layers=1, bidirectional=True):
        super(Resnet, self).__init__()
        self.way = way
        self.shot = shot
        self.query = query

        # encoder(resnet18)
        model = resnet18(pretrained=True)
        self.encoder = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
        )
        
        # lstm
        self.lstm = nn.LSTM(model.fc.in_features, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)

        # scaler
        self.scaler = nn.Parameter(torch.tensor(5.0))

    def forward(self, shot, query):
        x = torch.cat((shot, query), dim=0)
        b, d, c, h, w = x.shape

        # pre trained feature extractor =================
        x = x.view(b * d, c, h, w)
        x = self.encoder(x).squeeze()
        # pre trained feature extractor =================

        # additional layers =================
        # lstm layer
        x = x.view(b, d, 512)
        x = (self.lstm(x)[0]).mean(1)
        # # additional layers =================

        # calculate similarity =================
        shot, query = x[:shot.size(0)], x[shot.size(0):]
        # make prototype
        shot = shot.reshape(self.shot, self.way, -1).mean(dim=0)
        shot = F.normalize(shot, dim=-1)
        query = F.normalize(query, dim=-1)
        # cosine similarity
        logits = torch.mm(query, shot.t())
        # calculate similarity =================

        return logits * self.scaler