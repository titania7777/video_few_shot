import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnet18
from torchvision.models.video import r2plus1d_18
# torch.backends.cudnn.enabled = False

class Model(nn.Module):
    def __init__(self, way=5, shot=1, query=15, hidden_size=1024, num_layers=1, bidirectional=True, attention=True):
        super(Model, self).__init__()
        self.way = way
        self.shot = shot
        self.query = query

        # Encoder(resnet18)
        # model = resnet18(pretrained=True)
        # self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        # Encoder(r2plus1d_18)
        model = r2plus1d_18(pretrained=True)
        self.encoder = nn.Sequential(
            model.stem,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            nn.AdaptiveAvgPool3d((None, 1, 1))
        )
        # Encoder(freeze)
        # self._freeze(self.encoder)
        
        # Encoder Linear
        # self.encoder_linear = nn.Sequential(
            # nn.Linear(model.fc.in_features, model.fc.in_features),
            # nn.BatchNorm1d(model.fc.in_features),
            # nn.ReLU(),
        # )
        # self.encoder_linear.apply(self._initialize)

        self.scaler = nn.Parameter(torch.tensor(5.0))

        # LSTM
        self.lstm = nn.LSTM(model.fc.in_features, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        
        # Linear
        self.linear = nn.Linear(2*hidden_size if bidirectional else hidden_size, hidden_size if bidirectional else int(hidden_size/2))
        self.linear.apply(self._initialize)

        # linear attention
        # self.use_attention = attention
        # if self.use_attention:
            # self.attention = nn.Linear(2 * hidden_size if bidirectional else hidden_size, 1)
            # self.attention.apply(self._initialize)

    def forward(self, shot, query):
        x = torch.cat((shot, query), dim=0)
        b, d, c, h, w = x.shape
        x = x.transpose(1, 2).contiguous() # b, c, d, h, w
        x = self.encoder(x).squeeze()
        x = x.transpose(1, 2).contiguous() # b, d, c
        x = (self.lstm(x)[0]).mean(1)
        # if self.use_attention:
        #     attention = F.softmax(self.attention(x).squeeze(-1), dim=-1)
        #     x = torch.sum(attention.unsqueeze(-1) * x, dim=1)
        # else:
        #     x = x[:, -1]# last hidden
        x = self.linear(x)

        shot, query = x[:shot.size(0)], x[shot.size(0):]
        # make prototype
        shot = shot.reshape(self.shot, self.way, -1).mean(dim=0)
        shot = F.normalize(shot, dim=-1)
        query = F.normalize(query, dim=-1)
        # cosine similarity
        logits = torch.mm(query, shot.t())
        return logits * self.scaler
    
    def _freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False

    # only linear
    def _initialize(self, model):
        if type(model) == nn.Linear:
            nn.init.xavier_uniform_(model.weight)
            model.bias.data.fill_(0.01)