import torch.nn as nn
import torch.nn.functional as F


class ReIDModel(nn.Module):

    def __init__(self, width, num_classes):
        super(ReIDModel, self).__init__()
        self.fc1 = nn.LazyLinear(out_features=width, bias=True)
        self.fc2 = nn.Linear(in_features=width, out_features=width, bias=False)
        self.cls = nn.Linear(in_features=width,
                             out_features=num_classes,
                             bias=False)

    def forward(self, x):
        trans = F.relu(self.fc1(x))
        trans = F.relu(self.fc2(trans))
        return self.cls(trans)
