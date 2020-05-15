import torch.nn as nn

class GenreClassifier(nn.Module):
    def __init__(self):
        super(GenreClassifier, self).__init__()
        # 1st half
        self.conv1layer = nn.Conv1d(1, 128, (3, 128))
        self.maxpool1 = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU()

        # 2nd half
        self.conv2layer = nn.Conv1d(1, 64, (3, 64))
        self.maxpool2 = nn.MaxPool2d((105, 1))

        # Output
        self.fcl = nn.Linear(64, 5)
        self.dropout = nn.Dropout(p=0.25)
        self.softmax = nn.LogSoftmax(1)


    def forward(self, x):
        x = self.conv1layer(x)
        x = x.view(-1, 214, 128)
        x = self.relu(x)
        x = self.maxpool1(x).unsqueeze(1)

        x = self.conv2layer(x)
        x = x.view(-1, 105, 64)

        x = self.relu(x)
        x = self.maxpool2(x)
        x = x.view(-1, 64)

        x = self.dropout(x)
        x = self.fcl(x)
        x = self.softmax(x)
        return x
