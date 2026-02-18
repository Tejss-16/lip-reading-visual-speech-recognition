# model.py
import torch
import torch.nn as nn
import torchvision.models as models

class LipNetSimple(nn.Module):
    def __init__(self, num_chars, cnn_out=512, hidden_size=512, num_layers=2):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        # remove final fc
        modules = list(resnet.children())[:-1]
        self.cnn = nn.Sequential(*modules)     # input: (B,3,H,W) -> output (B,512,1,1)
        self.cnn_out = cnn_out
        self.rnn = nn.LSTM(input_size=cnn_out, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_size*2, num_chars+1)  # +1 for CTC blank

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feat = self.cnn(x)            # (B*T, 512, 1,1)
        feat = feat.view(B, T, -1)    # (B, T, 512)
        out, _ = self.rnn(feat)       # (B, T, 2*hidden)
        logits = self.classifier(out) # (B, T, num_chars+1)
        # CTC expects (T, B, C)
        logits = logits.permute(1,0,2)
        return logits  # (T, B, C)

