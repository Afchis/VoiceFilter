import torch
import torch.nn as nn

from .model_parts import ConvReluBlock1d, LinReluBlock
from .voice_encoder import SpeechEmbedder


class VoiceFilter(nn.Module):
    def __init__(self, RE_weights="./weights/voice_encoder/embedder.pt"):
        super(VoiceFilter, self).__init__()
        self.refer_encoder = SpeechEmbedder()
        self.refer_encoder.load_state_dict(torch.load(RE_weights))
        self.cnn = nn.Sequential(
            ConvReluBlock1d(in_ch=1, out_ch=64, kernel_size=7, dilation=1),
            ConvReluBlock1d(in_ch=64, out_ch=64, kernel_size=1, dilation=1),
            ConvReluBlock1d(in_ch=64, out_ch=64, kernel_size=5, dilation=1),
            ConvReluBlock1d(in_ch=64, out_ch=64, kernel_size=5, dilation=1),
            ConvReluBlock1d(in_ch=64, out_ch=64, kernel_size=5, dilation=1),
            ConvReluBlock1d(in_ch=64, out_ch=64, kernel_size=5, dilation=1),
            ConvReluBlock1d(in_ch=64, out_ch=64, kernel_size=5, dilation=1),
            ConvReluBlock1d(in_ch=64, out_ch=8, kernel_size=1, dilation=1)
            )
        self.lstm = nn.LSTM(input_size=(8*501)+256, hidden_size=400)
        self.fc = LinReluBlock(in_ch=400, out_ch=501)

    def forward(self, refer_spec, noicy_spec):
        out_spec = noicy_spec
        b, time = noicy_spec.size(0), noicy_spec.size(3)
        # refer_spec, noicy_spec --> [b, c, mel, time], [b, c, mel, time]
        with torch.no_grad():
            d_vec = list()
            for batch in range(refer_spec.size(0)):
                d_vec.append(self.refer_encoder(refer_spec[batch]))
            d_vec = torch.stack(d_vec, dim=0) # [b, c]
        noicy_spec = noicy_spec.permute(0, 3, 1, 2)
        noicy_spec = noicy_spec.reshape(-1, noicy_spec.size(2), noicy_spec.size(3)) # [b*time, c, mel]
        out = self.cnn(noicy_spec) # [b*time, c, mel]
        out = out.reshape(b, time, -1) # [b, time, c*mel]
        with torch.no_grad():
            d_vec = d_vec.unsqueeze(1).expand(d_vec.size(0), out.size(1), d_vec.size(1))
        out = torch.cat([out, d_vec], dim=2)
        out, _ = self.lstm(out)
        out = self.fc(out).permute(0, 2, 1).unsqueeze(1)
        out = out * out_spec
        return out
