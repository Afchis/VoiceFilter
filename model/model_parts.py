import torch
import torch.nn as nn

class VoiceEncoder(nn.Module):
	def __init__(self):
		super(VoiceEncoder, self).__init__()