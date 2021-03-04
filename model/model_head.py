import torch
import torch.nn as nn


class VoiceFilter(nn.Module):
	def __init__(self):
		super(VoiceFilter, self).__init__()
