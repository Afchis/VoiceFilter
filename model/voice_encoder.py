import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeechEmbedder(nn.Module):
    def __init__(self):
        super(SpeechEmbedder, self).__init__()
        self.lstm = nn.LSTM(128,
                            768,
                            num_layers=3,
                            batch_first=True)
        self.proj = nn.Linear(768, 256)
        self.loss_w = nn.Parameter(torch.tensor([10.]))
        self.loss_b = nn.Parameter(torch.tensor([-5.]))
        self.loss_w.requires_grad = True
        self.loss_b.requires_grad = True
        self.sigmoid = nn.Sigmoid()

    def _pos_centroids(self, dvec):
        return torch.mean(dvec, dim=1).unsqueeze(1)

    def _neg_centroids(self, dvec):
        neg_list = list()
        for sp_idx in range(dvec.size(0)):
            neg_cents = torch.cat([dvec[:sp_idx], dvec[sp_idx+1:]], dim=0)
            neg_cents = torch.mean(neg_cents, dim=1)
            neg_list.append(neg_cents)
        return torch.stack(neg_list, dim=0)

    def _sim_matrix(self, dvec, pos_centroids, neg_cintroids):
        '''
        dvec.shape          --> [speaker_idx, utterance_idx, emb_dim]
        pos_centroids.shape --> [speaker_idx,       1      , emb_dim]
        neg_cintroids.shape --> [speaker_idx, speaker_idx-1, emb_dim]
        '''
        pos_sim = list()
        neg_sim = list()
        for sp_idx in range(dvec.size(0)):
            pos_sim_speaker = self.loss_w * F.cosine_similarity(dvec[sp_idx], pos_centroids[sp_idx], dim=1, eps=1e-6) + self.loss_b # [utterance_idx]
            neg_sim_speaker = list()
            for utt_idx in range(dvec.size(1)):
                neg_sim_utterance = self.loss_w * F.cosine_similarity(dvec[sp_idx, utt_idx].unsqueeze(0), neg_cintroids[sp_idx], dim=1, eps=1e-6) + self.loss_b # [speaker_idx-1]
                neg_sim_speaker.append(neg_sim_utterance)
            neg_sim_speaker = torch.stack(neg_sim_speaker, dim=0) # [utterance_idx, speaker_idx-1]
            pos_sim.append(pos_sim_speaker)
            neg_sim.append(neg_sim_speaker)
        pos_sim = torch.stack(pos_sim, dim=0) # [speaker_idx, utterance_idx]
        neg_sim = torch.stack(neg_sim, dim=0) # [speaker_idx, utterance_idx, speaker_idx-1]
        return pos_sim, neg_sim
 
    def _contrast_loss(self, pos_sim, neg_sim):
        print(neg_sim.shape)
        loss =  1 - self.sigmoid(pos_sim) + self.sigmoid(neg_sim).max(2)[0]
        # loss =  1 - pos_sim + neg_sim.max(2)[0]
        neg_sim = torch.tensor([0])
        return loss.mean(), pos_sim.min().item(), neg_sim.max().item()

    def _softmax_loss(self, pos_sim, neg_sim):
        loss = -pos_sim + torch.log(torch.exp(neg_sim).sum(dim=2))
        return loss.mean(), pos_sim.min().item(), neg_sim.max().item()

    def _ge2e_loss(self, dvec):
        dvec = dvec.reshape(8, 10, -1)
        pos_centroids = self._pos_centroids(dvec)
        neg_cintroids = self._neg_centroids(dvec)
        pos_sim, neg_sim = self._sim_matrix(dvec, pos_centroids, neg_cintroids)
        loss, pos_sim, neg_sim = self._softmax_loss(pos_sim, neg_sim)
        return loss, pos_sim, neg_sim

    def forward(self, mel):
        # (b, c, num_mels, T) b --> [speaker_idx*utterance_idx]
        mel = mel.squeeze() # (b, num_mels, T)   c == 1
        mel = mel.permute(0, 2, 1) # (b, T, num_mels)
        dvec, _ = self.lstm(mel) # (b, T, lstm_hidden)
        dvec = dvec[:, -1, :] # (b, lstm_hidden), use last frame only
        dvec = self.proj(dvec) # (b, emb_dim)
        dvec = dvec.div(dvec.norm(p=2, dim=-1, keepdim=True))
        if self.train:
            loss, pos_sim, neg_sim = self._ge2e_loss(dvec)
            return loss, pos_sim, neg_sim
        else:
            return dvec