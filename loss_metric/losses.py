import torch
import torch.nn as nn
import torch.nn.functional as F


class GE2ELoss(nn.Module):
    def __init__(self):
        super(GE2ELoss, self).__init__()
        self.weight = nn.Parameter(torch.tensor([10.]))
        self.bias = nn.Parameter(torch.tensor([-5.]))
        self.weight.requires_grad = True
        self.bias.requires_grad = True
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
            pos_sim_speaker = self.weight * F.cosine_similarity(dvec[sp_idx], pos_centroids[sp_idx], dim=1, eps=1e-6) + self.bias # [utterance_idx]
            neg_sim_speaker = list()
            for utt_idx in range(dvec.size(1)):
                neg_sim_utterance = self.weight * F.cosine_similarity(dvec[sp_idx, utt_idx].unsqueeze(0), neg_cintroids[sp_idx], dim=1, eps=1e-6) + self.bias # [speaker_idx-1]
                neg_sim_speaker.append(neg_sim_utterance)
            neg_sim_speaker = torch.stack(neg_sim_speaker, dim=0) # [utterance_idx, speaker_idx-1]
            pos_sim.append(pos_sim_speaker)
            neg_sim.append(neg_sim_speaker)
        pos_sim = torch.stack(pos_sim, dim=0) # [speaker_idx, utterance_idx]
        neg_sim = torch.stack(neg_sim, dim=0) # [speaker_idx, utterance_idx, speaker_idx-1]
        return pos_sim, neg_sim

    def _contrast_loss(self, pos_sim, neg_sim):
        loss =  1 - self.sigmoid(pos_sim) + self.sigmoid(neg_sim.max(2)[0])
        return loss.sum()

    def forward(self, dvec):
        pos_centroids = self._pos_centroids(dvec)
        neg_cintroids = self._neg_centroids(dvec)
        pos_sim, neg_sim = self._sim_matrix(dvec, pos_centroids, neg_cintroids)
        loss = self._contrast_loss(pos_sim, neg_sim)
        return loss


if __name__ == "__main__":
    dvec = torch.rand([64, 16, 256])
    model = GE2ELoss()
    loss = model(dvec)
    print(loss)