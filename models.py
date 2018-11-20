import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
from utils import norm_col_init, weights_init

import pdb


def em_initialize(em, img_seq_feat, steps):
    if em is None:
        em = [img_seq_feat.data] * 8
    elif steps < 8:
        for i in range(8 - steps):
            em[steps + i] = img_seq_feat.data
    return em

def em_add(em, img_feat):
    em.append(img_feat.data)
    if len(em) > 8:
        em = em[1:9]
    return em

def em_add_last(em, img_feat):
    em[7] = img_feat.data
    return em

class A3C_LSTM_GA(torch.nn.Module):
    def __init__(self):
        super(A3C_LSTM_GA, self).__init__()
        ## convolution network
        self.conv1 = nn.Conv2d(4, 64, kernel_size=5, stride=2)
        self.batchnorm1 = nn.BatchNorm2d(64, track_running_stats=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(64, track_running_stats=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.batchnorm3 = nn.BatchNorm2d(128, track_running_stats=False)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=5, stride=2)
        self.batchnorm4 = nn.BatchNorm2d(128, track_running_stats=False)

        self.fc = nn.Linear(1024, 256)

        self.img_lstm = nn.LSTMCell(256, 256)

        # Instruction Processing, MLP
        self.embedding = nn.Embedding(5, 25)
        # self.embedding = nn.Linear(5, 25)
        self.target_att_linear = nn.Linear(25, 256)

        ## a3c-lstm network
        #self.lstm = nn.LSTMCell(512+256, 256)   #512

        self.em_mlp = nn.Linear(2048, 256)
        #self.prelu = nn.PReLU()
        #self.internal = nn.Linear(256, 1)
        self.gated = nn.Linear(512, 256)

        self.mlp = nn.Linear(512, 192)  #512

        self.mlp_policy = nn.Linear(128, 64)
        self.actor_linear = nn.Linear(64, 5)

        self.mlp_value = nn.Linear(64, 32) #64
        self.critic_linear = nn.Linear(32, 1)

        self.kl_maximized = 0.
        self.em_flag = False


    def forward(self, state, instruction_idx, hx, cx, steps, em, gpu_id, debugging=False):
        x = state

        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = F.relu(self.batchnorm4(self.conv4(x)))

        if debugging is True:
            pdb.set_trace()

        x = x.view(x.size(0), -1)
        img_feat = F.relu(self.fc(x))

        '''
            input something math that works divide memory units
            and
            if minimized with exploration, give more rewards  
            
            kl(p(x,y)||p(x)p(y)) => how can i find the joint distribution?       
        
        # dist1 = Normal(torch.ones(256), torch.ones(256))    # after batchnorm, mu=0,sigma=1
        if steps > 2:
            dist1 = Normal(img_feat.mean(dim=1), img_feat.var(dim=1))
            dist2 = Normal(hx.mean(dim=1), hx.var(dim=1))
            # is it enough that dim=1? use it first, encoding second and pca third
            # dist3 = Normal((img_feat.mean(dim=1)+img_seq_feat.mean(dim=1))/2., (img_seq_feat.var(dim=1)+img_seq_feat.var(dim=1))/2)
            # kl_divergence = kl.kl_divergence(dist3, dist1 * dist2)
            with torch.cuda.device(gpu_id):
                kl_t = kl.kl_divergence(dist1, dist2).cuda()
                if kl_t >= self.kl_maximized:   # add rewards
                    self.kl_maximized = kl_t
                    self.em = em_add(self.em, hx)
            if debugging is True:
                print(kl_t)
        '''

        img_seq_feat, cx = self.img_lstm(img_feat, (hx, cx))

        if steps == 0:
            em = None
            em = em_initialize(em, img_seq_feat, steps)  # 1. em test whether it works well. 2. learning

        dist1 = Normal(img_feat.mean(dim=1), img_feat.var(dim=1))
        dist2 = Normal(hx.mean(dim=1), hx.var(dim=1))
        # is it enough that dim=1? use it first, encoding second and pca third
        # dist3 = Normal((img_feat.mean(dim=1)+img_seq_feat.mean(dim=1))/2., (img_seq_feat.var(dim=1)+img_seq_feat.var(dim=1))/2)
        # kl_divergence = kl.kl_divergence(dist3, dist1 * dist2)
        with torch.cuda.device(gpu_id):
            kl_t = kl.kl_divergence(dist1, dist2).cuda()
            if steps % 10 == 0:
                self.kl_maximized = 0.
                self.em_flag = False

            if kl_t >= self.kl_maximized:  # add rewards
                self.kl_maximized = kl_t
                if self.em_flag is True:
                    em = em_add_last(em, hx)
                else:
                    em = em_add(em, hx)
                    self.em_flag = True

        em_tensor = torch.stack(em)
        em_tensor = em_tensor.view(-1, 2048)    #em_tensor.view(em_tensor.size(0), -1)

        em_output = F.relu(self.em_mlp(em_tensor))
        #internal_reward = F.hardtanh(self.internal(em_output))

        # Get the instruction representation
        word_embedding = self.embedding(instruction_idx)
        word_embedding = word_embedding.view(word_embedding.size(0), -1)

        ## calculate gated attention
        word_embedding = self.target_att_linear(word_embedding)
        gated_att = torch.sigmoid(word_embedding)

        ## apply gated attention
        gated_fusion = torch.mul(img_seq_feat, gated_att)
        mlp_input = torch.cat([gated_fusion, gated_att], 1)

        mlp_input = F.relu(self.gated(mlp_input))

        mlp_input = torch.cat([mlp_input, em_output], 1)

        #mlp_input = torch.cat([gated_fusion, _hx], 1)
        mlp_input = self.mlp(mlp_input)

        policy1, policy2, value = torch.chunk(mlp_input, 3, dim=1)

        policy = torch.cat([policy1, policy2], 1)
        policy = self.mlp_policy(policy)

        value = self.mlp_value(value)

        return self.critic_linear(value), self.actor_linear(policy), img_seq_feat, cx, em