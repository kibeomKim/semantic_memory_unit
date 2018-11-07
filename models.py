import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import norm_col_init, weights_init

import pdb

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
        self.prelu = nn.PReLU()
        #self.internal = nn.Linear(256, 1)
        self.gated = nn.Linear(512, 256)

        self.mlp = nn.Linear(512, 192)  #512


        self.mlp_policy = nn.Linear(128, 64)
        self.actor_linear = nn.Linear(64, 5)

        self.mlp_value = nn.Linear(64, 32) #64
        self.critic_linear = nn.Linear(32, 1)

    def forward(self, state, instruction_idx, hx, cx, em, steps, debugging=False):
        x = state

        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = F.relu(self.batchnorm4(self.conv4(x)))

        if debugging is True:
            pdb.set_trace()

        x = x.view(x.size(0), -1)
        img_feat = F.relu(self.fc(x))

        img_seq_feat, cx = self.img_lstm(img_feat, (hx, cx))

        if em is None:
            em = [img_seq_feat.data] * 8
        elif steps < 8:
            for i in range(8-steps):
                em[steps+i] = img_seq_feat.data
        if steps % 20 == 0:
            em[0] = em[1]
            em[1] = em[2]
            em[2] = em[3]
            em[3] = em[4]
            em[4] = em[5]
            em[5] = em[6]
            em[6] = em[7]
            em[7] = img_seq_feat.data

        em_tensor = torch.stack(em)
        em_tensor = em_tensor.view(-1, 2048)    #em_tensor.view(em_tensor.size(0), -1)

        em_output = self.prelu(self.em_mlp(em_tensor))
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