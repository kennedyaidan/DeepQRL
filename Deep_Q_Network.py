import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import numpy as np
import os

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, name, checkpoint_dir):
        super(DeepQNetwork, self).__init__()

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.c1 = nn.Conv2d(input_dims[0], 32, 8, stride=4) #first channel of input dimensions (size 1 for gray images), 32 filters of 8x8, with stride 4
        self.c2 = nn.Conv2d(32, 64, 4, stride=2)            #input of 32, 64 filters of 4x4 with stride 2
        self.c3 = nn.Conv2d(64, 64, 3, stride=1)            #input of 64 filters, output of 64 filters, 3x3 with stride 1

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)


        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss() #I think this is standard in DeepRL cause we don't know what's better yet

        #setting up gpu
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        #letting the convolutional layers dertermine the dimensions
        state = T.zeros(1, *input_dims)
        dims = self.c1(state)
        dims = self.c2(dims)
        dims = self.c3(dims)
        return int(np.prod(dims.size())) #flattening stack of 2-D images for input into fully connected linear network


    def forward(self, state):
        conv1 = F.relu(self.c1(state))
        conv2 = F.relu(self.c2(conv1))
        conv3 = F.relu(self.c3(conv2)) #shape of BatchSize x n_filters x height x width

        conv_state = conv3.view(conv3.size()[0], -1)
        flat1 = F.relu(self.fc1(conv_state))
        flat2 = self.fc2(flat1)
        actions = flat2

        return actions

    def save_checkpoint(self):
        print('...saving check point...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('...loading check point...')
        self.load_state_dict(T.load(self.checkpoint_file))