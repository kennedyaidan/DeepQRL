import numpy as np
import torch as T
from Deep_Q_Network import DeepQNetwork
from replay_memory import ReplayBuffer

class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, memory_size, 
                 batch_size, eps_min=0.01, eps_decr=5e-7, replace=1000,
                 algorithm=None, env_name=None, checkpoint_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.input_dims = input_dims
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_decr = eps_decr
        self.replace_target_counter = replace
        self.algorithm = algorithm
        self.env_name = env_name
        self.checkpoint_dir = checkpoint_dir
        self.learn_step_counter = 0

        #creating the agents memory
        self.memory = ReplayBuffer(self.memory_size, self.input_dims, self.n_actions)

        #creating the Deep Q Network for Evaluation
        self.Q_NN_eval = DeepQNetwork(lr=self.lr, n_actions=self.n_actions, input_dims=self.input_dims,
                                      name=self.env_name+'_'+self.algorithm+'_'+'q_eval',
                                      checkpoint_dir=self.checkpoint_dir)


        #creating the Deep Q Network for the next state
        self.Q_NN_next = DeepQNetwork(lr=self.lr, n_actions=self.n_actions, input_dims=self.input_dims,
                                      name=self.env_name+'_'+self.algorithm+'_'+'q_next',
                                      checkpoint_dir=self.checkpoint_dir)

    def choose_action(self, state):
        #random action
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        #greedy action (using eval network)
        else:
           state = T.tensor([state], dtype=T.float).to(self.Q_NN_eval.device) #making the state into a tensor before passing into NN
           actions = self.Q_NN_eval.forward(state) #progating the state forward to get the action values
           action = T.argmax(actions).item()  #taking the max value action with T.argmax.item
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_decr if self.epsilon > self.eps_min \
                                      else self.eps_min
    


    def train(self):
        #only training after we collect enough memories
        if self.memory.memory_counter < self.batch_size:
            return
        #zeroing gradients
        self.Q_NN_eval.optimizer.zero_grad() #always zero gradient before trainig
        self.replace_target_network()        #replacing target network (will only replace according to specified rate)
        #calling sample memory
        states, actions, rewards, next_states, dones = self.sample_memory()
        #stepping forward the networks
        idx_BS = np.arange(self.batch_size) #to get index for batch sizes
        current_qs = self.Q_NN_eval.forward(states)[idx_BS, actions]   #current q for each batch (using Q Eval)
      
        #decoupled current and next Deep-Q networks for stability as outlined in paper
        future_qs_next = self.Q_NN_next.forward(next_states)      #future q for each batch (using Q Next)
        future_qs_next[dones] = 0                                 #setting all indices where done flag is true is zero (terminal states)
        future_qs_eval = self.Q_NN_eval.forward(next_states)      #future q for each batch (using Q Eval)
        max_next_actions = T.argmax(future_qs_eval, dim=1)        #finding max future actions from the eval network
        max_future_qs = future_qs_next[idx_BS, max_next_actions]  #the maximum future action as calculated from the next network

        q_targets = rewards + self.gamma*max_future_qs
        loss = self.Q_NN_eval.loss(q_targets, current_qs).to(self.Q_NN_eval.device) #difference between target and the actual (TD learning as before)
        #backpropogating
        loss.backward() #from the loss which we calucalted, we now calculate the gradient
        self.Q_NN_eval.optimizer.step() #now we backprogate and upate all the wieghts with the calculated gradient
        #increment learn step counter
        self.learn_step_counter += 1
        #decrement epsilon
        self.decrement_epsilon()





    def store_trans_memory(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def sample_memory(self):
        states, actions, rewards, next_states, dones = self.memory.sample_memory(self.batch_size)

        states = T.tensor(states).to(self.Q_NN_eval.device)
        actions = T.tensor(actions).to(self.Q_NN_eval.device)
        rewards = T.tensor(rewards).to(self.Q_NN_eval.device)
        next_states = T.tensor(next_states).to(self.Q_NN_eval.device)
        dones = T.tensor(dones).to(self.Q_NN_eval.device)

        return states, actions, rewards, next_states, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_counter == 0:
            self.Q_NN_next.load_state_dict(self.Q_NN_eval.state_dict())

    def save_models(self):
        self.Q_NN_eval.save_checkpoint()
        self.Q_NN_next.save_checkpoint()

    def load_models(self):
        self.Q_NN_eval.load_checkpoints()
        self.Q_NN_next.load_checkpoints()