import numpy as np
#CREATING THE AGENTS MEMORY

class ReplayBuffer():
	def __init__(self, max_size, input_shape, n_actions):
		self.memory_size = max_size #max memory size
		self.memory_counter = 0     #memory counter

		self.state_memory = np.zeros((self.memory_size, *input_shape), dtype=np.float32) #state memory of memory size by input shape, and float32 for compatibilty with pytorch
		self.new_state_memory = np.zeros((self.memory_size, *input_shape), dtype=np.float32) #state memory of memory size by input shape, and float32 for compatibilty with pytorch
		self.action_memory = np.zeros((self.memory_size), dtype=np.int64) #state memory of memory size by input shape, int64 for pytorch compatibility
		self.reward_memory = np.zeros((self.memory_size), dtype=np.float32) #state memory of memory size by input shape, and float32 
		self.terminal_state_memory = np.zeros(self.memory_size, dtype=np.bool) #unsigned int8, mask defined later

	def store_transition(self, state, action, reward, next_state, done):
		idx = self.memory_counter % self.memory_size #way to count up and reset at memory_size to count up again
		self.state_memory[idx] = state
		self.new_state_memory[idx] = next_state
		self.action_memory[idx] = action
		self.reward_memory[idx] = reward
		self.terminal_state_memory[idx] = done
		self.memory_counter += 1 

	def sample_memory(self, batch_size):
		max_memory = min(self.memory_counter, self.memory_size)
		batch = np.random.choice(max_memory, batch_size, replace=False) #choosing random episodes to make our batch

		states = self.state_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		next_states = self.new_state_memory[batch]
		dones = self.terminal_state_memory[batch]

		return states, actions, rewards, next_states, dones