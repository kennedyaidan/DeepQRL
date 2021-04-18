import matplotlib.pyplot as plt
import numpy as np
import collections
import cv2
import gym


# PLOT FUNCTION

def plot_learning_curve(x, scores, epsilons, filename):
    fig = plt.figure()
    ax1 = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax1.plot(x, epsilons, color="C0")
    ax1.set_xlabel("Training Steps", color="C0")
    ax1.set_ylabel("Epsilon", color="C0")
    ax1.tick_params(axis='x', colors="C0")
    ax1.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    plt.savefig(filename)

def plot_debug(x, debug, filename):
    fig = plt.figure()
    ax1 = fig.add_subplot(111, label="1")

    ax1.plot(x, debug, color="C0")
    ax1.set_xlabel("Training Steps", color="C0")
    ax1.set_ylabel("", color="C0")
    ax1.tick_params(axis='x', colors="C0")
    ax1.tick_params(axis='y', colors="C0")

    plt.savefig(filename)



# MAX FRAME CLASS

class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env=None, repeat=4, clip_reward=False, no_ops=0, fire_first=False):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape       #just obs shape (could be .high.shape as well)
        self.frame_buffer = np.zeros_like((2, self.shape)) 
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first

    def step(self, action):
        tot_reward = 0.0
        done = False
        for frame in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            if self.clip_reward:
                reward = np.clip(np.array([reward]), -1, 1)[0] #clipping reward between -1 and 1 as in Paper, making into numpy array
            tot_reward +=reward
            idx = frame % 2             #every other frame
            self.frame_buffer[idx] = obs # idx will be 0 or 1 alternating
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1]) #elementwise maximum of frames (merging frames together for atari)
        return max_frame, tot_reward, done, info

    def reset(self):
        obs = self.env.reset()
        no_ops = np.random.randint(self.no_ops)+1 if self.no_ops > 0 else 0 #generate random number of steps for no_operations
        for _ in range(no_ops):
            _, _, done, _  = self.env.step(0) #why zero, just any action cause it doesn't matter?
            if done:
                self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE' #will error if meaning is not FIRE
            obs, _, _, _ = self.env.step(1) #first action is fire (we need to know that ahead of time)
        self.frame_buffer = np.zeros_like((2,self.shape))
        self.frame_buffer[0] = obs
        return obs





# PREPROCESS FRAME CLASS

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self). __init__(env)
        self.shape = (shape[2], shape[0], shape[1]) #swapping channels
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32) #creating the observation_space

    def observation(self, obs):
        new_frame = cv2.cvtColor(obs,cv2.COLOR_RGB2GRAY) #converting from color to gray-scale
        resized_screen = cv2.resize(new_frame, self.shape[1:], interpolation=cv2.INTER_AREA) #resizing, don't quite understand
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)  #making into numpy array, swapping axis, of datatype and shape. DONT GET THIS
        new_obs = new_obs/255.0 #normalizing to be between 0 and 1
        return new_obs







# STACKING FRAME CLASS

class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(env.observation_space.low.repeat(repeat, axis=0),
                                                env.observation_space.high.repeat(repeat, axis=0),
                                                dtype=np.float32)
        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        obs = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(obs)
        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, obs):
        self.stack.append(obs)
        return np.array(self.stack).reshape(self.observation_space.low.shape)



# MAKE ENVIRONMENT DEFINITION

def make_env(env_name, shape=(84,84,1), repeat=4, clip_rewards=False, no_ops=0, fire_first=False):
    #stacking changes to the environment with each line
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    env = PreprocessFrame(shape, env)  
    env = StackFrames(env, repeat)

    return env


