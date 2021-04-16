import numpy as np
from Agent import Agent
from utils import make_env, plot_learning_curve, plot_debug

#np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning) 

if __name__ == '__main__':
    env_name = 'PongNoFrameskip-v4'
    env = make_env(env_name)
    best_score = -np.inf #will set first score to save
    load_checkpoint = False
    n_games = 500
    render = False
    debug = True
    agent  = Agent(gamma=0.99, epsilon=1.0, lr=1e-4,
                  n_actions=env.action_space.n, 
                  input_dims=env.observation_space.shape,
                  memory_size=50000,
                  batch_size=32, replace=1000, eps_decr=1e-5,
                  checkpoint_dir='models/', algorithm='DQNAgent',
                  env_name=env_name) #careful of memory size (50,000) depending on RAM (this should be about 12Gb of RAM)
    if load_checkpoint:
        agent.load_models()

    file_name = agent.algorithm +'-'+ agent.env_name + '_' + 'lr' + str(agent.lr) + '_' + \
                '_' + str(n_games) + 'games'
    figure_file = 'plots/' + file_name + '.png'
    figure_name_actions = 'debug/' + file_name + '_actions' + '.png'
    figure_name_rewards = 'debug/' + file_name + '_rewards' + '.png'

    n_steps = 0
    scores = []
    eps_history = []
    steps_array = []

    for i in range(n_games):
        done = False
        score = 0
        state = env.reset()
        steps_ep = 0
        actions_debug = []
        steps_ep_debug = []
        rewards_debug = []
        while not done:
            steps_ep += 1
            steps_ep_debug.append(steps_ep)
            action = agent.choose_action(state)
            if debug:
                actions_debug.append(action)
            if render:
                env.render()
            next_state, reward, done, info = env.step(action)
            score += reward
            if debug:
                rewards_debug.append(reward)

            if not load_checkpoint:
                agent.store_trans_memory(state, action, reward, next_state, int(done))
                agent.train()
            state = next_state
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score: ', score,
              'average score %.1f best score %.1f epsilon %.2f' %
              (avg_score, best_score, agent.epsilon),
              'steps ', n_steps)

        if debug:
            plot_debug(steps_ep_debug, actions_debug, figure_name_actions)
            plot_debug(steps_ep_debug, rewards_debug, figure_name_rewards)

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    plot_learning_curve(steps_array, scores, eps_history, figure_file)



