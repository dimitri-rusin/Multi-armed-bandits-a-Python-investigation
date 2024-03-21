from plotly.subplots import make_subplots
import gymnasium
import inspectify
import numpy
import plotly.graph_objects as go



class MultiArmedBandit(gymnasium.Env):
  def __init__(self, number_levers):
    super(MultiArmedBandit, self).__init__()
    self.action_space = gymnasium.spaces.Discrete(number_levers)
    self.observation_space = gymnasium.spaces.Discrete(1)
    self.num_optimal_selected = None
    self.reward_means = None
    self.optimal_action = None

  def step(self, action):
    assert self.action_space.contains(action)

    if self.optimal_action == action:
      self.num_optimal_selected += 1

    # Reward is drawn from a normal distribution centered at the mean of the chosen arm
    reward = numpy.random.normal(self.reward_means[action], 1.0)

    return 0, reward, True, False, {}

  def reset(self):
    self.num_optimal_selected = 0
    self.reward_means = numpy.random.normal(0.0, 1.0, self.action_space.n)
    self.optimal_action = numpy.argmax(self.reward_means)
    assert self.action_space.contains(self.optimal_action)

    return 0, {}  # Dummy state

  def render(self, mode='human'):
    pass

  def close(self):
    pass



def bandit_random(env, average_reward_per_action):
  action_list = list(range(env.action_space.n))
  action = numpy.random.choice(action_list)
  return action

def greedy(env, average_reward_per_action):
  max_average = max(average_reward_per_action.values())  # Get the maximum average reward
  best_actions = [action for action, avg_reward in average_reward_per_action.items() if avg_reward == max_average]
  return numpy.random.choice(best_actions)



def experiment(num_runs, num_timesteps, number_levers, seed, algorithm):
  numpy.random.seed(seed)
  env = MultiArmedBandit(number_levers = number_levers)
  rewards = numpy.zeros(num_timesteps)
  num_optimal_selected = numpy.zeros(num_timesteps)

  for run in range(num_runs):
    total_reward = 0
    average_reward_per_action = {action:0 for action in range(env.action_space.n)}
    state = env.reset()
    for timestep in range(1, num_timesteps + 1):

      action = algorithm(env, average_reward_per_action)
      _, reward, _, _, _ = env.step(action)

      average_reward_per_action[action] += (1 / timestep) * (reward - average_reward_per_action[action])

      rewards[timestep - 1] += reward
      num_optimal_selected[timestep - 1] += env.num_optimal_selected / timestep

  average_rewards = rewards / num_runs
  average_num_optimal_selected = (num_optimal_selected / num_runs) * 100

  return average_rewards, average_num_optimal_selected



if __name__ == '__main__':

  random_average_rewards, random_average_num_optimal_selected = experiment(
    num_runs = 200,
    num_timesteps = 100,
    number_levers = 10,
    seed = 42,
    algorithm = bandit_random,
  )

  # Create a subplot figure
  fig = make_subplots(rows=2, cols=1, subplot_titles=('RANDOM - Average Cumulative Rewards', 'RANDOM - % Optimal Action Selected'))

  # Add the first plot for Average Cumulative Rewards
  fig.add_trace(go.Scatter(x=list(range(num_timesteps)), y=random_average_rewards, mode='lines', name='random'), row=1, col=1)

  # Add the second plot for Percentage of Optimal Action Selected
  fig.add_trace(go.Scatter(x=list(range(num_timesteps)), y=random_average_num_optimal_selected, mode='lines', name='random'), row=2, col=1)

  # Update layout to fill the browser tab's real estate
  fig.update_layout(
      height=900,  # Adjust the height to fit your screen
      width=1200,  # Adjust the width to fit your screen
      title_text="Multi-Armed Bandit Experiment Results",
      margin=dict(l=20, r=20, t=50, b=20)  # Reduce margins to use more space
  )

  # Show the figure
  fig.show()



  greedy_average_rewards, greedy_average_num_optimal_selected = experiment(
    num_runs = 200,
    num_timesteps = 100,
    number_levers = 10,
    seed = 42,
    algorithm = greedy,
  )

  # Create a subplot figure
  fig = make_subplots(rows=2, cols=1, subplot_titles=('GREEDY - Average Cumulative Rewards', 'GREEDY - % Optimal Action Selected'))

  # Add the first plot for Average Cumulative Rewards
  fig.add_trace(go.Scatter(x=list(range(num_timesteps)), y=greedy_average_rewards, mode='lines', name='random'), row=1, col=1)

  # Add the second plot for Percentage of Optimal Action Selected
  fig.add_trace(go.Scatter(x=list(range(num_timesteps)), y=greedy_average_num_optimal_selected, mode='lines', name='random'), row=2, col=1)

  # Update layout to fill the browser tab's real estate
  fig.update_layout(
      height=900,  # Adjust the height to fit your screen
      width=1200,  # Adjust the width to fit your screen
      title_text="Multi-Armed Bandit Experiment Results",
      margin=dict(l=20, r=20, t=50, b=20)  # Reduce margins to use more space
  )

  # Show the figure
  fig.show()
