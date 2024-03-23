import gymnasium
import inspectify
import numpy
import plotly.graph_objects
import plotly.subplots

class MultiArmedBandit(gymnasium.Env):
  def __init__(self, num_levers):
    super(MultiArmedBandit, self).__init__()
    self.action_space = gymnasium.spaces.Discrete(num_levers)
    self.observation_space = gymnasium.spaces.Discrete(1)
    self.num_optimal_selected = None
    self.reward_means = None
    self.optimal_action = None

  def reset(self):
    self.num_optimal_selected = 0
    self.reward_means = numpy.random.normal(0.0, 1.0, self.action_space.n)
    self.optimal_action = numpy.argmax(self.reward_means)
    assert self.action_space.contains(self.optimal_action)

    return 0, {}  # Dummy state

  def step(self, action):
    assert self.action_space.contains(action)

    if self.optimal_action == action:
      self.num_optimal_selected += 1

    # Reward is drawn from a normal distribution centered at the mean of the chosen arm
    reward = numpy.random.normal(self.reward_means[action], 1.0)

    return 0, reward, True, False, {}

def bandit_random(env, average_reward_per_action):
  action_list = list(range(env.action_space.n))
  action = numpy.random.choice(action_list)
  return action

def epsilon_greedy(env, average_reward_per_action, epsilon=0):
  if numpy.random.random() < epsilon:
    # Exploration: randomly choose any action
    return bandit_random(env, average_reward_per_action)
  else:
    # Exploitation: choose the best action based on average rewards
    max_average = max(average_reward_per_action.values())
    best_actions = [action for action, avg_reward in average_reward_per_action.items() if avg_reward == max_average]
    return numpy.random.choice(best_actions)

def experiment(num_runs, num_timesteps, num_levers, seed, algorithm):

  numpy.random.seed(seed)

  env = MultiArmedBandit(num_levers = num_levers)
  rewards = numpy.zeros(num_timesteps)
  num_optimal_selected = numpy.zeros(num_timesteps)

  for run in range(num_runs):
    average_reward_per_action = {action:0 for action in range(env.action_space.n)}
    action_visit_count = {action:0 for action in range(env.action_space.n)}
    state = env.reset()
    for timestep in range(1, num_timesteps + 1):

      action = algorithm(env, average_reward_per_action)
      _, reward, _, _, _ = env.step(action)

      action_visit_count[action] += 1
      average_reward_per_action[action] += (1 / action_visit_count[action]) * (reward - average_reward_per_action[action])

      rewards[timestep - 1] += reward
      num_optimal_selected[timestep - 1] += env.num_optimal_selected / timestep

  average_rewards = rewards / num_runs
  average_num_optimal_selected = (num_optimal_selected / num_runs) * 100

  return average_rewards, average_num_optimal_selected

if __name__ == '__main__':
  num_timesteps = 1_000
  num_runs = 2_000
  num_levers = 10
  seed = 42

  random_average_rewards, random_average_num_optimal_selected = experiment(
    num_runs = num_runs,
    num_timesteps = num_timesteps,
    num_levers = num_levers,
    seed = seed,
    algorithm = bandit_random,
  )

  greedy_average_rewards, greedy_average_num_optimal_selected = experiment(
    num_runs = num_runs,
    num_timesteps = num_timesteps,
    num_levers = num_levers,
    seed = seed,
    algorithm = lambda env, average_reward_per_action: epsilon_greedy(env, average_reward_per_action, epsilon = 0),
  )

  greedy_epsilon_0_01_average_rewards, greedy_epsilon_0_01_average_num_optimal_selected = experiment(
    num_runs = num_runs,
    num_timesteps = num_timesteps,
    num_levers = num_levers,
    seed = seed,
    algorithm = lambda env, average_reward_per_action: epsilon_greedy(env, average_reward_per_action, epsilon = 0.01),
  )

  greedy_epsilon_0_1_average_rewards, greedy_epsilon_0_1_average_num_optimal_selected = experiment(
    num_runs = num_runs,
    num_timesteps = num_timesteps,
    num_levers = num_levers,
    seed = seed,
    algorithm = lambda env, average_reward_per_action: epsilon_greedy(env, average_reward_per_action, epsilon = 0.1),
  )

  # Create a subplot figure with 2 rows and 1 column
  fig = plotly.subplots.make_subplots(rows=2, cols=1, subplot_titles=(
    'Average Cumulative Rewards',
    '% Optimal Action Selected'))

  # Define colors for each strategy
  colors = {
    'Random': 'orange',
    'Greedy': 'green',
    'Epsilon 0.01': 'red',
    'Epsilon 0.1': 'blue'
  }

  # Add plots for Average Cumulative Rewards
  fig.add_trace(plotly.graph_objects.Scatter(x=list(range(num_timesteps)), y=random_average_rewards, mode='lines', name='Random', line=dict(color=colors['Random'])), row=1, col=1)
  fig.add_trace(plotly.graph_objects.Scatter(x=list(range(num_timesteps)), y=greedy_average_rewards, mode='lines', name='Greedy', line=dict(color=colors['Greedy'])), row=1, col=1)
  fig.add_trace(plotly.graph_objects.Scatter(x=list(range(num_timesteps)), y=greedy_epsilon_0_01_average_rewards, mode='lines', name='Epsilon 0.01', line=dict(color=colors['Epsilon 0.01'])), row=1, col=1)
  fig.add_trace(plotly.graph_objects.Scatter(x=list(range(num_timesteps)), y=greedy_epsilon_0_1_average_rewards, mode='lines', name='Epsilon 0.1', line=dict(color=colors['Epsilon 0.1'])), row=1, col=1)

  # Add plots for % Optimal Action Selected
  fig.add_trace(plotly.graph_objects.Scatter(x=list(range(num_timesteps)), y=random_average_num_optimal_selected, mode='lines', name='Random', line=dict(color=colors['Random'])), row=2, col=1)
  fig.add_trace(plotly.graph_objects.Scatter(x=list(range(num_timesteps)), y=greedy_average_num_optimal_selected, mode='lines', name='Greedy', line=dict(color=colors['Greedy'])), row=2, col=1)
  fig.add_trace(plotly.graph_objects.Scatter(x=list(range(num_timesteps)), y=greedy_epsilon_0_01_average_num_optimal_selected, mode='lines', name='Epsilon 0.01', line=dict(color=colors['Epsilon 0.01'])), row=2, col=1)
  fig.add_trace(plotly.graph_objects.Scatter(x=list(range(num_timesteps)), y=greedy_epsilon_0_1_average_num_optimal_selected, mode='lines', name='Epsilon 0.1', line=dict(color=colors['Epsilon 0.1'])), row=2, col=1)

  # Update layout
  fig.update_layout(
    height=900, width=1200,
    title_text="Multi-Armed Bandit Experiment Results",
    margin=dict(l=20, r=20, t=50, b=20)
  )

  # Show the figure
  fig.show()
