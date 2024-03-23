import copy
import inspectify
import numpy
import plotly.graph_objects
import plotly.subplots



class MultiArmedBandit:
  def __init__(self, levers):
    self.levers = levers
    self.action_space = list(range(levers))
    self.best_lever = None
    self.lever_rewards = None
    self.best_chosen = 0

  def reset(self):
    self.lever_rewards = numpy.random.normal(0.0, 1.0, self.levers)
    self.best_lever = numpy.argmax(self.lever_rewards)
    self.best_chosen = 0
    return 0

  def play(self, action):
    assert action in self.action_space

    if action == self.best_lever:
      self.best_chosen += 1

    reward = numpy.random.normal(self.lever_rewards[action], 1.0)

    return reward

  def copy(self):
    return copy.deepcopy(self)


def bandit_random(env, sample_average):
  action = numpy.random.choice(env.action_space)
  return action

def epsilon_greedy(env, sample_average, epsilon=0):
  if numpy.random.random() < epsilon:
    # Exploration: randomly choose any action
    return bandit_random(env, sample_average)
  else:
    # Exploitation: choose the best action based on average rewards
    max_average = max(sample_average.values())
    best_actions = [action for action, avg_reward in sample_average.items() if avg_reward == max_average]
    return numpy.random.choice(best_actions)

def experiment(num_runs, num_timesteps, num_levers, seed, algorithm):

  numpy.random.seed(seed)

  env = MultiArmedBandit(levers = num_levers)
  rewards = numpy.zeros(num_timesteps)
  best_chosen = numpy.zeros(num_timesteps)

  optimal_rewards = numpy.zeros(num_runs)
  for run in range(num_runs):
    sample_count = {action:0 for action in env.action_space}
    sample_average = {action:0 for action in env.action_space}
    state = env.reset()

    optimal_reward = env.lever_rewards[env.best_lever]
    optimal_rewards[run] = optimal_reward

    for timestep in range(1, num_timesteps + 1):

      action = algorithm(env, sample_average)
      reward = env.play(action)

      sample_count[action] += 1
      sample_average[action] += (1 / sample_count[action]) * (reward - sample_average[action])

      rewards[timestep - 1] += reward
      best_chosen[timestep - 1] += env.best_chosen / timestep

  average_rewards = rewards / num_runs
  average_num_optimal_selected = (best_chosen / num_runs) * 100
  average_optimal_reward = optimal_rewards.mean()

  return average_rewards, average_num_optimal_selected, average_optimal_reward

if __name__ == '__main__':
  num_timesteps = 1_000
  num_runs = 2_000
  num_levers = 10
  seed = 41

  random_average_rewards, random_average_num_optimal_selected, average_optimal_reward = experiment(
    num_runs = num_runs,
    num_timesteps = num_timesteps,
    num_levers = num_levers,
    seed = seed,
    algorithm = bandit_random,
  )

  greedy_average_rewards, greedy_average_num_optimal_selected, _ = experiment(
    num_runs = num_runs,
    num_timesteps = num_timesteps,
    num_levers = num_levers,
    seed = seed,
    algorithm = lambda env, sample_average: epsilon_greedy(env, sample_average, epsilon = 0),
  )

  greedy_epsilon_0_01_average_rewards, greedy_epsilon_0_01_average_num_optimal_selected, _ = experiment(
    num_runs = num_runs,
    num_timesteps = num_timesteps,
    num_levers = num_levers,
    seed = seed,
    algorithm = lambda env, sample_average: epsilon_greedy(env, sample_average, epsilon = 0.01),
  )

  greedy_epsilon_0_1_average_rewards, greedy_epsilon_0_1_average_num_optimal_selected, _ = experiment(
    num_runs = num_runs,
    num_timesteps = num_timesteps,
    num_levers = num_levers,
    seed = seed,
    algorithm = lambda env, sample_average: epsilon_greedy(env, sample_average, epsilon = 0.1),
  )

  # Create a subplot figure with 2 rows and 1 column
  fig = plotly.subplots.make_subplots(rows=2, cols=1, subplot_titles=(
    'Average reward',
    '% Optimal action'))

  # Define colors for each strategy
  colors = {
    'Random': 'orange',
    'Greedy': 'green',
    'Epsilon 0.01': 'red',
    'Epsilon 0.1': 'blue'
  }

  # Add plots for Average reward
  fig.add_trace(plotly.graph_objects.Scatter(x=list(range(num_timesteps)), y=random_average_rewards, mode='lines', name='Random', line=dict(color=colors['Random'])), row=1, col=1)
  fig.add_trace(plotly.graph_objects.Scatter(x=list(range(num_timesteps)), y=greedy_average_rewards, mode='lines', name='Greedy', line=dict(color=colors['Greedy'])), row=1, col=1)
  fig.add_trace(plotly.graph_objects.Scatter(x=list(range(num_timesteps)), y=greedy_epsilon_0_01_average_rewards, mode='lines', name='Epsilon 0.01', line=dict(color=colors['Epsilon 0.01'])), row=1, col=1)
  fig.add_trace(plotly.graph_objects.Scatter(x=list(range(num_timesteps)), y=greedy_epsilon_0_1_average_rewards, mode='lines', name='Epsilon 0.1', line=dict(color=colors['Epsilon 0.1'])), row=1, col=1)

  # Add plots for % Optimal action
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
