import copy
import inspectify
import numpy
import plotly.graph_objects
import plotly.subplots

class MultiArmedBandit:
  def __init__(self, num_levers, max_timesteps):
    self.max_timesteps = max_timesteps
    self.num_timesteps = None
    self.action_space = numpy.arange(num_levers)
    self.best_lever = None
    self.best_lever_chosen_count = None
    self.best_lever_chosen_probability = numpy.zeros(self.max_timesteps)
    self.lever_rewards = None
    self.num_runs = 0
    self.random_state = None
    self.rewards = numpy.zeros(self.max_timesteps)
    self.optimal_reward = 0

  def reset(self, seed):
    self.random_state = numpy.random.RandomState(seed)
    self.lever_rewards = self.random_state.normal(0.0, 1.0, self.action_space.size)
    self.optimal_reward += numpy.max(self.lever_rewards)
    self.best_lever = numpy.argmax(self.lever_rewards)
    self.best_lever_chosen_count = 0
    self.num_timesteps = 0
    self.num_runs += 1

  def step(self, action):
    assert action in self.action_space
    if action == self.best_lever: self.best_lever_chosen_count += 1
    reward = self.random_state.normal(self.lever_rewards[action], 1.0)
    self.rewards[self.num_timesteps] += reward
    self.best_lever_chosen_probability[self.num_timesteps] += self.best_lever_chosen_count / (self.num_timesteps + 1)
    self.num_timesteps += 1
    return reward

  def stats(self):
    average_rewards = self.rewards / self.num_runs
    average_probability_optimally_acted = (self.best_lever_chosen_probability / self.num_runs) * 100
    return average_rewards, average_probability_optimally_acted

  def optimal_stats(self):
    average_optimal_reward = numpy.ones(self.max_timesteps) * (self.optimal_reward / self.num_runs)
    average_optimal_probability_optimally_acted = numpy.ones(self.max_timesteps) * 100
    return average_optimal_reward, average_optimal_probability_optimally_acted

class RandomAgent:
  def __init__(self, num_levers):
    self.action_space = numpy.arange(num_levers)
    self.random_state = None

  def reset(self, seed):
    self.random_state = numpy.random.RandomState(seed)

  def act(self):
    action = self.random_state.choice(self.action_space)
    return action

  def learn(self, action, reward):
    pass

class EpsilonGreedyAgent:
  def __init__(self, num_levers, epsilon):
    self.action_space = numpy.arange(num_levers)
    self.random_state = None
    self.sample_count = None
    self.sample_average = None
    self.epsilon = epsilon

  def reset(self, seed):
    self.random_state = numpy.random.RandomState(seed)
    self.sample_count = {action:0 for action in self.action_space}
    self.sample_average = {action:0 for action in self.action_space}

  def act(self):
    if self.random_state.random() < self.epsilon:
      # Exploration: randomly choose any action
      return self.random_state.choice(self.action_space)
    else:
      # Exploitation: choose the best action based on average rewards
      max_average = max(self.sample_average.values())
      best_actions = [action for action, avg_reward in self.sample_average.items() if avg_reward == max_average]
      return self.random_state.choice(best_actions)

  def learn(self, action, reward):
    self.sample_count[action] += 1
    self.sample_average[action] += (1 / self.sample_count[action]) * (reward - self.sample_average[action])

def experiment(num_runs, max_timesteps, num_levers, seed, agent, return_optimal = False):
  numpy.random.seed(seed)
  bandit_environment = MultiArmedBandit(num_levers = num_levers, max_timesteps = max_timesteps)
  for run in range(num_runs):
    agent_seed = numpy.random.randint(999_999)
    env_seed = numpy.random.randint(999_999)
    agent.reset(agent_seed)
    bandit_environment.reset(env_seed)
    for timestep in range(1, bandit_environment.max_timesteps + 1):
      action = agent.act()
      reward = bandit_environment.step(action)
      agent.learn(action, reward)

  average_rewards, average_probability_optimally_acted = bandit_environment.stats()
  if not return_optimal:
    return average_rewards, average_probability_optimally_acted
  else:
    average_optimal_reward, average_optimal_probability_optimally_acted = bandit_environment.optimal_stats()
    return average_rewards, average_probability_optimally_acted, average_optimal_reward, average_optimal_probability_optimally_acted

if __name__ == '__main__':
  max_timesteps = 1_000
  num_runs = 2_00
  num_levers = 10
  seed = 41

  random_average_rewards, random_average_probability_optimally_acted, average_optimal_reward, average_optimal_probability_optimally_acted = experiment(
    num_runs = num_runs,
    max_timesteps = max_timesteps,
    num_levers = num_levers,
    seed = seed,
    agent = RandomAgent(num_levers = 10),
    return_optimal = True,
  )

  greedy_average_rewards, greedy_average_probability_optimally_acted = experiment(
    num_runs = num_runs,
    max_timesteps = max_timesteps,
    num_levers = num_levers,
    seed = seed,
    agent = EpsilonGreedyAgent(num_levers = 10, epsilon = 0),
  )

  greedy_epsilon_0_01_average_rewards, greedy_epsilon_0_01_average_probability_optimally_acted = experiment(
    num_runs = num_runs,
    max_timesteps = max_timesteps,
    num_levers = num_levers,
    seed = seed,
    agent = EpsilonGreedyAgent(num_levers = 10, epsilon = 0.01),
  )

  greedy_epsilon_0_1_average_rewards, greedy_epsilon_0_1_average_probability_optimally_acted = experiment(
    num_runs = num_runs,
    max_timesteps = max_timesteps,
    num_levers = num_levers,
    seed = seed,
    agent = EpsilonGreedyAgent(num_levers = 10, epsilon = 0.1),
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
  fig.add_trace(plotly.graph_objects.Scatter(x=list(range(max_timesteps)), y=average_optimal_reward, mode='lines', name='Optimal', line=dict(color=colors['Random'])), row=1, col=1)
  fig.add_trace(plotly.graph_objects.Scatter(x=list(range(max_timesteps)), y=random_average_rewards, mode='lines', name='Random', line=dict(color=colors['Random'])), row=1, col=1)
  fig.add_trace(plotly.graph_objects.Scatter(x=list(range(max_timesteps)), y=greedy_average_rewards, mode='lines', name='Greedy', line=dict(color=colors['Greedy'])), row=1, col=1)
  fig.add_trace(plotly.graph_objects.Scatter(x=list(range(max_timesteps)), y=greedy_epsilon_0_01_average_rewards, mode='lines', name='Epsilon 0.01', line=dict(color=colors['Epsilon 0.01'])), row=1, col=1)
  fig.add_trace(plotly.graph_objects.Scatter(x=list(range(max_timesteps)), y=greedy_epsilon_0_1_average_rewards, mode='lines', name='Epsilon 0.1', line=dict(color=colors['Epsilon 0.1'])), row=1, col=1)

  # Add plots for % Optimal action
  fig.add_trace(plotly.graph_objects.Scatter(x=list(range(max_timesteps)), y=average_optimal_probability_optimally_acted, mode='lines', name='Optimal', line=dict(color=colors['Random'])), row=2, col=1)
  fig.add_trace(plotly.graph_objects.Scatter(x=list(range(max_timesteps)), y=random_average_probability_optimally_acted, mode='lines', name='Random', line=dict(color=colors['Random'])), row=2, col=1)
  fig.add_trace(plotly.graph_objects.Scatter(x=list(range(max_timesteps)), y=greedy_average_probability_optimally_acted, mode='lines', name='Greedy', line=dict(color=colors['Greedy'])), row=2, col=1)
  fig.add_trace(plotly.graph_objects.Scatter(x=list(range(max_timesteps)), y=greedy_epsilon_0_01_average_probability_optimally_acted, mode='lines', name='Epsilon 0.01', line=dict(color=colors['Epsilon 0.01'])), row=2, col=1)
  fig.add_trace(plotly.graph_objects.Scatter(x=list(range(max_timesteps)), y=greedy_epsilon_0_1_average_probability_optimally_acted, mode='lines', name='Epsilon 0.1', line=dict(color=colors['Epsilon 0.1'])), row=2, col=1)

  # Update layout
  fig.update_layout(
    height=900, width=1200,
    title_text="Multi-Armed Bandit Experiment Results",
    margin=dict(l=20, r=20, t=50, b=20)
  )

  # Show the figure
  fig.show()
