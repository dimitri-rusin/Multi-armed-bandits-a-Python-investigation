import chart_studio
import dotenv
import inspectify
import numpy
import os
import pandas
import plotly.graph_objects
import scipy.ndimage

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
  def __init__(self, num_levers, epsilon, alpha, initial_sample_average):
    self.action_space = numpy.arange(num_levers)
    self.random_state = None
    self.alpha = alpha
    self.sample_count = None
    self.sample_average = None
    self.epsilon = epsilon
    self.initial_sample_average = initial_sample_average

  def reset(self, seed):
    self.random_state = numpy.random.RandomState(seed)
    self.sample_count = {action:0 for action in self.action_space}
    self.sample_average = {action:self.initial_sample_average for action in self.action_space}

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
    if self.alpha:
      self.sample_average[action] += self.alpha * (reward - self.sample_average[action])
    else:
      self.sample_count[action] += 1
      self.sample_average[action] += (1 / self.sample_count[action]) * (reward - self.sample_average[action])

class UpperConfidenceBoundAgent:
  def __init__(self, num_levers, c):
    self.action_space = numpy.arange(num_levers)
    self.c = c

  def reset(self, seed):
    self.random_state = numpy.random.RandomState(seed)
    self.num_timesteps = 0
    actions = pandas.Index(self.action_space, name='action')
    self.sample_count = pandas.Series(0, index=actions)
    self.sample_average = pandas.Series(0.0, index=actions)

  def act(self):
    ucb_values = self.sample_average + self.c * numpy.sqrt(numpy.log(self.num_timesteps + 1) / (self.sample_count + 1))
    return ucb_values.idxmax()

  def learn(self, action, reward):
    self.num_timesteps += 1
    self.sample_count[action] += 1
    self.sample_average[action] += (reward - self.sample_average[action]) / self.sample_count[action]

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
  dotenv.load_dotenv()

  username = os.getenv('CHART_STUDIO.USERNAME')
  api_key = os.getenv('CHART_STUDIO.API_KEY')

  chart_studio.tools.set_credentials_file(username = username, api_key = api_key)

  max_timesteps = 4_000
  num_runs = 500
  num_levers = 10
  seed = 41
  window_size = 100

  figure = plotly.graph_objects.Figure()
  figure.update_layout(
    height=900, width=1200,
    title_text=f"{num_levers}-armed bandits, doing {num_runs} trials, seeded with {seed}",
    margin=dict(l=20, r=20, t=50, b=20)
  )
  colors = {
    'Optimal': 'black',
    'Random': 'orange',
    'Greedy': 'green',
    'Optimistic greedy': 'red',
    'Epsilon greedy with 0.01': 'pink',
    'Epsilon greedy with 0.1': 'purple',
    'UCB1': 'yellow',
  }

  average_random_rewards, _, average_optimal_reward, _ = experiment(
    num_runs = num_runs,
    max_timesteps = max_timesteps,
    num_levers = num_levers,
    seed = seed,
    agent = RandomAgent(num_levers = 10),
    return_optimal = True,
  )

  figure.add_trace(plotly.graph_objects.Scatter(
    x=numpy.arange(len(average_optimal_reward)),
    y=scipy.ndimage.uniform_filter1d(average_optimal_reward, size=window_size),
    mode='lines',
    name='Optimal',
    line=dict(color=colors['Optimal'])
  ))
  figure.add_trace(plotly.graph_objects.Scatter(
    x=numpy.arange(len(average_random_rewards)),
    y=scipy.ndimage.uniform_filter1d(average_random_rewards, size=window_size),
    mode='lines',
    name='Random',
    line=dict(color=colors['Random'])
  ))

  figure.write_html(
    "computed/1.html",
    full_html=True,  # Output as a standalone HTML document
    include_plotlyjs=True,  # Include the full Plotly.js in the HTML
    config=None,  # Default plot configuration without modifications
    auto_play=True,  # Animations start automatically
    include_mathjax=False,  # Do not include MathJax unless specifically required
  )


  average_greedy_rewards, _ = experiment(
    num_runs = num_runs,
    max_timesteps = max_timesteps,
    num_levers = num_levers,
    seed = seed,
    agent = EpsilonGreedyAgent(num_levers = 10, epsilon = 0, alpha = None, initial_sample_average = 0),
  )

  figure.add_trace(plotly.graph_objects.Scatter(
    x=numpy.arange(len(average_greedy_rewards)),
    y=scipy.ndimage.uniform_filter1d(average_greedy_rewards, size=window_size),
    mode='lines',
    name='Greedy',
    line=dict(color=colors['Greedy'])
  ))

  figure.write_html(
    "computed/2.html",
    full_html=True,  # Output as a standalone HTML document
    include_plotlyjs=True,  # Include the full Plotly.js in the HTML
    config=None,  # Default plot configuration without modifications
    auto_play=True,  # Animations start automatically
    include_mathjax=False,  # Do not include MathJax unless specifically required
  )



  average_001_epsilon_greedy_rewards, _ = experiment(
    num_runs = num_runs,
    max_timesteps = max_timesteps,
    num_levers = num_levers,
    seed = seed,
    agent = EpsilonGreedyAgent(num_levers = 10, epsilon = 0.01, alpha = None, initial_sample_average = 0),
  )

  average_01_epsilon_greedy_rewards, _ = experiment(
    num_runs = num_runs,
    max_timesteps = max_timesteps,
    num_levers = num_levers,
    seed = seed,
    agent = EpsilonGreedyAgent(num_levers = 10, epsilon = 0.1, alpha = None, initial_sample_average = 0),
  )

  inspectify.d(average_01_epsilon_greedy_rewards)

  figure.add_trace(plotly.graph_objects.Scatter(
    x=numpy.arange(len(average_001_epsilon_greedy_rewards)),
    y=scipy.ndimage.uniform_filter1d(average_001_epsilon_greedy_rewards, size=window_size),
    mode='lines',
    name='Epsilon greedy with 0.01',
    line=dict(color=colors['Epsilon greedy with 0.01']),
  ))
  figure.add_trace(plotly.graph_objects.Scatter(
    x=numpy.arange(len(average_01_epsilon_greedy_rewards)),
    y=scipy.ndimage.uniform_filter1d(average_01_epsilon_greedy_rewards, size=window_size),
    mode='lines',
    name='Epsilon greedy with 0.1',
    line=dict(color=colors['Epsilon greedy with 0.1']),
  ))

  figure.write_html(
    "computed/3.html",
    full_html=True,  # Output as a standalone HTML document
    include_plotlyjs=True,  # Include the full Plotly.js in the HTML
    config=None,  # Default plot configuration without modifications
    auto_play=True,  # Animations start automatically
    include_mathjax=False,  # Do not include MathJax unless specifically required
  )



  average_optimistic_greedy_rewards, _ = experiment(
    num_runs = num_runs,
    max_timesteps = max_timesteps,
    num_levers = num_levers,
    seed = seed,
    agent = EpsilonGreedyAgent(num_levers = 10, epsilon = 0, alpha = 0.1, initial_sample_average = 1),
  )

  figure.add_trace(plotly.graph_objects.Scatter(
    x=numpy.arange(len(average_optimistic_greedy_rewards)),
    y=scipy.ndimage.uniform_filter1d(average_optimistic_greedy_rewards, size=window_size),
    mode='lines',
    name='Optimistic greedy (alpha = 0.1, Q_0 = 1)',
    line=dict(color=colors['Optimistic greedy'])
  ))

  figure.write_html(
    "computed/4.html",
    full_html=True,  # Output as a standalone HTML document
    include_plotlyjs=True,  # Include the full Plotly.js in the HTML
    config=None,  # Default plot configuration without modifications
    auto_play=True,  # Animations start automatically
    include_mathjax=False,  # Do not include MathJax unless specifically required
  )




  average_ucb1_rewards, _ = experiment(
    num_runs = num_runs,
    max_timesteps = max_timesteps,
    num_levers = num_levers,
    seed = seed,
    agent = UpperConfidenceBoundAgent(num_levers = 10, c = 2),
  )

  figure.add_trace(plotly.graph_objects.Scatter(
    x = numpy.arange(len(average_ucb1_rewards)),
    y = scipy.ndimage.uniform_filter1d(average_ucb1_rewards, size=window_size),
    mode = 'lines',
    name = 'UCB1 with c = 2',
    line = {'color': colors['UCB1']},
  ))

  figure.write_html(
    "computed/5.html",
    full_html=True,  # Output as a standalone HTML document
    include_plotlyjs=True,  # Include the full Plotly.js in the HTML
    config=None,  # Default plot configuration without modifications
    auto_play=True,  # Animations start automatically
    include_mathjax=False,  # Do not include MathJax unless specifically required
  )



  chart_studio.plotly.plot(figure, filename="plotly_scatter", auto_open = True)
