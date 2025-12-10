---
sidebar_label: Reinforcement Learning
---

# Reinforcement Learning

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties based on its actions and aims to maximize the cumulative reward over time through trial and error.

## Overview of Reinforcement Learning

In reinforcement learning, an agent learns to achieve a goal by interacting with an environment. The key components are:

- **Agent**: The learner or decision-maker
- **Environment**: Everything the agent interacts with
- **State**: The current situation of the agent
- **Action**: What the agent can do
- **Reward**: Feedback from the environment
- **Policy**: The agent's strategy for choosing actions
- **Value Function**: Prediction of future rewards

## Key Concepts in RL

### 1. The Agent-Environment Interface

The fundamental interaction loop:
1. The environment provides state S to the agent
2. The agent selects action A based on its policy
3. The environment transitions to new state S' and provides reward R
4. The process repeats

### 2. Exploration vs. Exploitation
- **Exploration**: Trying new actions to discover their effects
- **Exploitation**: Using known information to maximize reward
- Balancing these is crucial for effective learning

## Types of Reinforcement Learning

### 1. Model-Free vs. Model-Based

#### Model-Free RL
The agent learns directly from experience without modeling the environment.

##### Q-Learning Example
```python
import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
    
    def choose_action(self, state):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.epsilon:
            # Explore: random action
            return random.choice(self.actions)
        else:
            # Exploit: best known action
            state_actions = self.q_table[state]
            return self.actions[np.argmax(state_actions)]
    
    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][self.actions.index(action)]
        
        # Calculate maximum future reward
        max_next_q = np.max(self.q_table[next_state])
        
        # Update Q-value using Bellman equation
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][self.actions.index(action)] = new_q

# Simple grid world example
class GridWorld:
    def __init__(self, width=5, height=5):
        self.width = width
        self.height = height
        self.start = (0, 0)
        self.goal = (4, 4)
        self.state = self.start
        self.actions = ['up', 'down', 'left', 'right']
        
    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        x, y = self.state
        
        # Update position based on action
        if action == 'up' and y > 0:
            y -= 1
        elif action == 'down' and y < self.height - 1:
            y += 1
        elif action == 'left' and x > 0:
            x -= 1
        elif action == 'right' and x < self.width - 1:
            x += 1
            
        self.state = (x, y)
        
        # Calculate reward
        if self.state == self.goal:
            reward = 100  # Goal reached
            done = True
        elif self.state == (2, 2):  # Penalty position
            reward = -10
            done = False
        else:
            reward = -1  # Time penalty
            done = False
            
        return self.state, reward, done

# Train the agent
env = GridWorld()
agent = QLearningAgent(env.actions)

# Training loop
for episode in range(1000):
    state = env.reset()
    total_reward = 0
    
    while True:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        
        agent.learn(state, action, reward, next_state)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    # Decay exploration rate
    if agent.epsilon > 0.01:
        agent.epsilon *= 0.995

print("Training complete!")
print("Final Q-table for a few states:")
for state in [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]:
    print(f"State {state}: {dict(zip(env.actions, agent.q_table[state]))}")
```

#### Deep Q-Network (DQN)
Combines Q-learning with deep neural networks.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # Neural networks
        self.q_network = DQN(state_size, hidden_size, action_size)
        self.target_network = DQN(state_size, hidden_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.update_target_freq = 100
        self.step_count = 0
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.95 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Example usage would involve training the DQN on an environment
```

### 2. Policy-Based Methods

Instead of learning a value function, these methods learn a policy directly.

#### Policy Gradient Example (REINFORCE)
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

class PolicyGradientAgent:
    def __init__(self, state_size, action_size, hidden_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.network = PolicyNetwork(state_size, hidden_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.01)
        self.log_probs = []
        self.rewards = []
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.network(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        
        # Store log probability for gradient computation
        self.log_probs.append(action_dist.log_prob(action))
        return action.item()
    
    def store_reward(self, reward):
        self.rewards.append(reward)
    
    def update_policy(self):
        # Compute discounted rewards
        discounted_rewards = []
        running_add = 0
        for reward in reversed(self.rewards):
            running_add = reward + 0.99 * running_add
            discounted_rewards.insert(0, running_add)
        
        # Normalize rewards
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        # Compute loss
        policy_loss = []
        for log_prob, reward in zip(self.log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        
        # Update network
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Reset for next episode
        self.log_probs = []
        self.rewards = []
```

### 3. Actor-Critic Methods
Combine value-based and policy-based approaches.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value) head
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, state):
        shared_out = self.shared(state)
        action_probs = self.actor(shared_out)
        state_value = self.critic(shared_out)
        return action_probs, state_value

class ActorCriticAgent:
    def __init__(self, state_size, action_size, hidden_size=32):
        self.network = ActorCritic(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, state_value = self.network(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        return action.item(), state_value, action_dist.log_prob(action)
    
    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        
        action_probs, current_value = self.network(state)
        _, next_value = self.network(next_state)
        
        # Calculate advantage
        target_value = reward + 0.99 * next_value * (1 - done)
        advantage = target_value - current_value
        
        # Calculate actor and critic losses
        action_dist = torch.distributions.Categorical(action_probs)
        log_prob = action_dist.log_prob(torch.tensor([action]))
        actor_loss = -log_prob * advantage.detach()
        critic_loss = advantage.pow(2)
        
        # Update network
        total_loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
```

## RL Environments and Applications

### 1. Classic Control Problems
```python
import gym

def cartpole_example():
    # CartPole environment from OpenAI Gym
    env = gym.make('CartPole-v1')
    
    # Simple random agent
    observation = env.reset()
    total_reward = 0
    
    for step in range(1000):
        env.render()  # Comment out if running without display
        
        # Choose random action
        action = env.action_space.sample()
        
        # Take action
        observation, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            print(f"Episode finished after {step+1} timesteps with total reward: {total_reward}")
            break
    
    env.close()

# Run the example
# cartpole_example()
```

### 2. Multi-Armed Bandit Problem
```python
import numpy as np
import matplotlib.pyplot as plt

class MultiArmedBandit:
    def __init__(self, n_arms):
        # True reward probabilities for each arm
        self.probabilities = np.random.uniform(0, 1, n_arms)
        self.n_arms = n_arms
        
    def pull_arm(self, arm):
        # Return reward based on arm's probability
        return 1 if np.random.random() < self.probabilities[arm] else 0

class EpsilonGreedyAgent:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        
    def select_action(self):
        if np.random.random() > self.epsilon:
            # Exploit: choose best arm
            return np.argmax(self.values)
        else:
            # Explore: choose random arm
            return np.random.randint(self.n_arms)
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        # Update value using incremental formula
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

def run_bandit_experiment(n_arms=10, n_steps=1000):
    bandit = MultiArmedBandit(n_arms)
    agent = EpsilonGreedyAgent(n_arms)
    
    rewards = []
    optimal_action_count = 0
    true_optimal_arm = np.argmax(bandit.probabilities)
    
    for step in range(n_steps):
        arm = agent.select_action()
        reward = bandit.pull_arm(arm)
        agent.update(arm, reward)
        
        rewards.append(reward)
        
        if arm == true_optimal_arm:
            optimal_action_count += 1
    
    print(f"Best arm: {true_optimal_arm}, estimated probability: {agent.values[true_optimal_arm]:.3f}")
    print(f"True probability: {bandit.probabilities[true_optimal_arm]:.3f}")
    print(f"Optimal action selected {optimal_action_count}/{n_steps} times")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(rewards) / (np.arange(len(rewards)) + 1))
    plt.title('Average Reward Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(agent.values, 'o-', label='Estimated Values')
    plt.plot(bandit.probabilities, 's-', label='True Values')
    plt.title('Estimated vs True Arm Values')
    plt.xlabel('Arm')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

run_bandit_experiment()
```

## Advanced RL Concepts

### 1. Deep Reinforcement Learning
Combining RL with deep neural networks for complex tasks.

```python
import torch
import torch.nn as nn
import numpy as np

class DeepNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[64, 64]):
        super(DeepNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Example of a deep network for a complex RL task
# This would typically be used in more sophisticated algorithms like PPO, A3C, etc.
def create_deep_rl_model(state_size, action_size):
    return DeepNetwork(state_size, action_size, hidden_sizes=[128, 64, 32])
```

### 2. Continuous Action Spaces
For environments with continuous action spaces, algorithms like Deep Deterministic Policy Gradient (DDPG) are used.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        action = self.tanh(self.fc3(x))  # Output between -1 and 1
        return action

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)
        return value
```

## RL Applications

### 1. Game Playing
RL agents have achieved superhuman performance in games like:
- Go (AlphaGo)
- Chess and Shogi (AlphaZero)
- Atari games (DQN)
- Dota 2 and StarCraft II

### 2. Robotics
- Manipulation tasks
- Navigation
- Adaptive control

### 3. Finance
- Portfolio management
- Algorithmic trading
- Risk assessment

### 4. Healthcare
- Treatment optimization
- Drug discovery
- Personalized medicine

## Challenges in Reinforcement Learning

1. **Sample Efficiency**: Often requiring many interactions with the environment
2. **Exploration**: Balancing exploration of unknown states with exploitation of known good actions
3. **Credit Assignment**: Determining which actions were responsible for rewards
4. **Non-stationarity**: Environment changing as the agent learns
5. **Scalability**: Handling high-dimensional state and action spaces
6. **Safety**: Ensuring agents don't take dangerous actions during learning

## Future Directions

- **Meta-learning**: Agents that learn to learn quickly
- **Multi-agent RL**: Coordination between multiple agents
- **Transfer Learning**: Applying learned policies to new domains
- **Explainable RL**: Understanding and interpreting agent decisions

Reinforcement learning continues to advance rapidly, with new algorithms and applications emerging regularly. Its ability to learn from interaction makes it particularly suitable for complex decision-making problems.