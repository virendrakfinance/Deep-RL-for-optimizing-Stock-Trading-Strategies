# Deep Q-Learning Trading Agent

## Project Overview

In financial trading, optimizing investment strategies involves managing risk and maximizing returns. This project utilizes Deep Q-Learning, a reinforcement learning technique, to develop a trading agent that learns optimal trading strategies from historical data. The goal is to balance the trade-off between risk and return and construct a robust portfolio.

## Team Member

- **Virendra Kumawat (210110121)**
  

## Key Components

### Risk-Return Trade-Off

Deep Q-Learning helps optimize the balance between risk and return by learning from historical data. Higher returns typically come with increased risk, and this approach aims to find the best balance.

### Portfolio Management

Effective portfolio management involves selecting the right mix of assets to achieve the desired balance between risk and return. The Deep Q-Learning agent simulates different trading strategies to assess their impact on portfolio performance, aiming to maximize returns while adhering to risk tolerance.

## Dataset

The dataset comprises historical NIFTY50 stock data from January 2011 to 2019, sourced from [Yahoo Finance](https://finance.yahoo.com/quote/%5ENSEI/history?period1=1262304000&period2=1559347200&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true). The data was cleaned and includes daily opening prices.

## State Representation

The state representation function generates states based on the dataset, current day, and a window size (set to 5). It calculates price differences between consecutive days and normalizes values using a sigmoid function to keep them between 0 and 1.

## Actions Representation

Three actions are considered:
- **0**: Hold
- **1**: Buy
- **2**: Sell

## Reward Function

The reward function reflects trading profit:
\[ \text{reward} = \max(\text{sell price} - \text{buy price}, 0) \]
Only positive rewards are considered.

## Deep Q-Learning in Trading

Deep Q-Learning is used to train the reinforcement learning (RL) agent with the objective to maximize long-term returns (\( G_t \)). The Q-value estimates the expected return for a given state-action pair:
\[ Q(s, a) = E[G_t \mid S_t = s, A_t = a] \]
A deep neural network approximates these Q-values.

### Bellman Equation

The Q-value from the Bellman equation is used as the target Q-value:
\[ Q(s, a) = R(s, a) + \gamma \max_{a'} Q'(s', a') \]
where \( \gamma \) is the discount factor (set to 0.85).

### Loss Function

The loss function used to update the model weights is:
\[ L(\theta) = \frac{1}{N} \sum_{i \in N} (Q_\theta(S_i, A_i) - Q_{\theta'}(S_i, A_i))^2 \]
This loss is minimized using Stochastic Gradient Descent (SGD):
\[ \theta \leftarrow \theta - \alpha \frac{\partial L}{\partial \theta} \]

## Training the Deep Q Model

### Epsilon Greedy Strategy

The epsilon greedy strategy balances exploration and exploitation. Epsilon (\( \epsilon \)) starts at 1 and decays over time to encourage exploitation. A random number \( r \) determines whether the agent explores or exploits.

### Replay Memory

Replay memory stores experiences:
\[ e_t = (s_t, a_t, r_{t+1}, s_{t+1}) \]
Experiences are used for training by sampling random batches (batch size of 32), calculating Q-targets, and optimizing the model. Epsilon decays with 1000 episodes and a decay rate of 0.01.

## Conclusion

This Deep Q-Learning Trading Agent demonstrates the application of reinforcement learning techniques to financial trading. By integrating state representation, reward mechanisms, and neural network approximation, the model aims to develop effective trading strategies while managing risk and optimizing returns. The approach allows for simulating and evaluating different strategies to create a well-balanced portfolio, highlighting the potential of machine learning in dynamic financial markets.
