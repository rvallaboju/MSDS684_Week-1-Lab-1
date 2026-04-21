# MSDS 684 – Week 1 Lab: Multi-Armed Bandits and MDP Foundations

## Overview

This lab introduces fundamental concepts in reinforcement learning, focusing on the **exploration vs. exploitation tradeoff**, action-value methods, and Markov Decision Processes (MDPs). Students implement custom bandit environments, develop intelligent agents, and experiment with Gymnasium environments.

## Lab Objectives

1. **Create a custom multi-armed bandit environment** following the Gymnasium API
2. **Implement action-value agents**:
   - ε-greedy agent with incremental updates
   - Upper Confidence Bound (UCB) agent
3. **Run large-scale experiments** (2000 steps × 1000 runs) to compare exploration strategies
4. **Explore standard Gymnasium environments** (FrozenLake-v1, Taxi-v3)
5. **Evaluate random policy performance** and map theoretical MDP concepts to practical implementations

## Key Topics Covered

- **Exploration vs. Exploitation**: Understanding the tradeoff between trying new actions and exploiting known good actions
- **Action-Value Methods**:
  - ε-greedy: Simple exploration strategy with fixed exploration probability
  - UCB (Upper Confidence Bound): Optimistic exploration based on uncertainty estimates
- **Markov Decision Processes (MDPs)**:
  - State spaces (S) and action spaces (A)
  - Transition probabilities (P)
  - Reward functions (R)
  - Value functions and Bellman equations
- **Gymnasium API**: Modern RL environment framework for creating and interacting with environments

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/rvallaboju/MSDS684_Week-1-Lab-1.git
cd MSDS684_Week-1-Lab-1
```

2. **Install required packages**:
```bash
pip install -r requirements.txt
```

This will install:
- **numpy** (≥1.21.0): Numerical computing
- **matplotlib** (≥3.4.0): Visualization and plotting
- **gymnasium** (≥0.29.0): RL environment framework

### Running the Lab

1. **Launch Jupyter Notebook**:
```bash
jupyter notebook Vallaboju_Ruthika_Lab1.ipynb
```

2. **Execute all cells** in order to run the complete lab

## Lab Structure

### Part 1: Custom Multi-Armed Bandit Environment

**BanditEnv Class**:
- Implements a k-armed bandit with Gaussian rewards
- Each arm has a true mean (`q_star[a]`) sampled from N(0,1)
- Rewards are drawn from N(mean, std=1.0)
- Follows Gymnasium API conventions

**EpsilonGreedyAgent**:
- Maintains estimated action-values (Q) and visit counts (N)
- Explores with probability ε; exploits otherwise
- Updates Q using incremental sample averaging: Q ← Q + (R - Q) / N

**UCBAgent**:
- Uses Upper Confidence Bound selection rule
- Action selection: UCB = Q(a) + c√(ln(t) / N(a))
- Parameter c controls exploration-exploitation balance
- Initially tries each arm once for unbiased estimates

**Experiments**:
- Runs 1000 independent 2000-step episodes per configuration
- ε-Greedy tested with ε ∈ {0.0, 0.01, 0.1, 0.2}
- UCB tested with c ∈ {0.5, 1.0, 2.0, 4.0}
- Plots average reward and optimal action rate

**Key Observations**:
- ε=0 (pure greedy): Gets stuck on suboptimal arms
- ε=0.01: Slow learning but finds optimal arm eventually
- ε=0.1, 0.2: Fast convergence but lower asymptotic performance
- UCB c=2: Best balance of exploration and convergence
- UCB c=4: Over-explores; UCB c=0.5: May converge prematurely

### Part 2: Standard Gymnasium Environments

**FrozenLake-v1**:
- 16-state grid world (4×4)
- 4 discrete actions (up, down, left, right)
- Goal: Reach destination without falling on ice
- Deterministic transitions
- Reward: 1 at goal, 0 elsewhere

**Taxi-v3**:
- 500-state environment (5×5 grid + 5 locations + 4 passenger states)
- 6 discrete actions (4 movements + pickup + dropoff)
- Goal: Pick up and drop off passengers efficiently
- Random transitions
- Reward: +20 for successful dropoff, -1 per step, -10 for illegal actions

**Random Policy Evaluation**:
- FrozenLake-v1: 1.8% success rate (avg. return: 0.02)
- Taxi-v3: Avg. return: -769.79 (inefficient random exploration)

### Part 3: MDP to Gymnasium Mapping

| MDP Component | Gymnasium Implementation |
|---|---|
| State space (S) | `env.observation_space` |
| Action space (A) | `env.action_space` |
| Transition dynamics (P) | Hidden in `env.step()`; not directly accessible |
| Reward function (R) | Returned as `reward` by `env.step()` |
| Discount factor (γ) | Defined by agent; not in environment |
| Policy (π) | Implemented in `agent.select_action()` |
| Episodes | Loop from `env.reset()` until `terminated` or `truncated` |

## Expected Output

The notebook generates:
1. **Smoothed reward curves** comparing exploration strategies
2. **Optimal action rate graphs** showing convergence behavior
3. **Environment specifications** and action space details
4. **Random policy statistics** for baseline comparison
5. **MDP component mapping table** for theoretical understanding

## Files

- `Vallaboju_Ruthika_Lab1.ipynb`: Main lab notebook with all code and visualizations
- `requirements.txt`: Python package dependencies
- `README.md`: This file

## Key Takeaways

1. **Exploration matters**: No exploration (ε=0) leads to suboptimal performance
2. **Parameter tuning is critical**: Different ε and c values suit different tasks
3. **Gymnasium provides a standard interface** for implementing RL algorithms
4. **Large-scale experiments** (1000 runs) reveal true algorithm performance
5. **MDP theory connects to practice** through the Gymnasium API

## Future Work

This lab foundation enables:
- Dynamic Programming algorithms (Value/Policy Iteration)
- Model-free reinforcement learning (Q-Learning, SARSA)
- Deep Reinforcement Learning (DQN)
- Policy gradient methods (REINFORCE, Actor-Critic)

## References

- Sutton & Barto (2018): "Reinforcement Learning: An Introduction"
- Gymnasium Documentation: https://gymnasium.farama.org/
- Multi-Armed Bandits: Classic exploration-exploitation problem

---

**Last Updated**: 2026-04-21 17:16:38 UTC  
**Author**: Ruthika Vallaboju  
**Course**: MSDS 684 – Reinforcement Learning