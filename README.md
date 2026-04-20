# MSDS 684 – Week 1 Lab: Multi-Armed Bandits and MDP Foundations

A comprehensive implementation of exploration-exploitation strategies and Markov Decision Process (MDP) fundamentals using custom Gymnasium environments.

## Overview

This lab explores foundational reinforcement learning concepts through practical implementations:
- **Exploration vs. Exploitation Tradeoff**: Understanding when to exploit known rewards vs. explore new actions
- **Action-Value Methods**: ε-greedy and Upper Confidence Bound (UCB) algorithms
- **Markov Decision Processes**: Value functions, Bellman equations, and optimal policies
- **Gymnasium API**: Building custom environments and interfacing with standard benchmark tasks

## Contents

### Part 1: Custom Multi-Armed Bandit Environment
- **BanditEnv**: A k-armed bandit implementation following the Gymnasium API
  - Gaussian reward distributions with configurable means
  - Tracks optimal action for analysis
  
- **Agent Implementations**:
  - **EpsilonGreedyAgent**: Balances exploration via ε-random actions with exploitation of best-known arms
  - **UCBAgent**: Uses upper confidence bounds for principled exploration

### Part 2: Standard Gymnasium Environments
- **FrozenLake-v1**: 4×4 grid world navigation task (16 states, 4 actions)
- **Taxi-v3**: Taxi pickup/dropoff environment (500 states, 6 actions)

## Key Results

### ε-Greedy Performance
- **ε=0.0** (pure greedy): Converges quickly but often to suboptimal arms
- **ε=0.01**: Slow learning but eventually finds optimal arm
- **ε=0.1 & ε=0.2**: Fast convergence with continuous exploration trade-off

### UCB Performance
- **c=2.0**: Optimal balance between exploration and exploitation
- **c=4.0**: Over-exploration reduces convergence
- **c=0.5**: May converge prematurely to suboptimal arms

### Random Policy Baseline
- **FrozenLake-v1**: 1.8% success rate (average return: 0.02)
- **Taxi-v3**: Average return: -769.79

## Installation

```bash
pip install gymnasium numpy matplotlib
