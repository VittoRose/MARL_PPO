# MARL_PPO

Repository that contains my implementation of Multi Agent Deep Reinforcement Learning

## Dependency

* pygame 2.6.0
* torch  2.3.1+cu121
* tensorboard 2.15.0
* gymnasium 1.0.0

## Multi Agent Grid Coverage

This environment was created in order to try different approach of multi agent algorithm. The environment work like the vacuum environment [here](https://youtu.be/qgb0gyrpiGk).
The environment was created using gymnasium API and i provided a Jupiter Notebook (environment-example.ipynb) to explain the usage with example.

### Usage

* To train a network for a single agent job run *Single_agent.py*
* To train two agents with Independent PPO (IPPO) run *IPPO_main.py*

### After training

The agents will be saved in *Saved_agents* folder, in the *log* folder you will find a tensorboard file with some information about the training and the report with the hyperparameters used for training.