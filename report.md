# Udacity Deep Reinforcement Learning - Continuous Control Project.

## Environment
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that 
the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location 
for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. 
Each action is a vector with four numbers, corresponding to torque applicable to two joints.  Every entry in the action 
vector should be a number between -1 and 1.

## Agent

The agent used for this project is a modified version of the [BipedalWalker](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal) 
tutorial provided by the Udacity in their Deep RL nano-degree, which makes use of [DDPG](https://arxiv.org/abs/1509.02971) agent.

Deterministic Policy Gradients is an Actor-Critic method that that has a nice property of being off-policy. DDPG belongs
to the A2C family, but with thge deterministic policy which means that it directly provides us with the action to take from
the state. This means that we can directly improving the policy by maximsing the Q-Value.

DDPG is considered to be an an Actor-Critic method.
* The actor takes in the State and directly maps it to the Action, which is scaled from -1 to 1, so it could account for continuity.
* The critic accepts two values, the current state and the action from the actor network. It concatenates them together
and outputs a single number which corresponds to the Q-Value.

I used a number of techniques to enhance the vanilla DDPG agent and also make it more stable.
First of all, I opted a second version of Unity environment that provided 20 independent environmental agents and supplied
the all of the states to the single DDPG agent with a single rewply buffer. This was done to increase the number of observations
in the reply buffer and with addition to a large batch size of 1024 observations has forced the agent to lear quicker.

In order to make the training more stable I chose to update the agents 10 times in every 20 timestamps as suggested by the
course instructors.


## Model Architecture

### Actor Model


## Results

## Future work

