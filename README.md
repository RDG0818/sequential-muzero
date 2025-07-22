# toy_mazero

Description of major design choices of this MARL Algorithm:

Written in the JAX/FLAX framework for JIT compilation and XLA support
Model-based, similar to MuZero for multi-agent reinforcement learning environments

Utilizes a representation, dynamics, and prediction network for the world model
Utilizes a projection network to force the representation/dynamics network to learn a consistent hidden state
Dynamics network has an optional transformer-style attention mechanism for communication
Utilizes scalar to categorical distribution transforms for reward and value network training stability

Uses the MCTX library for MuZero and Gumbel MuZero functions
Split between independent (per agent) MCTS and joint MCTS

Asynchronous training, similar to EfficientZero, using Ray framework 
Multiple DataActors on CPU, single LearnerActor on GPU

Synchronous training with end-to-end JAX
Data collection and training entirely on GPU

Loss for each network is calculated over k unroll steps, n-step returns for value
Prioritized Experience Replay Buffer
Utilizes LR scheduler, adamw optimizer, gradient clipping, and loss scaling for training stability
Utilizes jaxMARL for jax environments
Uses wandb for logging
Uses docker for environment setup

To Do:
Comment everything in tests, Add more unit tests, clean up existing unit tests, use logging module for unit tests
Make sure no function in main relies on CONFIG, also log learning rate, n-step returns, dirichelet noise for other MCTS versions
model saving/checkpointing/testing/rendering
clean up config
Implement research idea

world model pretraining
Update joint to use optimistic search lambda and AWPO
automated hyperparameter tuning
evaluation script
direchelet noise to root
include state in replay buffer, more metrics return from run_episode
Tuning and variations of attention mechanism

Possible Future Addtions:
Dreamer style dynamics?
Better attention mechanisms?
Value decomposition?
Stochastic World Model?
GNNs for coordination?
Communication protocols?
Legal Actions?
Hydra?
Possibly add a network to create global state out of individual latent states?

Process for updating on AWS:
git pull
docker build -t mazero_clone .
docker run --gpus all --shm-size=32gb -it --rm mazero_clone
python3 main.py
