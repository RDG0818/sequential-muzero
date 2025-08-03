# Sequential MuZero Research Project
Multi-agent MuZero-style world-model in JAX/FLAX for research and prototyping. Developed originally for the Tactical Behaviors for Autonomous Maneuver (TBAM) project in collaboration with Mississippi State University, Rutgers University, and the Army Research Lab. For more information, contact: rdg291@msstate.edu.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Project Structure and Implementation Notes](#project-structure-and-implementation-notes)
5. [Research Directions & Future Ideas](#research-directions--future-ideas)
6. [TODO & Must-Implement Features](#todo--must-implement-features)
7. [What Didn't Work](#what-didnt-work)
8. [Relevant Papers](#relevant-papers)
9. [License](#license)

## Overview
This project is a model-based multi-agent reinforcement learning framework built on JAX/FLAX. It extends MuZero-style planning (MCTS + learned world-model) to multi-agent environments.

The motivation behind this project was to explore more efficient ways to search over the exponential action space in multi-agent environments, such as sequential MCTS where each agent searches over its own action space and shares search statistics between each tree. This project also serves as a clean, fast, and modular codebase to test new ideas and extensions of MuZero in multi-agent environments.

## Features

- Uses **[JAX](https://github.com/google/jax)** for fast, composable function transformations (JIT, grad, vmap, pmap) and XLA-accelerated computation.  

- Uses **[Flax](https://github.com/google/flax)** as a modular, flexible neural network library for JAX.

- **World-model**
    - The representation network encodes observations into a latent hidden state.
    - The dynamics network takes in a latent state and actions, and outputs the next (imagined) latent state.
    - The prediction network takes in a latent state and outputs the policy (action distribution) and value (long term reward).
    - The projection network encourages latent state temporal consistency and stabilizes training.
    - The transformer-style attention mechanism enables message passing for agent communication and more informed latent states.
    - The scalar to categorical distribution transforms for reward and value network improves training stability and reduces variance.

- **Planning**
    - **[MCTX](https://github.com/google-deepmind/mctx)** is a fast, JAX-native MuZero-style Monte Carlo Tree Search (MCTS) implementation developed by DeepMind. Enables differentiable, batched search with support for both standard and Gumbel MuZero variants. Used in all the following planning variants.
    - Dirichelet noise is added to the root state of each search to encourage exploration. 
    - In independent MCTS, each agent performs its own search independently and assumes other agents do the argmax of their policy network, inspired by adaptations of MuZero to multi-agent settings.
    - In sequential MCTS (WIP), each agent performs its own search conditioned on the results/statistics/actions of the previous agents that have planned.
    - In joint sampled MCTS, a single search tree explores joint actions across all agents, with sampling to improve scalability and efficiency for the exponential action space.


- **Training**
    - Uses an asynchronous actor-learner framework with Ray, inspired by EfficientZero. Uses a centralized GPU-based learner and multiple CPU-based data actors running in parallel. Actors collect experience independently and send data to a replay buffer, which is sampled to be used for the learner. The data actors are periodically updated with new parameters from the learner. 
    - A Prioritized Experience Replay is used to sample from the buffer with probabilities proportional to their TD error, which accelerates convergence.
    - Uses the AdamW optimizer with a learning rate scheduler, gradient clipping, and optional loss scaling to ensure stable training in deep latent models. These mechanisms mitigate exploding gradients, vanishing updates, and help tune across diverse environments.
    - The model is trained over `k` unroll steps through its dynamics network, learning to simulate forward in latent space. Targets for value prediction use `n`-step returns to improve learning signal and credit assignment.
- **Utilities**
    - Compatible with [jaxMARL](https://github.com/flairox/jaxmarl), a modular framework for multi-agent RL environments in JAX. 
    - All key metrics — rewards, losses, gradients, learning rates, and planning statistics — are tracked and visualized in real time via Weights & Biases integration.

## Installation
You can run this project either locally or using the provided Docker setup, which is recommended for consistent environments and GPU support (e.g. on AWS).

### Local Installation

**1. Clone the repository**
```bash
git clone https://github.com/RDG0818/sequential-muzero
cd sequential-muzero
```

**2. Create Conda environment**
```bash
conda create -n muzero python=3.10.18
conda activate muzero
```

**3. Install dependencies, setup logging, and quickstart**
```bash
pip install -r requirements.txt
wandb login
python train.py
```

### Docker
Full containerized training environment, tested on AWS g4dn.8xlarge EC2 instance with Nvidia GPUs. Note that NVIDIA Container Toolkit is required.

```bash
docker build -t muzero_clone .
docker run --gpus all --shm-size=32bg -it --rm muzero_clone
python3 train.py
```

## Project Structure and Implementation Notes

- `train.py` - defines the replay buffer, learner, and data actors as well as the training loop.
- `config.py` - defines the major hyperparameters used in the entire project
- `model/model.py` - defines the representation, dynamics, prediction, and projection networks and the `FlaxMAMuZeroNet` API
- `mcts/` - folder that contains files defining the abstract base class for the MCTS planners and the logic for the independent, sequential, and joint variations.
- `utils/replay_buffer.py` - defines the replay buffer and prioritized experience replay logic
- `utils/mpe_env_wrapper.py` - defines the environment wrapper for the MPE environments
- `utils/utils.py` - defines the categorical transforms for the reward and value as well as other useful functions

### For the Next Developer
This project includes several implementation-specific considerations that are important for stability and extensibility, especially due to JAX and Ray incompatibilites. Most critically, JAX eagerly allocates the entire GPU, and Ray spawns multiple processes, which can lead to segmentation faults or silent errors if JAX objects are misused across actor boundaries. To avoid this:

- Always import JAX modules *inside* each Ray actor.
- Never create or hold any JAX-related objects in the global scope of a Ray-managed process.
- The replay buffer converts everything to NumPy arrays before storage to prevent accidentally capturing JAX traces or DeviceArrays across processes. 

Violating these constraints often leads to `SEGFAULT` or driver-related crashes during execution. There are also driver issues that can happen during installation. Utilizing the docker installation is recommended to avoid these issues.

 >  If you're running into strange errors or debugging Ray+JAX interaction, feel free to reach out: rdg291@msstate.edu

## Research Directions and Future Ideas

The core research idea for this codebase is to reduce the exponential action space size of joint MCTS. 

---

### Key Idea
Instead of running a full joint MCTS over the joint action space $A^N$, we decompose the planning process into $N$ individual searches, each over an individual agent's action space $A$. This reduces the planning complexity from exponential to linear in the number of agents. This is implemented in `mcts/mcts_independent.py`. However, this approach does not encourage any form of coordination between the agents, and effectively makes each agent individually greedy. To address this, we take inspiration from [this paper](https://arxiv.org/pdf/2304.09870), specifically the Multi-Agent Decomposition Lemma. In this case, instead of conditioning on advantages, we condition on prior agents' searches. This introduces a form of coordination between the agents while still avoiding an exponential action space. The basic structure of this idea is implemented in `mcts/mcts_sequential.py`.

However, this approach introduces several research questions: 
- What information should be passed between the searches?
- What does it mean to "condition" a search?
- How should the information affect the search? 
- How can we do this without breaking Centralized Training, Decentralized Execution (CTDE)?

From several experimental dead ends (see *What Didn’t Work*), here is a distilled approach that seems most logical so far:

- Sequential Plan Propagation: There must be a mechanism (e.g., GRU, Transformer) to create and pass a stateful representation of the team plan between each agent's planning stage.

- Plan Integration: The team plan must be used to conditionally modify one of the core components of the search process: the policy (action selection), the value (search guidance), or the state representation. This can occur in either the macro level (between searches at the root, `_plan_loop`) or the micro level (inside the searches, `recurrent_fn`).

- The networks must work the same at the macro and micro levels to avoid diluting gradients. Having major differences in root level decision vs. search decisions will be less effective then even the most simple independent MCTS.

Here is a non-exhaustive list of future ideas, which may or may not work:
- Replacing the policy MLP network with an RNN based network (GRU, LSTM, transformer)
- Incorporating Dreamv3 style dynamics for improved sample efficiency
- Some form of value decomposition, such as QMIX
- some form of search optimism as discussed in the [MAZero paper](https://openreview.net/pdf?id=CpnKq3UJwp) (note that this is very demanding on implementation and requires significant modification the MCTX high level functions)

## TODO and Must-Implement Features

There are several TODOs listed throughout the codebase. If you are new to this codebase, I would suggest working on this first, as this will make later development significantly easier. Here is a non-exhaustive list of necessary features:

- model saving logic/testing script 
- visualizations of the agent in the environment
- reanalyze actors to avoid stale data in the replay buffer
- environment wrapper abstract base class (this exists in the synch branch, so you could use that one)
- SMAC environment wrapper
- minor optimizations/stylistic choices through out the codebase
- unit testing for all files in the repo (**highly suggested**)

## What Didn't Work

### Permutation Invariant Critic (implemented on the MAZero codebase)
**What I tried:**
- Integrated a [Permutation Invariant Critic](https://arxiv.org/pdf/1911.00025) (PIC) as a replacement for the MLP-based reward and value networks.

**Why I thought it might help:**
- Theoretically, a permutation-invariant architecture would better model inter-agent relationships and generalize across agent orderings, leading to improved reward/value estimation and more stable training.

**Why it didn't work:**
- In practice, it yielded minimal performance gains on several SMAC environments while adding significant architectural complexity. It also didn’t offer enough novelty to justify the tradeoff from a research perspective.

**Notes:**
- May still be beneficial in settings with larger agent counts or more complex coordination — as shown in the original PIC paper. Also relatively easy to plug into existing pipelines if revisiting.

### Synchronous Training
**What I tried:**
- Replaced the asynchronous EfficientZero-style architecture with a fully synchronous training loop written entirely in JAX (see `synch` branch).

**Why I thought it might help:**
- Expected significant wall-clock performance gains by removing CPU-GPU communication overhead and leveraging JAX’s speed and simplicity. The synchronous loop also makes the system easier to debug and reason about.

**Why it didn't work:**
- Performance was worse than random — the policy entered a negative learning cycle and failed to improve. The root cause remains unclear despite extensive debugging. While reduced data diversity could be a factor, it doesn’t fully explain the degradation.

**Notes:**
- If someone can debug and fix this, the codebase could become much simpler and faster. The `synch` branch includes several unit tests and instrumentation to help trace the problem. Worth revisiting if you want to eliminate Ray entirely.

### Delta Network + Coordination Context
**What I tried:**
- A coordination cell (a GRU) that aggregated hidden states and action statistics from the previous agent’s search. This produced a "planning vector" passed to the next agent, allowing agents to condition their search on the decisions of earlier agents. 
- A delta network (an MLP) that took the current state, predicted policy, and planning vector to compute a delta — a correction applied to the off-turn agents' actions to make them more coordinated with the current agent

**Why I thought it might help:**
- This idea attempts to mitigate the coordination problem of independent MCTS while still maintaining Centralized Training with Decentralized Execution (CTDE). My hypothesis was that the gap between an optimal (coordinated) policy and a greedy, independent one should be learnable. The delta network would capture this difference and apply it during search to bias agents toward coordinated behavior.

**Why it didn't work:**
- The main issue was a lack of a clear, effective loss function. Training the delta network to match MCTS policy targets simply pushed the delta to zero, having no effect. Value-based gradient methods weren’t applicable due to the strictly off-policy nature of the data. Additionally, the planning vector was only meaningful at the root of the search tree, yet had to be applied several simulation steps later. This introduced temporal staleness, instability, and noise in coordination signals. The overall system added substantial complexity and did not improve performance on tested environments.

**Notes:**
- Conceptually, this remains one of the few approaches I can think of that combines inter-agent conditioning with CTDE constraints. If someone can address the temporal staleness issue and creating a proper learning objective, then this architecture may still offer a promising way to guide multi-agent coordination during latent-space planning.

### Policy Network Conditioning

**What I tried:**
- A simplified variation of the delta network idea, where instead of applying a learned delta to the policy, I directly fed the planning vector into the policy network as an additional input. During unconditioned rollouts, a zero vector was passed instead.

**Why I thought it might help:**
- The delta network was collapsing to zero and lacked a stable training objective. By integrating the planning signal directly into the policy network, I hoped it would learn to interpret and use the coordination information more effectively, simplifying the architecture.

**Why it didn't work:**
- This doesn't address the temporal staleness in the planning vector. During the search, a planning vector with information about state $s_t$ is not applicable to state $s_{t+z}$. Thus, even with the combined roles, the additional information given simply isn't useful. Additionally, it was difficult to ensure the network was meaningfully using the vector without over-relying on it or ignoring it entirely. It also splits the gradients for the policy network between training to work with the planning vector and without it. This becomes more apparent as well with the training testing discrepancy. The training almost always has the planning vector accessible, but during execution, the planning vector is essentially always zero. The policy network improving with the planning vector doesn't imply that the policy network is improving its output without the planning vector, so its overall performance was worse than the independent implementation.

**Notes**:
- I don't think this idea really has any potential. However, there are some clever ideas to fix the temporal staleness issue. Having another network adjust the planning vector depending on the state could be an interesting idea. You could also make the training force the planning vector to encode short-term relevant information.

### Autoregressive Policy Network
**What I tried:**
- Attempted to implement the autoregressive policy network from [this paper](https://github.com/PKU-MARL/Multi-Agent-Transformer), which models inter-agent dependencies by generating agent actions sequentially. This breaks CTDE, but shows strong empirical performance in cooperative multi-agent tasks.

**Why I thought it might help:**
- The paper aligns closely with our research direction, especially in modeling agent-level coordination through structured action generation. Although it was demonstrated in an on-policy PPO-style setup, the same principles seemed applicable to our setting.

**Why it didn't work:**
- I was unable to complete the implementation due to the architectural complexity of translating a full transformer-based autoregressive model into this codebase. The `transformer` branch contains an early-stage attempt, but progress was blocked by difficult-to-debug JAX tracer issues, likely caused by the model's recursive, sequential structure.
Even with a working implementation, the approach would be computationally infeasible under current infrastructure: each policy call becomes a quadratic-time operation in both the number of agents and search steps, which is incompatible with efficient planning in anything but toy environments.

**Notes:**
- I believe this is the most promising idea for improving coordination quality. There is strong supporting evidence from the multi-agent RL literature, and if the wall-clock time issue is addressed and the CTDE violations can be tolerated, then I believe this is the most paper-worthy idea. However, this is an extremely complicated implementation, and implementing optimizations to the attention process (caching, quantization, etc.) will drastically increase code complexity. If you want to pursue this direction, expect to dedicate a significant amount of time debugging, profiling, and implementing just the architecture of this idea.

## Relevant Papers

- [MuZero](https://arxiv.org/pdf/1911.08265)
- [MAZero](https://openreview.net/pdf?id=CpnKq3UJwp)
- [Multi-agent Transformer](https://arxiv.org/pdf/2205.14953)
- [Permutation Invariant Critic](https://arxiv.org/pdf/1911.00025)
- [Hetergeneous Agent Reinforcment Learning](https://arxiv.org/pdf/2304.09870)

## License

This project is licensed under the [MIT License](LICENSE).  
You are free to use, modify, and distribute this software with attribution. See the LICENSE file for details.
