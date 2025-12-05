# CS 4320 Final Project: Reinforcement Learning Lunar Lander

## Overview

This project implements a comprehensive Reinforcement Learning (RL) system focused on solving the Lunar Lander environment using Deep Q-Networks (DQN) and other advanced techniques. The project explores various RL algorithms, curriculum learning, and reward shaping strategies to achieve optimal performance on the LunarLander-v2 environment.

## Project Structure

```
├── src/
│   ├── agents/          # RL agent implementations (DQN, PPO, etc.)
│   ├── env/            # Environment wrappers and utilities
│   └── utils/          # Core utilities
│       ├── __init__.py
│       └── replay_buffer.py  # Cyclic replay buffer implementation
├── experiments/        # Experiment configurations and checkpoints
├── reports/           # Analysis, plots, and documentation
├── requirements.txt   # Python dependencies
├── verify_environments.py  # Environment verification script
└── README.md         # This file
```

## Features

### Core Components
- **Replay Buffer**: Efficient cyclic buffer storing 50,000 transitions with uniform batch sampling (128 experiences)
- **Environment Verification**: Automated testing of LunarLander-v3 rendering and CartPole smoke tests
- **Modular Architecture**: Clean separation of agents, environments, and utilities

### Planned Techniques
- **Deep Q-Network (DQN)**: MLP-based function approximation with target networks
- **Proximal Policy Optimization (PPO)**: Alternative policy-gradient baseline
- **Advanced Features**:
  - Prioritized Experience Replay
  - Curriculum Learning
  - Reward Shaping
  - Exploration strategies (ε-greedy, entropy bonuses)

## Installation

### Prerequisites
- Python 3.11
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify environment setup:
```bash
python verify_environments.py
```

### Dependencies
- `gymnasium[box2d]==0.29.1` - LunarLander environment
- `torch==2.3.0` - PyTorch for neural networks
- `numpy>=1.24.0` - Numerical computations
- `tqdm>=4.65.0` - Progress bars
- `tensorboard>=2.15.0` - Experiment tracking

## Usage

### Environment Verification
Test your setup with the verification script:
```bash
python verify_environments.py
```
This will check:
- CartPole environment functionality
- LunarLander-v3 rendering capabilities
- Basic environment API compatibility

### Training an Agent
```python
from src.utils.replay_buffer import ReplayBuffer
from src.agents.dqn import DQNAgent

# Initialize replay buffer
buffer = ReplayBuffer(capacity=50000, batch_size=128)

# Initialize agent
agent = DQNAgent(state_dim=8, action_dim=4)

# Training loop would go here...
```

## Configuration

### DQN Hyperparameters
- **Network Architecture**: 2 hidden layers (256 units each) with ReLU activation
- **Learning Rate**: 3e-4 with Adam optimizer
- **Target Network Update**: Every 1,000 gradient steps
- **Gradient Clipping**: 5.0
- **Loss Function**: Huber loss

### Training Parameters
- **Episodes**: 500-1,000 per configuration
- **Random Seeds**: 3 seeds per experiment
- **Evaluation Metrics**:
  - Moving average reward (100-episode window)
  - Landing success rate
  - Fuel efficiency proxy

## Experiments & Results

Experiments are organized in the `experiments/` directory with:
- YAML configuration files
- TensorBoard logs
- Model checkpoints
- Result summaries

### Baseline Performance Goals
- Reproduce Gymnasium reference reward (~200 points)
- Stable training without catastrophic forgetting
- Consistent performance across random seeds

### Advanced Experiments
- **Curriculum Learning**: Gradual difficulty increase
- **Reward Shaping**: Vertical velocity damping, landing gear bonuses
- **Exploration Strategies**: Scheduled ε-greedy decay, parameter noise

## Development Roadmap

### Phase 1: Foundation (Current)
- [x] Project structure setup
- [x] Dependencies configuration
- [x] Replay buffer implementation
- [x] Environment verification

### Phase 2: Core Implementation
- [ ] DQN agent implementation
- [ ] Training pipeline
- [ ] Basic evaluation framework
- [ ] Reproduce baseline performance

### Phase 3: Advanced Features
- [ ] PPO implementation
- [ ] Prioritized replay buffer
- [ ] Curriculum learning
- [ ] Reward shaping experiments

### Phase 4: Analysis & Optimization
- [ ] Hyperparameter sweeps
- [ ] Ablation studies
- [ ] Performance analysis
- [ ] Final report preparation

## Team & Responsibilities

- **Primary Contact**: Environment setup, RL implementation, experiment automation, repository maintenance
- **Evaluation Lead**: Experimental design, hyperparameter sweeps, result aggregation, quantitative analysis
- **Exploration Lead**: Advanced techniques (PPO variants, visualization, curriculum learning), documentation, presentation

## Technical Details

### Environment
- **Task**: LunarLander-v2 (continuous state, discrete action)
- **State Space**: 8-dimensional continuous features
- **Action Space**: 4 discrete actions
- **Reward Structure**: Complex landing mechanics with fuel penalties

### Implementation Notes
- **Deterministic Training**: Proper seeding for reproducibility
- **GPU Support**: CPU-friendly with optional GPU acceleration
- **Logging**: TensorBoard integration with CSV exports
- **Error Handling**: Gradient clipping and replay buffer monitoring

## Contributing

1. Follow the established project structure
2. Implement agents in `src/agents/`
3. Add utilities to `src/utils/`
4. Store experiment configs in `experiments/`
5. Document results in `reports/`

## Risk Mitigation

- **Dependency Issues**: Tested setup with Homebrew/pip fallbacks
- **Training Instability**: Gradient clipping, target networks, reward normalization
- **Compute Limitations**: CPU-optimized with GPU support
- **Scope Creep**: Feature-locked after baseline validation

## License

This project is developed for CS 4320 course requirements.

## Acknowledgments

- OpenAI Gymnasium for the LunarLander environment
- PyTorch team for the deep learning framework
- CS 4320 course staff for project guidance