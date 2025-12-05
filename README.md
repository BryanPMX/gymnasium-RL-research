# CS 4320 Final Project: Reinforcement Learning Lunar Lander

## Overview

This project extends our study of reinforcement learning to the LunarLander-v2 environment, implementing advanced RL techniques following the CS 4320 final project requirements. The project applies RL methods to a moderately complex environment and investigates techniques for scaling RL and improving learning efficiency.

### Project Requirements (CS 4320)
**Mandatory Techniques:**
- **Function Approximation**: Non-tabular methods using neural networks for value function representation
- **Experience Replay Buffer**: Implementation of experience replay for more efficient learning

**Additional Variations (at least 2):**
- Different learning updates (Proximal Policy Optimization)
- Reward shaping methods
- Curriculum learning strategies
- Advanced exploration techniques

**Deliverables:**
- **Initial Report** (1-2 pages): Team members, environment selection, techniques plan, anticipated issues
- **Final Report** (5-10 pages + source code): Overview, methods, results, analysis, and source code

### Current Status
**Phase 1 (Foundation) is complete** - Project structure, dependencies, replay buffer, and environment verification are all implemented and tested. The team is now executing a revised 3-part work division plan optimized for parallel development across 3 team members.

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

### Mandatory Requirements (CS 4320)
- **Function Approximation**: Neural network-based value function representation (DQN implementation)
- **Experience Replay Buffer**: Cyclic buffer storing 50,000 transitions with uniform batch sampling (128 experiences)
- **Environment Verification**: Automated testing of LunarLander-v2 rendering and CartPole smoke tests

### Additional Techniques (Minimum 2 Required)
- **Deep Q-Network (DQN)**: MLP-based function approximation with target networks
- **Proximal Policy Optimization (PPO)**: Alternative policy-gradient baseline
- **Reward Shaping**: Vertical velocity damping, landing gear contact bonuses
- **Curriculum Learning**: Progressive difficulty increase based on performance
- **Advanced Exploration**: Epsilon-greedy decay, entropy bonuses, parameter noise
- **Prioritized Experience Replay**: Importance-weighted sampling for improved learning efficiency

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

## Project Deliverables

### Initial Report (Due: November 15)
- **Length**: 1-2 pages
- **Content**: Team members and responsibilities, environment selection, techniques plan, anticipated issues
- **File**: `initial_report.md` (already drafted)

### Final Report (Due: December 8)
- **Length**: 5-10 pages plus source code
- **Content**:
  - Project overview and motivation
  - Team member contributions
  - Environment description
  - Methods implemented (with focus on mandatory techniques)
  - Individual method evaluation results
  - Analysis and discussion of findings
- **Files**: Report document and complete source code in `reports/`

### Dependencies
- `gymnasium[box2d]==0.29.1` - LunarLander-v2 environment
- `torch>=2.6.0` - PyTorch for neural networks (function approximation)
- `numpy>=1.24.0` - Numerical computations
- `tqdm>=4.65.0` - Progress bars
- `tensorboard>=2.15.0` - Experiment tracking and logging

## Usage

### Environment Verification
Test your setup with the verification script:
```bash
python verify_environments.py
```
This will check:
- CartPole environment functionality
- LunarLander-v2 rendering capabilities
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

### Phase 1: Foundation (Complete)
- [x] Project structure setup
- [x] Dependencies configuration
- [x] Replay buffer implementation
- [x] Environment verification

### Phase 2: Core Implementation (In Progress - Revised Plan)
Following the revised 3-person work division:

#### Part 1: DQN Agent + Environment Wrappers (Person A - Primary Contact)
- [ ] DQN agent implementation (8→256→256→4, Xavier init, target networks)
- [ ] Environment wrappers (reward shaping, curriculum learning)
- [ ] Agent testing and validation

#### Part 2: Training Pipeline + Evaluation Framework (Person B - Evaluation Lead)
- [ ] Training orchestrator with multi-seed support
- [ ] Configuration management (YAML-based)
- [ ] Evaluation framework with TensorBoard logging
- [ ] Baseline performance reproduction (target: ~200 reward)

#### Part 3: Advanced Features (Person C - Exploration Lead)
- [ ] PPO agent implementation
- [ ] Prioritized experience replay
- [ ] Curriculum learning system
- [ ] Advanced exploration strategies

### Phase 3: Analysis & Optimization (Planned)
- [ ] Hyperparameter sweeps across configurations
- [ ] Ablation studies comparing techniques
- [ ] Performance analysis and visualization
- [ ] Final report preparation and presentation

## Team & Responsibilities

### Work Division (Revised 3-Part Plan)

- **Person A (Primary Contact)**: *Part 1 - DQN Agent + Environment Wrappers*
  - DQN agent implementation with correct architecture (8→256→256→4 MLP)
  - Environment wrappers (reward shaping, curriculum learning)
  - Agent testing, validation, and repository maintenance
  - *Timeline*: Weeks 1-2 | *Deliverables*: Working DQN agent, environment wrappers

- **Person B (Evaluation Lead)**: *Part 2 - Training Pipeline + Evaluation Framework*
  - Training orchestrator supporting 500-1000 episodes across 3 seeds
  - YAML configuration system for hyperparameters
  - TensorBoard logging and evaluation metrics framework
  - Baseline performance reproduction (target: ~200 reward)
  - *Timeline*: Weeks 2-3 | *Deliverables*: Training scripts, evaluation framework, baseline results

- **Person C (Exploration Lead)**: *Part 3 - Advanced Features*
  - PPO agent implementation with actor-critic architecture
  - Prioritized experience replay and advanced exploration
  - Curriculum learning system and experimental variants
  - Documentation, visualization, and presentation preparation
  - *Timeline*: Weeks 3-4 | *Deliverables*: Advanced agents, curriculum system, experimental comparisons

### Integration Points
- **Weekly Check-ins**: Code reviews and integration testing
- **Shared Testing**: Common test suite ensures compatibility
- **Documentation**: Each component includes usage examples and hyperparameter justifications

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