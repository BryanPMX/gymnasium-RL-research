CS 4320 Final Project - Initial Report (Draft)
Date: 15 Nov 2025

## 1. Team Members and Responsibilities
Collaborator A will serve as the primary contact and will focus on environment setup, reinforcement learning implementation, experiment automation, and repository maintenance. Collaborator B (Evaluation Lead) will handle experimental design, hyperparameter sweeps, result aggregation, and drafting the quantitative sections of the reports. Collaborator C (Exploration Lead) will own stretch initiatives such as adding policy-gradient variants, visualization tooling, and curriculum or reward-shaping experiments, while also supporting documentation and presentation prep. The three-member roster and division of labor will be finalized with the instructor before the next milestone.

## 2. Environment / Problem Statement
The project will use the Gymnasium Box2D environment LunarLander-v3. This task offers a state space with continuous features and a discrete action set that remains approachable while still satisfying the requirement that the domain be of low to moderate complexity. The environment is well documented, which allows us to focus on designing learning strategies rather than on debugging simulation details. It is also easy to scale: the discrete setting can later transition to LunarLanderContinuous-v3 or to variants with perturbed dynamics, enabling curriculum-style experimentation without changing tools. Before committing to the full lander problem, the team will validate the training stack on CartPole and selected ToyText problems to ensure that logging, seeding, and replay functionality work as expected.

## 3. Techniques to Investigate
All experiments will run in Python 3.11 with the `gymnasium[box2d]` extras, PyTorch 2.3, NumPy, `tqdm`, and TensorBoard. Optional integrations include Weights & Biases for experiment tracking. The repository will reserve `src/agents` for DQN, PPO, and shared modules; `src/env` for Gymnasium wrappers and reward shaping hooks; `src/utils` for replay buffers, schedulers, and logging helpers; `experiments` for YAML configuration files and checkpoints; and `reports` for notebooks, plots, and interim writeups.

Function approximation will begin with a Deep Q-Network whose multilayer perceptron maps the state vector through two 256-unit ReLU layers to the action logits. The plan includes Xavier initialization, an Adam optimizer with learning rate 3e-4, Huber loss, gradient clipping at 5.0, and a target network updated every 1,000 gradient steps. The experience replay buffer will store 50,000 transitions in a cyclic queue, sample uniformly sized batches of 128 experiences for the baseline, and support prioritized replay (alpha 0.6 with beta annealing) for ablation studies.

Beyond the mandatory techniques, the study will consider reward shaping for vertical velocity damping and landing gear contact, curriculum learning that gradually increases gravity or wind once the agent achieves an average reward above 200, and PPO as an alternative policy-gradient baseline. Exploration strategies such as cosine-scheduled epsilon-greedy, parameter noise, and entropy bonuses will also be piloted to compare sample efficiency.

## 4. Evaluation Plan
Each configuration will be trained for 500 to 1,000 episodes across three random seeds. The primary metrics include the moving average of episodic reward over a 100-episode window, landing success rate, crash frequency, and a proxy for fuel usage derived from the sum of thruster magnitudes. Logging will rely on TensorBoard summaries and CSV exports stored under `experiments/logs`, with saved checkpoints enabling reproducibility. The first milestone is to reproduce the Gymnasium reference reward of roughly 200 points before layering advanced methods.

## 5. Anticipated Risks & Mitigations
Box2D dependency issues are the most immediate concern, so the team will test installation on macOS early and document the required Homebrew and pip commands, with Docker as a fallback for CI. Training instability will be mitigated through deterministic seeding, gradient clipping, replay buffer monitoring, and optional reward normalization. Compute limitations are addressed by keeping configurations CPU friendly until profiling indicates whether GPU acceleration is worthwhile. To prevent scope creep, the feature set will be locked after the baseline and two advanced variations are validated, leaving extra ideas for the final report stretch goals.

## 6. Immediate Next Actions (Week of 15 Nov)
1. Create shared repo (GitHub Classroom or private repo).
2. Set up Python environment + Gymnasium (verify `LunarLander-v3` renders locally).
3. Implement reusable replay buffer + DQN skeleton (no tuning).
4. Run smoke tests on CartPole to validate training loop/logging.
5. Draft initial-report text (1-2 pages) using this plan, include anticipated challenges section.

This document fulfills the "first part" requirement described in `Project_Description.docx` by specifying the environment, planned methods, tooling, and anticipated issues needed for the initial submission. Update it as decisions are finalized and include it (or its distilled version) in the official initial report deliverable.

