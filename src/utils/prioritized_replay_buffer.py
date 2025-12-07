import numpy as np
import random
from typing import Tuple, Any


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer (proportional variant).

    - Stores transitions: (state, action, reward, next_state, done)
    - Samples transitions with probability proportional to their priority.
    - Uses importance-sampling (IS) weights to correct bias when training.

    This is designed to be used with DQN.
    """

    def __init__(
        self,
        capacity: int = 50000,
        batch_size: int = 128,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-5,
    ):
        """
        Args:
            capacity: maximum number of transitions to store
            batch_size: minibatch size when sampling
            alpha: how much prioritization is used (0 = uniform, 1 = full PER)
            beta_start: starting value of beta for importance-sampling weights
            beta_frames: number of sampling steps over which beta -> 1
            epsilon: small constant added to TD error to avoid zero priority
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon

        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.num_steps = 0  # how many times we've sampled (for beta schedule)

    # --------------------------------------------------------------------- #
    # basic buffer API
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.buffer)

    def is_full(self) -> bool:
        return len(self.buffer) == self.capacity

    def clear(self) -> None:
        self.buffer = []
        self.priorities[:] = 0.0
        self.position = 0
        self.num_steps = 0

    # --------------------------------------------------------------------- #
    # adding transitions
    # --------------------------------------------------------------------- #
    def push(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        """
        Add transition to buffer and assign it a high priority.

        New transitions are given max priority so they are sampled soon.
        """
        transition = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        # max priority so far (default 1.0 if buffer is empty)
        max_prio = self.priorities[: len(self.buffer)].max() if self.buffer else 1.0
        if max_prio == 0.0:
            max_prio = 1.0

        self.priorities[self.position] = max_prio

        self.position = (self.position + 1) % self.capacity

    # --------------------------------------------------------------------- #
    # sampling with priorities
    # --------------------------------------------------------------------- #
    def _beta_by_step(self) -> float:
        """
        Linearly increase beta from beta_start to 1.0 over beta_frames steps.
        """
        return min(
            1.0,
            self.beta_start + (1.0 - self.beta_start) * self.num_steps / self.beta_frames,
        )

    def sample(
        self,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
               np.ndarray,
               np.ndarray]:
        """
        Sample a batch of transitions.

        Returns:
            batch: (states, actions, rewards, next_states, dones)
            indices: indices of sampled transitions
            weights: importance-sampling weights (shape: [batch_size])
        """
        if len(self.buffer) < self.batch_size:
            raise ValueError(
                f"Not enough transitions in buffer. Have {len(self.buffer)}, "
                f"need {self.batch_size}"
            )

        self.num_steps += 1

        # Compute sampling probabilities from priorities
        prios = self.priorities[: len(self.buffer)]
        scaled_prios = prios ** self.alpha
        probs = scaled_prios / scaled_prios.sum()

        indices = np.random.choice(
            len(self.buffer),
            self.batch_size,
            p=probs,
        )

        # importance-sampling weights
        beta = self._beta_by_step()
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalize to [0, 1]

        # gather transitions
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[idx] for idx in indices]
        )

        batch = (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

        return batch, indices, weights.astype(np.float32)

    # --------------------------------------------------------------------- #
    # updating priorities after learning step
    # --------------------------------------------------------------------- #
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities for sampled transitions using TD errors.

        Args:
            indices: indices in the buffer that were sampled
            td_errors: TD errors for those transitions (same length as indices)
        """
        td_errors = np.abs(td_errors) + self.epsilon
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = float(err)
