import torch
import numpy as np
from stable_baselines3.common.buffers import DictReplayBuffer, DictReplayBufferSamples
import torch.nn.functional as F
from stable_baselines3.dqn.dqn import DQN

class CpuDictReplayBuffer(DictReplayBuffer):
    def __init__(self, *args, sample_device=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_device = torch.device(sample_device) if sample_device else None

    @staticmethod
    def _to_device(batch: DictReplayBufferSamples, device: torch.device):
        """Return a *new* DictReplayBufferSamples living on `device`."""
        obs        = {k: v.to(device) for k, v in batch.observations.items()}
        next_obs   = {k: v.to(device) for k, v in batch.next_observations.items()}
        actions    = batch.actions.to(device)
        rewards    = batch.rewards.to(device)
        dones      = batch.dones.to(device)
        return DictReplayBufferSamples(obs, actions, next_obs, dones, rewards)

    # override -----------------------------------------------------------
    def sample(self, batch_size: int, env=None, device=None):
        batch = super().sample(batch_size, env=env)   # still on CPU

        target_device = device or self.sample_device
        if target_device is not None:
            batch = self._to_device(batch, target_device)
        return batch
    

class PERBatch:
    """Simple container for PER samples (compatible with our custom train())."""
    def __init__(self, observations, actions, next_observations, dones, rewards, weights, indices):
        self.observations = observations
        self.actions = actions
        self.next_observations = next_observations
        self.dones = dones
        self.rewards = rewards
        self.weights = weights
        self.indices = indices


class PrioritizedCpuDictReplayBuffer(DictReplayBuffer):
    def __init__(
        self,
        *args,
        alpha=0.6,
        beta0=0.4,
        beta_steps=1_000_000,
        eps=1e-6,
        sample_device=None,
        **kwargs,
    ):
        """
        alpha: priority exponent (0 => uniform, 1 => fully prioritized)
        beta0: initial IS-correction exponent
        beta_steps: steps to anneal beta to 1.0 (linear)
        eps: small floor to keep non-zero priority
        sample_device: torch device for returned batches (e.g., 'cuda:0')

        Important:
        Priorities are stored per transition slot: (buffer_pos, env_idx),
        i.e. shape (buffer_size, n_envs), not just (buffer_size,).
        """
        super().__init__(*args, **kwargs)

        self.alpha = float(alpha)
        self.beta0 = float(beta0)
        self.beta_steps = int(beta_steps)
        self.beta_updates = 0
        self.beta = self.beta0
        self.eps = float(eps)
        self.sample_device = torch.device(sample_device) if sample_device else None

        # One priority per stored transition, not per row
        self.priorities = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.max_priority = 1.0

    def add(self, *args, **kwargs):
        """
        SB3 writes one replay-buffer row self.pos containing transitions for all envs.
        We assign max priority to every env slot in that row.
        """
        idx = self.pos
        super().add(*args, **kwargs)
        self.priorities[idx, :] = self.max_priority

    def _current_beta(self):
        if self.beta_steps > 0:
            frac = min(1.0, self.beta_updates / float(self.beta_steps))
            self.beta = self.beta0 + frac * (1.0 - self.beta0)
        return self.beta

    def _get_samples_for_indices(self, batch_indices: np.ndarray, env_indices: np.ndarray, env=None):
        """
        Fetch exactly the transitions identified by (batch_indices, env_indices).

        This mirrors SB3's multi-env replay-buffer indexing pattern:
        obs[key][batch_indices, env_indices, ...]
        """
        # Normalize observations if needed
        obs_ = self._normalize_obs(
            {key: obs[batch_indices, env_indices, :] for key, obs in self.observations.items()},
            env,
        )
        next_obs_ = self._normalize_obs(
            {key: obs[batch_indices, env_indices, :] for key, obs in self.next_observations.items()},
            env,
        )

        assert isinstance(obs_, dict)
        assert isinstance(next_obs_, dict)

        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        actions = self.to_torch(self.actions[batch_indices, env_indices])

        dones = self.to_torch(
            self.dones[batch_indices, env_indices] * (1 - self.timeouts[batch_indices, env_indices])
        ).reshape(-1, 1)

        rewards = self.to_torch(
            self._normalize_reward(self.rewards[batch_indices, env_indices].reshape(-1, 1), env)
        )

        return DictReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
        )

    def sample(self, batch_size: int, env=None, device=None):
        """
        Sample PER minibatch over individual transition slots (buffer_pos, env_idx).
        Returns flat indices into priorities.reshape(-1), so update_priorities() can
        update the exact sampled transitions.
        """
        # Number of valid replay rows
        valid_rows = self.buffer_size if self.full else self.pos
        assert valid_rows > 0, "Cannot sample from an empty buffer."

        # Valid priorities over rows actually written so far
        valid_prios_2d = self.priorities[:valid_rows, :]   # shape: (valid_rows, n_envs)
        flat_prios = valid_prios_2d.reshape(-1).copy()     # shape: (valid_rows * n_envs,)

        if flat_prios.max() == 0.0:
            flat_prios[:] = 1.0

        probs = flat_prios ** self.alpha
        probs /= probs.sum()

        n_valid_transitions = flat_prios.shape[0]
        replace = n_valid_transitions < batch_size

        # Sample flat transition indices, then recover row/env coordinates
        flat_indices = np.random.choice(
            n_valid_transitions,
            size=batch_size,
            p=probs,
            replace=replace,
        )

        batch_indices, env_indices = np.unravel_index(flat_indices, (valid_rows, self.n_envs))

        beta = self._current_beta()
        self.beta_updates += 1

        weights = (n_valid_transitions * probs[flat_indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.as_tensor(weights, dtype=torch.float32)

        raw_batch = self._get_samples_for_indices(batch_indices, env_indices, env=env)

        target_device = device or self.sample_device
        if target_device is not None:
            obs = {k: v.to(target_device) for k, v in raw_batch.observations.items()}
            next_obs = {k: v.to(target_device) for k, v in raw_batch.next_observations.items()}
            actions = raw_batch.actions.to(target_device)
            rewards = raw_batch.rewards.to(target_device)
            dones = raw_batch.dones.to(target_device)
            weights = weights.to(target_device)
        else:
            obs = raw_batch.observations
            next_obs = raw_batch.next_observations
            actions = raw_batch.actions
            rewards = raw_batch.rewards
            dones = raw_batch.dones

        return (
            PERBatch(
                obs,
                actions,
                next_obs,
                dones,
                rewards,
                weights,
                flat_indices,   # flat indices into priorities[:valid_rows, :].reshape(-1)
            ),
            (
                probs[flat_indices].mean() / probs.mean(),
                probs[flat_indices].std() / (probs.std() + 1e-12),
            ),
        )

    def update_priorities(self, indices: np.ndarray, new_priorities: np.ndarray):
        """
        indices: flat indices returned by sample(), referring to
                 priorities[:valid_rows, :].reshape(-1)

        Since flat indices are over the logical valid region, we map them back to
        (buffer_pos, env_idx) using the full (buffer_size, n_envs) layout.
        """
        idx = np.asarray(indices).reshape(-1)
        new_p = np.asarray(new_priorities, dtype=np.float32).reshape(-1)
        new_p = np.maximum(new_p, self.eps)

        row_idx, env_idx = np.unravel_index(idx, (self.buffer_size, self.n_envs))
        self.priorities[row_idx, env_idx] = new_p
        self.max_priority = max(self.max_priority, float(new_p.max()))


class PERDQN(DQN):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []

        for _ in range(gradient_steps):
            # --- Sample with PER: returns weights + indices
            replay_data, per_stats = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            obs       = replay_data.observations
            next_obs  = replay_data.next_observations
            actions   = replay_data.actions.long().flatten()
            rewards   = replay_data.rewards.flatten()
            dones     = replay_data.dones.flatten()
            weights   = replay_data.weights
            indices   = replay_data.indices

            with torch.no_grad():
                # --- VANILLA DQN TARGET: max over target net
                q_next_all = self.q_net_target(next_obs)            # shape [B, n_actions]
                next_q_max = q_next_all.max(dim=1)[0]               # max_a Q_target(s', a)
                target_q   = rewards + (1.0 - dones) * self.gamma * next_q_max

            # --- Current Q(s,a)
            q_all = self.q_net(obs)                                 # shape [B, n_actions]
            q_sa  = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

            # --- Per-sample Huber (smooth L1) loss
            td_errors = (target_q - q_sa).detach()
            loss_elements = F.smooth_l1_loss(q_sa, target_q, reduction='none')

            # --- PER: weight the loss by IS-weights
            loss = (weights * loss_elements).mean()

            # --- Optimize
            self.policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # (optional) torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm)
            self.policy.optimizer.step()

            # --- Update priorities (|TD-error|)
            if hasattr(self.replay_buffer, "update_priorities"):
                new_prios = td_errors.abs().cpu().numpy() + 1e-6
                self.replay_buffer.update_priorities(indices, new_prios)

            losses.append(loss.item())

            # --- Target network update (SB3 handles soft update via tau in parent class)
            self._on_step()  # preserves standard SB3 behavior (train_freq, target updates, etc.)

        # Log
        if len(losses) > 0:
            self.logger.record("train/loss", float(np.mean(losses)))
            self.logger.record("per/td_abs_mean_batch", float(td_errors.abs().mean().item()))
            self.logger.record("per/batch_mean_factor", float(per_stats[0]))
            self.logger.record("per/batch_std_factor", float(per_stats[1]))
