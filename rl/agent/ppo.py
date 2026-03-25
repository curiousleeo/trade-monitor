"""
ppo.py
──────
Pure-NumPy Proximal Policy Optimisation (PPO-Clip) with Actor-Critic.

Same learn() / predict() / save() / load() interface as DQNAgent so that
walk_forward.py and train.py can swap it in with a one-line change.

Architecture:
  Shared MLP backbone → policy head (direction logits)
                      → value head  (scalar state value)
                      → sizing head (sigmoid position size in [0, 1])

  Direction : Categorical(4)  — 0=Hold 1=Long 2=Short 3=Close
  Sizing    : Deterministic sigmoid (on-policy mean, no extra noise needed
              because categorical direction already drives exploration)

On-policy rollout:
  Collect n_steps env transitions → compute GAE advantages →
  run n_epochs of mini-batch gradient updates → repeat.
"""

import pickle
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ── Activation helpers ─────────────────────────────────────────────────────────

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def _relu_mask(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-8)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


# ── Actor-Critic network ───────────────────────────────────────────────────────

class ActorCritic:
    """
    Two shared hidden layers (ReLU) + three heads.
    Maintains Adam optimiser state internally.
    """

    def __init__(
        self,
        obs_dim:   int,
        n_actions: int,
        hidden:    Tuple[int, ...] = (256, 256),
        lr:        float = 3e-4,
    ):
        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        self.lr        = lr
        rng = np.random.default_rng(42)

        # ── Shared layers (orthogonal init, scale √2 for ReLU) ────────────────
        dims = [obs_dim] + list(hidden)
        self.Ws: List[np.ndarray] = []
        self.bs: List[np.ndarray] = []
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            W = rng.standard_normal((dims[i], dims[i+1])).astype(np.float32) * scale
            self.Ws.append(W)
            self.bs.append(np.zeros(dims[i+1], dtype=np.float32))

        feat = hidden[-1]

        # ── Heads (small init to keep logits near 0 at start) ─────────────────
        self.W_pi = (rng.standard_normal((feat, n_actions)) * 0.01).astype(np.float32)
        self.b_pi = np.zeros(n_actions, dtype=np.float32)
        self.W_v  = (rng.standard_normal((feat, 1)) * 0.01).astype(np.float32)
        self.b_v  = np.zeros(1, dtype=np.float32)
        self.W_sz = (rng.standard_normal((feat, 1)) * 0.01).astype(np.float32)
        self.b_sz = np.zeros(1, dtype=np.float32)

        # ── Adam state ────────────────────────────────────────────────────────
        self._t     = 0
        self._beta1 = 0.9
        self._beta2 = 0.999
        self._eps   = 1e-8

        n = len(self.Ws)
        self._mW = [np.zeros_like(w) for w in self.Ws]
        self._vW = [np.zeros_like(w) for w in self.Ws]
        self._mb = [np.zeros_like(b) for b in self.bs]
        self._vb = [np.zeros_like(b) for b in self.bs]

        for name in ("W_pi", "b_pi", "W_v", "b_v", "W_sz", "b_sz"):
            p = getattr(self, name)
            object.__setattr__(self, f"_m_{name}", np.zeros_like(p))
            object.__setattr__(self, f"_v_{name}", np.zeros_like(p))

    # ── Adam step ─────────────────────────────────────────────────────────────

    def _adam(
        self,
        param: np.ndarray,
        grad:  np.ndarray,
        m:     np.ndarray,
        v:     np.ndarray,
    ) -> np.ndarray:
        m[:] = self._beta1 * m + (1 - self._beta1) * grad
        v[:] = self._beta2 * v + (1 - self._beta2) * (grad ** 2)
        m_hat = m / (1 - self._beta1 ** self._t)
        v_hat = v / (1 - self._beta2 ** self._t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self._eps)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, obs: np.ndarray):
        """
        obs: (batch, obs_dim)  — must be 2-D float32.
        Returns (logits, values, sizes) each (batch,) or (batch, n_actions).
        Caches pre-activations and post-relu activations for backward.
        """
        obs = obs.astype(np.float32)
        pre_acts: List[np.ndarray] = []   # pre-ReLU z = x@W+b
        post_acts: List[np.ndarray] = [obs]  # post-ReLU h (starts with input)

        x = obs
        for W, b in zip(self.Ws, self.bs):
            z = x @ W + b
            pre_acts.append(z)
            x = _relu(z)
            post_acts.append(x)

        self._pre_acts  = pre_acts
        self._post_acts = post_acts

        feat   = x
        logits     = feat @ self.W_pi + self.b_pi               # (batch, n_actions)
        values     = (feat @ self.W_v  + self.b_v).squeeze(-1)  # (batch,)
        raw_sz_out = (feat @ self.W_sz + self.b_sz).squeeze(-1)  # (batch,) pre-sigmoid
        sizes      = _sigmoid(raw_sz_out)                        # (batch,)

        self._raw_sz_out = raw_sz_out   # cache for backward
        return logits, values, sizes

    # ── PPO update ────────────────────────────────────────────────────────────

    def update(
        self,
        obs:          np.ndarray,   # (batch, obs_dim)
        actions:      np.ndarray,   # (batch,) int
        old_log_probs: np.ndarray,  # (batch,) float
        advantages:   np.ndarray,   # (batch,) float — GAE
        returns:      np.ndarray,   # (batch,) float — discounted returns
        clip_range:   float = 0.2,
        vf_coef:      float = 0.5,
        ent_coef:     float = 0.01,
        sz_coef:      float = 0.1,  # sizing head loss coefficient
    ) -> dict:
        """One mini-batch gradient step. Returns loss info dict."""
        batch = obs.shape[0]
        self._t += 1

        logits, values, sizes = self.forward(obs)
        probs         = _softmax(logits)                                 # (B, A)
        log_probs_all = np.log(probs + 1e-8)                            # (B, A)
        log_probs_new = log_probs_all[np.arange(batch), actions]        # (B,)

        # Normalise advantages per mini-batch (reduces variance)
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO clipped surrogate
        ratio     = np.exp(np.clip(log_probs_new - old_log_probs, -10, 10))
        ratio_clip = np.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)
        surr1 = ratio      * adv
        surr2 = ratio_clip * adv
        policy_loss = -np.minimum(surr1, surr2).mean()

        # Value loss (MSE, unclipped for simplicity)
        value_loss = 0.5 * ((values - returns) ** 2).mean()

        # Entropy bonus  H = -Σ p log p
        entropy     = -(probs * log_probs_all).sum(axis=-1).mean()
        entropy_loss = -entropy

        # ── Backward pass ─────────────────────────────────────────────────────

        # 1. Policy gradient w.r.t. log_prob_new
        #    d(-min(s1,s2))/d(log_prob):
        #      when s1 is active min → grad = -adv * ratio
        #      when s2 is active min AND ratio clipped → grad = 0
        is_clipped = (ratio < 1.0 - clip_range) | (ratio > 1.0 + clip_range)
        s1_active  = (surr1 <= surr2).astype(np.float32)
        grad_flows = s1_active + (1.0 - s1_active) * (1.0 - is_clipped.astype(np.float32))
        d_logprob  = -adv * ratio * grad_flows / batch           # (B,)

        # d(log_prob_i)/d(logits_i[k]) = 1[k==a_i] - p_i[k]
        d_logits_pi = -probs.copy()                              # (B, A)
        d_logits_pi[np.arange(batch), actions] += 1.0
        grad_logits = d_logprob[:, np.newaxis] * d_logits_pi    # (B, A)

        # 2. Entropy gradient  d(-H)/d(logits)
        #    = p*(log_p + 1) - p*Σ_j[p_j*(log_p_j+1)]
        log_p_plus1 = log_probs_all + 1.0
        sum_term    = (probs * log_p_plus1).sum(axis=-1, keepdims=True)
        d_ent       = probs * (log_p_plus1 - sum_term)
        grad_logits += ent_coef * d_ent / batch

        # 3. Value gradient  d(0.5*(v-R)^2)/d(v) = v - R
        grad_v = (values - returns) * vf_coef / batch            # (B,)

        # 4. Sizing gradient  (policy gradient on deterministic size)
        #    Only applies to Long (1) and Short (2) actions — sizing is irrelevant
        #    for Hold and Close.
        #    We use RAW advantages (not batch-normalised) so that absolute conviction
        #    matters: a strong signal (large |adv|) → large size, weak → small size.
        #    sizing_loss = -mean(adv_scaled * log(size + ε) * is_trade)
        #    When adv > 0 (good trade):  push size UP   (bet bigger on strong signals)
        #    When adv < 0 (bad trade):   push size DOWN (bet smaller on weak signals)
        is_trade  = ((actions == 1) | (actions == 2)).astype(np.float32)  # (B,)
        sz_eps    = 1e-6
        # Scale raw advantages to [-1, 1] range to keep gradient magnitudes stable
        adv_scale = np.abs(advantages).max() + 1e-8
        adv_raw   = advantages / adv_scale                              # (B,) in [-1,1]
        # d(sizing_loss)/d(size) = -adv_raw / (size + ε) * is_trade / batch
        d_sz_d_size = -adv_raw * is_trade / (sizes + sz_eps) * sz_coef / batch
        # d(size)/d(raw_sz_out) = size * (1 - size)  [sigmoid derivative]
        d_sz_d_raw  = d_sz_d_size * sizes * (1.0 - sizes)              # (B,)

        sizing_loss = float(-np.mean(adv_raw * np.log(sizes + sz_eps) * is_trade))

        # 5. Backprop from heads to feature layer
        feat     = self._post_acts[-1]                           # (B, feat)
        grad_W_pi = feat.T @ grad_logits                         # (feat, A)
        grad_b_pi = grad_logits.sum(axis=0)
        grad_W_v  = feat.T @ grad_v[:, np.newaxis]
        grad_b_v  = grad_v.sum(keepdims=True).reshape(1)
        grad_W_sz = feat.T @ d_sz_d_raw[:, np.newaxis]
        grad_b_sz = d_sz_d_raw.sum(keepdims=True).reshape(1)

        # Gradient into feat (from all three heads)
        delta = (grad_logits @ self.W_pi.T
                 + grad_v[:, np.newaxis] * self.W_v.T
                 + d_sz_d_raw[:, np.newaxis] * self.W_sz.T)

        # 5. Backprop through shared layers (reverse order)
        grad_Ws: List[np.ndarray] = [None] * len(self.Ws)
        grad_bs: List[np.ndarray] = [None] * len(self.bs)
        for i in reversed(range(len(self.Ws))):
            delta     = delta * _relu_mask(self._pre_acts[i])
            grad_Ws[i] = self._post_acts[i].T @ delta
            grad_bs[i] = delta.sum(axis=0)
            delta      = delta @ self.Ws[i].T

        # 6. Apply Adam updates
        for i in range(len(self.Ws)):
            self.Ws[i] = self._adam(self.Ws[i], grad_Ws[i], self._mW[i], self._vW[i])
            self.bs[i] = self._adam(self.bs[i], grad_bs[i], self._mb[i], self._vb[i])

        self.W_pi = self._adam(self.W_pi, grad_W_pi,
                               self._m_W_pi, self._v_W_pi)
        self.b_pi = self._adam(self.b_pi, grad_b_pi,
                               self._m_b_pi, self._v_b_pi)
        self.W_v  = self._adam(self.W_v,  grad_W_v,
                               self._m_W_v,  self._v_W_v)
        self.b_v  = self._adam(self.b_v,  grad_b_v,
                               self._m_b_v,  self._v_b_v)
        self.W_sz = self._adam(self.W_sz, grad_W_sz,
                               self._m_W_sz, self._v_W_sz)
        self.b_sz = self._adam(self.b_sz, grad_b_sz,
                               self._m_b_sz, self._v_b_sz)

        return {
            "policy_loss": float(policy_loss),
            "value_loss":  float(value_loss),
            "entropy":     float(entropy),
            "sizing_loss": sizing_loss,
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    def get_action(
        self,
        obs:          np.ndarray,   # (obs_dim,)
        deterministic: bool = False,
    ):
        """Returns (direction, size, log_prob, value)."""
        obs2d   = obs[np.newaxis].astype(np.float32)
        logits, value, size = self.forward(obs2d)
        logits = logits[0]
        probs  = _softmax(logits[np.newaxis])[0]

        if deterministic:
            direction = int(np.argmax(probs))
        else:
            direction = int(np.random.choice(self.n_actions, p=probs))

        log_prob = float(np.log(probs[direction] + 1e-8))
        return direction, float(size[0]), log_prob, float(value[0])


# ── Rollout buffer ─────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores one rollout's worth of on-policy experience."""

    def __init__(self, n_steps: int, obs_dim: int):
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.reset()

    def reset(self):
        self.obs       = np.zeros((self.n_steps, self.obs_dim), dtype=np.float32)
        self.actions   = np.zeros(self.n_steps, dtype=np.int32)
        self.sizes     = np.zeros(self.n_steps, dtype=np.float32)
        self.log_probs = np.zeros(self.n_steps, dtype=np.float32)
        self.rewards   = np.zeros(self.n_steps, dtype=np.float32)
        self.dones     = np.zeros(self.n_steps, dtype=np.float32)
        self.values    = np.zeros(self.n_steps, dtype=np.float32)
        self._ptr      = 0

    def add(
        self,
        obs:      np.ndarray,
        action:   int,
        size:     float,
        log_prob: float,
        reward:   float,
        done:     bool,
        value:    float,
    ):
        i = self._ptr
        self.obs[i]       = obs
        self.actions[i]   = action
        self.sizes[i]     = size
        self.log_probs[i] = log_prob
        self.rewards[i]   = reward
        self.dones[i]     = float(done)
        self.values[i]    = value
        self._ptr += 1

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """Compute GAE advantages and discounted returns in-place."""
        advantages = np.zeros(self.n_steps, dtype=np.float32)
        last_gae   = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = last_value
                next_done  = 0.0
            else:
                next_value = self.values[t + 1]
                next_done  = self.dones[t + 1]
            delta    = (self.rewards[t]
                        + gamma * next_value * (1.0 - next_done)
                        - self.values[t])
            last_gae = delta + gamma * gae_lambda * (1.0 - next_done) * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns    = advantages + self.values

    def get_samples(self):
        """Return all stored data as dict of arrays."""
        return {
            "obs":       self.obs,
            "actions":   self.actions,
            "sizes":     self.sizes,
            "log_probs": self.log_probs,
            "advantages": self.advantages,
            "returns":   self.returns,
        }


# ── Frame stacking buffer ──────────────────────────────────────────────────────

class FrameBuffer:
    """
    Maintains a rolling window of the last `seq_len` market observations.

    The env returns obs = [market(market_dim), pos_state(pos_dim)].
    We stack the last seq_len market frames and append the CURRENT pos_state,
    giving the network temporal context without lookahead:
        stacked = [mkt_t-23, mkt_t-22, ..., mkt_t-1, mkt_t, pos_t]
                = (seq_len * market_dim + pos_dim,)

    On episode reset call reset() to clear history (fills with zeros).
    """

    def __init__(self, seq_len: int, market_dim: int, pos_dim: int = 2):
        self.seq_len    = seq_len
        self.market_dim = market_dim
        self.pos_dim    = pos_dim
        self.stacked_dim = seq_len * market_dim + pos_dim
        self._buf = np.zeros((seq_len, market_dim), dtype=np.float32)

    def reset(self):
        self._buf[:] = 0.0

    def push_and_get(self, obs: np.ndarray) -> np.ndarray:
        """
        Push new raw obs, return stacked observation.
        obs: (market_dim + pos_dim,)
        """
        market = obs[:self.market_dim]
        pos    = obs[self.market_dim:]
        # Shift buffer left, add new frame at the end
        self._buf[:-1] = self._buf[1:]
        self._buf[-1]  = market
        return np.concatenate([self._buf.ravel(), pos]).astype(np.float32)


# ── PPO Agent ──────────────────────────────────────────────────────────────────

class PPOAgent:
    """
    Drop-in replacement for DQNAgent.

    Key differences vs DQN:
      - On-policy rollouts (no replay buffer, no experience replay)
      - Actor-Critic (learns value function alongside policy)
      - Continuous sizing head trained jointly
      - Better exploration via entropy bonus (no epsilon-greedy)
    """

    def __init__(
        self,
        obs_dim:    int,           # raw env obs dim (market_dim + pos_dim)
        n_actions:  int  = 4,
        hidden:     Tuple[int, ...] = (256, 256),
        lr:         float = 3e-4,
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        vf_coef:    float = 0.5,
        ent_coef:   float = 0.01,
        n_steps:    int   = 2048,
        n_epochs:   int   = 10,
        batch_size: int   = 64,
        seq_len:    int   = 1,     # frame stacking depth; 1 = no stacking
        pos_dim:    int   = 2,     # position-state dims appended by env
        **kwargs,
    ):
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef    = vf_coef
        self.ent_coef   = ent_coef
        self.n_steps    = n_steps
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.n_actions  = n_actions
        self.seq_len    = seq_len
        self.pos_dim    = pos_dim

        market_dim = obs_dim - pos_dim
        self._frame_buf  = FrameBuffer(seq_len, market_dim, pos_dim)
        self._eval_buf   = FrameBuffer(seq_len, market_dim, pos_dim)

        net_obs_dim = self._frame_buf.stacked_dim
        self.net    = ActorCritic(net_obs_dim, n_actions, hidden, lr)
        self.buffer = RolloutBuffer(n_steps, net_obs_dim)

    # ── Training ──────────────────────────────────────────────────────────────

    def learn(
        self,
        env,
        total_timesteps:  int,
        eval_env          = None,
        eval_freq:        int  = 20_000,
        n_eval_episodes:  int  = 3,
        save_path:        str  = None,
        verbose:          int  = 1,
    ):
        """
        Collect rollouts and update until total_timesteps steps are done.
        Prints ASCII progress bar identical to DQNAgent.
        """
        raw_obs, _ = env.reset()
        self._frame_buf.reset()
        obs = self._frame_buf.push_and_get(np.asarray(raw_obs, dtype=np.float32))

        step       = 0
        update_num = 0
        last_eval  = 0
        best_eval  = -np.inf
        t_start    = time.time()
        bar_width  = 30

        while step < total_timesteps:
            # ── Collect n_steps rollout ──────────────────────────────────────
            self.buffer.reset()
            for _ in range(self.n_steps):
                direction, size, log_prob, value = self.net.get_action(obs)
                action_dict = {"direction": direction, "sizing": float(size)}

                raw_next, reward, terminated, truncated, info = env.step(action_dict)
                done = terminated or truncated

                self.buffer.add(obs, direction, size, log_prob,
                                float(reward), done, value)

                if done:
                    raw_obs, _ = env.reset()
                    self._frame_buf.reset()
                    obs = self._frame_buf.push_and_get(
                        np.asarray(raw_obs, dtype=np.float32))
                else:
                    obs = self._frame_buf.push_and_get(
                        np.asarray(raw_next, dtype=np.float32))

                step += 1

                # Progress bar
                if verbose and step % 500 == 0:
                    pct  = step / total_timesteps
                    fill = int(bar_width * pct)
                    bar  = "#" * fill + "-" * (bar_width - fill)
                    elapsed = time.time() - t_start
                    eta     = (elapsed / pct - elapsed) if pct > 0 else 0
                    print(
                        f"\r[{bar}] {step:>7,}/{total_timesteps:,}  "
                        f"{pct*100:5.1f}%  "
                        f"elapsed {int(elapsed//60):02d}:{int(elapsed%60):02d}  "
                        f"ETA {int(eta//60):02d}:{int(eta%60):02d}",
                        end="", flush=True,
                    )

                if step >= total_timesteps:
                    break

            # ── Compute GAE ───────────────────────────────────────────────────
            _, last_value, _ = self.net.forward(obs[np.newaxis])
            self.buffer.compute_returns_and_advantages(
                float(last_value[0]), self.gamma, self.gae_lambda
            )
            data = self.buffer.get_samples()

            # ── PPO mini-batch updates ────────────────────────────────────────
            n      = len(data["obs"])
            losses = {"policy_loss": [], "value_loss": [], "entropy": [], "sizing_loss": []}
            for _ in range(self.n_epochs):
                idx = np.random.permutation(n)
                for start in range(0, n, self.batch_size):
                    mb = idx[start: start + self.batch_size]
                    info_l = self.net.update(
                        obs          = data["obs"][mb],
                        actions      = data["actions"][mb],
                        old_log_probs= data["log_probs"][mb],
                        advantages   = data["advantages"][mb],
                        returns      = data["returns"][mb],
                        clip_range   = self.clip_range,
                        vf_coef      = self.vf_coef,
                        ent_coef     = self.ent_coef,
                    )
                    for k in losses:
                        losses[k].append(info_l[k])

            update_num += 1

            # ── Periodic evaluation ───────────────────────────────────────────
            if eval_env is not None and step - last_eval >= eval_freq:
                mean_r = self._evaluate(eval_env, n_eval_episodes)
                pl     = np.mean(losses["policy_loss"])
                vl     = np.mean(losses["value_loss"])
                ent    = np.mean(losses["entropy"])
                print(
                    f"\n  step={step:,}  eval_reward={mean_r:.4f}  "
                    f"policy_loss={pl:.4f}  value_loss={vl:.4f}  "
                    f"entropy={ent:.4f}",
                    flush=True,
                )
                last_eval = step
                if save_path and mean_r > best_eval:
                    best_eval = mean_r
                    self.save(str(Path(save_path) / "best_model.pkl"))

        if verbose:
            print(flush=True)

        # Save final model
        if save_path:
            self.save(str(Path(save_path) / "final_model.pkl"))

        return self

    def _evaluate(self, env, n_episodes: int) -> float:
        """Run n_episodes deterministically. Returns mean total reward."""
        total = 0.0
        for _ in range(n_episodes):
            raw_obs, _ = env.reset()
            self._eval_buf.reset()
            obs  = self._eval_buf.push_and_get(np.asarray(raw_obs, dtype=np.float32))
            done = False
            ep_r = 0.0
            while not done:
                direction, size, _, _ = self.net.get_action(obs, deterministic=True)
                raw_next, r, terminated, truncated, _ = env.step(
                    {"direction": direction, "sizing": float(size)}
                )
                ep_r += r
                done  = terminated or truncated
                if not done:
                    obs = self._eval_buf.push_and_get(
                        np.asarray(raw_next, dtype=np.float32))
            total += ep_r
        return total / max(n_episodes, 1)

    # ── Inference ─────────────────────────────────────────────────────────────

    def reset_obs_buffer(self):
        """Call after env.reset() before the first predict() in an episode."""
        self._eval_buf.reset()

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> dict:
        """
        Returns action dict {"direction": int, "sizing": float}.
        Maintains an internal frame buffer across calls within an episode.
        Call reset_obs_buffer() after env.reset() to start a new episode.
        """
        stacked = self._eval_buf.push_and_get(np.asarray(obs, dtype=np.float32))
        direction, size, _, _ = self.net.get_action(stacked, deterministic=deterministic)
        return {"direction": direction, "sizing": float(size)}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "PPOAgent":
        with open(path, "rb") as f:
            return pickle.load(f)
