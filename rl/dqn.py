"""
dqn.py
──────
Pure-numpy DQN + continuous sizing head. No PyTorch/TensorFlow required.

Architecture:
  Shared trunk MLP → Q-head (4 direction Q-values, linear)
                   → Sizing head (1 sigmoid output, trained via REINFORCE)

Action space (Dict):
  direction: argmax(Q-values)         — 0=Hold, 1=Long, 2=Short, 3=Close
  sizing:    sigma(sizing_head(obs))  — [0,1], fraction of equity to risk

Training:
  DQN loss   — Huber(TD error) on direction head (every train_freq steps)
  Sizing loss — REINFORCE: -reward * log(sizing) on entry steps only
                (only steps where direction ∈ {Long=1, Short=2})
  Combined:  total_loss = dqn_loss + SIZING_LOSS_WEIGHT * sizing_loss
"""

import numpy as np
import pickle
from collections import deque
from pathlib import Path


SIZING_LOSS_WEIGHT = 0.1   # weight of policy-gradient sizing loss vs DQN loss


# ── Activation helpers ────────────────────────────────────────────────────────

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(np.float32)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# ── Neural network (MLP) ─────────────────────────────────────────────────────

class MLP:
    """
    Multi-layer perceptron with ReLU hidden activations, linear output.
    Weights initialised with He initialisation.
    """

    def __init__(self, layer_sizes: list[int], seed: int = 42):
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            w = rng.standard_normal((fan_in, layer_sizes[i + 1])).astype(np.float32)
            w *= np.sqrt(2.0 / fan_in)                      # He init
            b = np.zeros(layer_sizes[i + 1], dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)

        self._cache = None    # stores activations for backward pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, input_dim)  →  output: (batch, output_dim)"""
        self._cache = []
        self._input = x
        h = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ w + b
            if i < len(self.weights) - 1:           # hidden layers → ReLU
                h = relu(z)
                self._cache.append((h, z))
            else:                                    # output layer → linear
                h = z
        return h

    def backward(self, x: np.ndarray, grad_out: np.ndarray) -> tuple:
        """
        Backprop through the network.
        Returns (grad_weights, grad_biases) as lists.
        """
        grad_w = []
        grad_b = []

        acts = [x]
        for h, z in self._cache:
            acts.append(h)

        delta = grad_out
        for i in reversed(range(len(self.weights))):
            inp   = acts[i]
            gw    = inp.T @ delta / len(x)
            gb    = delta.mean(axis=0)
            grad_w.insert(0, gw)
            grad_b.insert(0, gb)
            if i > 0:
                delta = delta @ self.weights[i].T
                h, z  = self._cache[i - 1]
                delta *= relu_grad(z)

        return grad_w, grad_b

    def copy_from(self, other: "MLP"):
        """Hard-copy weights from another MLP (target network update)."""
        for i in range(len(self.weights)):
            self.weights[i] = other.weights[i].copy()
            self.biases[i]  = other.biases[i].copy()

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"weights": self.weights, "biases": self.biases}, f)

    @classmethod
    def load(cls, path: str) -> "MLP":
        with open(path, "rb") as f:
            d = pickle.load(f)
        net = cls.__new__(cls)
        net.weights = d["weights"]
        net.biases  = d["biases"]
        net._cache  = None
        return net


# ── Adam optimiser ────────────────────────────────────────────────────────────

class Adam:
    def __init__(self, lr: float = 1e-4, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.t     = 0
        self.m_w   = None
        self.v_w   = None
        self.m_b   = None
        self.v_b   = None

    def step(self, net: MLP, grad_w: list, grad_b: list):
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in net.weights]
            self.v_w = [np.zeros_like(w) for w in net.weights]
            self.m_b = [np.zeros_like(b) for b in net.biases]
            self.v_b = [np.zeros_like(b) for b in net.biases]

        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)

        for i in range(len(net.weights)):
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grad_w[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * grad_w[i] ** 2
            net.weights[i] -= lr_t * self.m_w[i] / (np.sqrt(self.v_w[i]) + self.eps)

            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grad_b[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * grad_b[i] ** 2
            net.biases[i] -= lr_t * self.m_b[i] / (np.sqrt(self.v_b[i]) + self.eps)


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Stores (state, direction, sizing, reward, next_state, done) tuples.
    sizing is stored so we can compute REINFORCE gradient during training.
    """

    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, state, direction: int, sizing: float,
             reward: float, next_state, done: bool):
        self.buf.append((
            state.astype(np.float32),
            int(direction),
            float(sizing),
            float(reward),
            next_state.astype(np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        idx   = np.random.choice(len(self.buf), batch_size, replace=False)
        batch = [self.buf[i] for i in idx]
        s, a, sz, r, ns, d = zip(*batch)
        return (
            np.stack(s),
            np.array(a,  dtype=np.int32),
            np.array(sz, dtype=np.float32),
            np.array(r,  dtype=np.float32),
            np.stack(ns),
            np.array(d,  dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


# ── DQN agent ─────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    DQN with:
      - experience replay
      - target network (hard update every target_update steps)
      - epsilon-greedy exploration on direction
      - Huber loss (smooth L1) for direction Q-values
      - REINFORCE policy gradient for continuous sizing head
      - exploration noise on sizing during training

    Outputs Dict actions compatible with TradingEnv:
      {"direction": int, "sizing": np.ndarray([float])}
    """

    def __init__(
        self,
        obs_dim:         int,
        n_actions:       int   = 4,
        hidden:          list[int] = None,
        lr:              float = 1e-4,
        gamma:           float = 0.99,
        buffer_size:     int   = 100_000,
        batch_size:      int   = 256,
        learning_starts: int   = 5_000,
        train_freq:      int   = 4,
        target_update:   int   = 1_000,
        eps_start:       float = 1.0,
        eps_end:         float = 0.02,
        eps_decay_steps: int   = 100_000,
        sizing_noise:    float = 0.1,   # Gaussian noise on sizing during exploration
        seed:            int   = 42,
    ):
        if hidden is None:
            hidden = [256, 256]
        np.random.seed(seed)

        layer_sizes     = [obs_dim] + hidden + [n_actions]
        self.online     = MLP(layer_sizes, seed=seed)
        self.target     = MLP(layer_sizes, seed=seed + 1)
        self.target.copy_from(self.online)

        # Sizing head: obs → 128 → 64 → 1 (linear; sigmoid applied at inference)
        sizing_layers   = [obs_dim, 128, 64, 1]
        self.sizing_net = MLP(sizing_layers, seed=seed + 2)

        self.optim       = Adam(lr=lr)
        self.sizing_optim = Adam(lr=lr * 0.5)   # slower learning rate for sizing
        self.buffer      = ReplayBuffer(buffer_size)

        self.gamma           = gamma
        self.batch_size      = batch_size
        self.learning_starts = learning_starts
        self.train_freq      = train_freq
        self.target_update   = target_update
        self.n_actions       = n_actions
        self.sizing_noise    = sizing_noise

        self.eps_start       = eps_start
        self.eps_end         = eps_end
        self.eps_decay_steps = eps_decay_steps

        self.total_steps     = 0
        self.losses          = []

    # ── Epsilon schedule ──────────────────────────────────────────────────────

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self.total_steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    # ── Action selection ──────────────────────────────────────────────────────

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> dict:
        """
        Returns {"direction": int, "sizing": np.ndarray([float])}.
        During exploration: random direction + noisy sizing.
        """
        # Direction
        if not deterministic and np.random.random() < self.epsilon:
            direction = np.random.randint(self.n_actions)
        else:
            q = self.online.forward(obs[None])[0]
            direction = int(np.argmax(q))

        # Sizing
        raw  = self.sizing_net.forward(obs[None])[0, 0]   # scalar, linear
        size = float(sigmoid(raw))

        if not deterministic and self.sizing_noise > 0:
            size = float(np.clip(size + np.random.normal(0, self.sizing_noise), 0.0, 1.0))

        return {
            "direction": direction,
            "sizing":    np.array([size], dtype=np.float32),
        }

    # ── Training step ─────────────────────────────────────────────────────────

    def _huber_loss_grad(self, td_error: np.ndarray, delta: float = 1.0):
        abs_err = np.abs(td_error)
        grad    = np.where(abs_err <= delta, td_error, delta * np.sign(td_error))
        loss    = np.where(abs_err <= delta,
                           0.5 * td_error ** 2,
                           delta * (abs_err - 0.5 * delta)).mean()
        return grad, float(loss)

    def learn_step(self):
        if len(self.buffer) < self.learning_starts:
            return None

        s, a, sz, r, ns, d = self.buffer.sample(self.batch_size)

        # ── DQN loss ──────────────────────────────────────────────────────────
        q_next   = self.target.forward(ns)
        q_target = r + self.gamma * (1 - d) * q_next.max(axis=1)

        q_pred_all = self.online.forward(s)
        q_pred     = q_pred_all[np.arange(self.batch_size), a]

        td_error      = q_pred - q_target
        loss_grad, dqn_loss = self._huber_loss_grad(td_error)

        grad_out = np.zeros_like(q_pred_all)
        grad_out[np.arange(self.batch_size), a] = loss_grad

        gw, gb = self.online.backward(s, grad_out)
        self.optim.step(self.online, gw, gb)

        # ── Sizing loss (REINFORCE, entry steps only) ─────────────────────────
        # Only train sizing when agent actually entered a position (LONG=1, SHORT=2)
        entry_mask = (a == 1) | (a == 2)   # boolean (batch,)
        sizing_loss = 0.0

        if entry_mask.sum() > 0:
            s_entry  = s[entry_mask]
            sz_entry = sz[entry_mask]                    # what sizing was chosen
            r_entry  = r[entry_mask]                     # reward received

            # Recompute sizing output for these states
            raw_out   = self.sizing_net.forward(s_entry)  # (n, 1)
            sig_out   = sigmoid(raw_out)                  # (n, 1)

            # REINFORCE: gradient = -reward * d/d(raw) log(sigma(raw))
            # log(sigma(x)) → gradient w.r.t. x = 1 - sigma(x)
            # Chain rule through sigmoid: d(log sigma) / d(raw) = sigmoid'(raw) / sigma(raw)
            #                                                    = (1 - sigma(raw))
            # So: grad_raw = -reward * (1 - sig_out)
            # We scale by the stored sizing to keep variance low (baseline-free REINFORCE)
            r_norm    = r_entry / (np.abs(r_entry).mean() + 1e-8)   # normalise rewards
            pg_grad   = -r_norm[:, None] * (1.0 - sig_out)          # (n, 1)

            sizing_loss = float((-r_norm * np.log(sz_entry + 1e-8)).mean())

            gw_sz, gb_sz = self.sizing_net.backward(s_entry, pg_grad)
            self.sizing_optim.step(self.sizing_net, gw_sz, gb_sz)

        total_loss = dqn_loss + SIZING_LOSS_WEIGHT * sizing_loss
        self.losses.append(total_loss)
        return total_loss

    # ── Main training loop ────────────────────────────────────────────────────

    def learn(self, env, total_timesteps: int, eval_env=None,
              eval_freq: int = 10_000, n_eval_episodes: int = 5,
              save_path: str = None, verbose: int = 1):
        """
        Train the agent for `total_timesteps` environment steps.
        Optionally evaluates and saves the best model.
        """
        import time

        best_eval_reward = -np.inf
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_count  = 0
        t_start   = time.time()

        BAR_WIDTH = 30

        def _print_bar():
            pct     = self.total_steps / total_timesteps
            filled  = int(BAR_WIDTH * pct)
            bar     = "#" * filled + "-" * (BAR_WIDTH - filled)
            elapsed = time.time() - t_start
            eta     = (elapsed / pct - elapsed) if pct > 0 else 0
            print(
                f"\r[{bar}] {self.total_steps:>7,}/{total_timesteps:,}"
                f"  {pct*100:5.1f}%"
                f"  elapsed {int(elapsed//60):02d}:{int(elapsed%60):02d}"
                f"  ETA {int(eta//60):02d}:{int(eta%60):02d}",
                end="", flush=True
            )

        while self.total_steps < total_timesteps:
            action_dict = self.predict(obs)
            direction   = action_dict["direction"]
            sizing      = float(action_dict["sizing"][0])

            next_obs, reward, terminated, truncated, info = env.step(action_dict)
            done = terminated or truncated

            self.buffer.push(obs, direction, sizing, reward, next_obs, done)
            obs        = next_obs
            ep_reward += reward
            self.total_steps += 1

            if done:
                obs, _ = env.reset()
                ep_count += 1
                ep_reward = 0.0

            # Train
            if self.total_steps % self.train_freq == 0:
                self.learn_step()

            # Target network update
            if self.total_steps % self.target_update == 0:
                self.target.copy_from(self.online)

            # Live progress bar every 500 steps
            if self.total_steps % 500 == 0:
                _print_bar()

            # Evaluation
            if eval_env is not None and self.total_steps % eval_freq == 0:
                mean_r   = self._evaluate(eval_env, n_eval_episodes)
                avg_loss = np.mean(self.losses[-500:]) if self.losses else 0
                if verbose:
                    print(flush=True)
                    print(
                        f"  step={self.total_steps:,}  "
                        f"eval_reward={mean_r:.4f}  "
                        f"loss={avg_loss:.5f}  "
                        f"eps={self.epsilon:.3f}",
                        flush=True
                    )
                if mean_r > best_eval_reward and save_path:
                    best_eval_reward = mean_r
                    self.save(str(Path(save_path) / "best_model.pkl"))

        print(flush=True)
        return self

    def _evaluate(self, env, n_episodes: int) -> float:
        rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            total  = 0.0
            done   = False
            while not done:
                action_dict = self.predict(obs, deterministic=True)
                obs, r, terminated, truncated, _ = env.step(action_dict)
                total += r
                done = terminated or truncated
            rewards.append(total)
        return float(np.mean(rewards))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "online":     {"weights": self.online.weights,      "biases": self.online.biases},
                "sizing_net": {"weights": self.sizing_net.weights,  "biases": self.sizing_net.biases},
                "total_steps": self.total_steps,
                "config": {
                    "gamma":           self.gamma,
                    "batch_size":      self.batch_size,
                    "learning_starts": self.learning_starts,
                    "train_freq":      self.train_freq,
                    "target_update":   self.target_update,
                    "n_actions":       self.n_actions,
                    "eps_start":       self.eps_start,
                    "eps_end":         self.eps_end,
                    "eps_decay_steps": self.eps_decay_steps,
                    "sizing_noise":    self.sizing_noise,
                },
            }, f)

    @classmethod
    def load(cls, path: str) -> "DQNAgent":
        with open(path, "rb") as f:
            d = pickle.load(f)
        cfg   = d["config"]
        agent = cls.__new__(cls)
        agent.__dict__.update(cfg)

        agent.online = MLP.__new__(MLP)
        agent.online.weights = d["online"]["weights"]
        agent.online.biases  = d["online"]["biases"]
        agent.online._cache  = None

        agent.target = MLP.__new__(MLP)
        agent.target.weights = [w.copy() for w in agent.online.weights]
        agent.target.biases  = [b.copy() for b in agent.online.biases]
        agent.target._cache  = None

        agent.sizing_net = MLP.__new__(MLP)
        if "sizing_net" in d:
            agent.sizing_net.weights = d["sizing_net"]["weights"]
            agent.sizing_net.biases  = d["sizing_net"]["biases"]
        else:
            # Legacy model without sizing head — initialise fresh
            obs_dim = agent.online.weights[0].shape[0]
            fresh   = MLP([obs_dim, 128, 64, 1], seed=99)
            agent.sizing_net.weights = fresh.weights
            agent.sizing_net.biases  = fresh.biases
        agent.sizing_net._cache = None

        agent.optim        = Adam()
        agent.sizing_optim = Adam()
        agent.buffer       = ReplayBuffer(100_000)
        agent.total_steps  = d.get("total_steps", 0)
        agent.losses       = []
        return agent
