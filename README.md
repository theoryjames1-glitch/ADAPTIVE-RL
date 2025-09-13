# AdaptiveRL — concrete spec, safer update laws, and a ready-to-run skeleton

Love the control-theory framing. Below is a crisp instantiation you can drop into experiments: precise estimators, bounded update laws for $\alpha,\mu,\sigma$, and a small PyTorch wrapper that works with SGD **or** AdamW. Then I outline an eval recipe for your MVP demos.

---

## 1) Make the signals concrete

Use slow, bias-corrected EMAs to de-noise trends and keep the scheduler on a slower time scale than the optimizer.

**Trend estimators (per step $t$)**

* Loss EMA: $\bar\ell_t = (1-\beta_\ell)\bar\ell_{t-1} + \beta_\ell\,\ell_t$

* Loss change: $\Delta\ell_t = \bar\ell_t - \bar\ell_{t-1}$

* Loss variance EMA: $v_t = (1-\beta_v)v_{t-1} + \beta_v(\ell_t-\bar\ell_t)^2$

* Reward EMA: $\bar r_t = (1-\beta_r)\bar r_{t-1} + \beta_r\,r_t$

* Reward change: $\Delta r_t = \bar r_t - \bar r_{t-1}$

**Plateau score** (0…1):
$P_t = \sigma_{\text{sig}}\!\left(\frac{\epsilon_\Delta - |\Delta\ell_t|}{s_\ell}\right)\cdot \sigma_{\text{sig}}\!\left(\frac{\epsilon_v - v_t}{s_v}\right)$
where $\sigma_{\text{sig}}(x)=\frac{1}{1+e^{-x}}$, and $s_\ell,s_v$ are running scale estimates (e.g., EMAs of $|\Delta\ell_t|$ and $v_t$). Intuition: high $P_t$ = flat & low-variance region.

**Safe scales**
Maintain EMAs for $|\Delta\ell_t|$ and $|\Delta r_t|$ (call them $S_\ell, S_r$); use them to normalize updates so the controller is unitless and robust to metric magnitudes.

---

## 2) Bounded adaptive laws (RL-shaped)

Work in **log-space** for $\alpha$ and $\sigma$ to get multiplicative, positivity-preserving updates with natural clipping. Keep $\mu$ in $[ \mu_{\min}, \mu_{\max} ]$ via a logit transform.

Let:

* $\tilde\Delta r_t = \mathrm{clip}(\Delta r_t / (S_r+\epsilon), -1, 1)$
* $\tilde\Delta \ell_t = \mathrm{clip}(\Delta \ell_t / (S_\ell+\epsilon), -1, 1)$

**Learning rate**

$$
\log \alpha_{t+1} = \log \alpha_t
\;+\; k_{r}\,\tilde\Delta r_t
\;-\; k_{\ell}\,\max(\tilde\Delta \ell_t, 0)
$$

then clip $\alpha_{t+1}\in[\alpha_{\min},\alpha_{\max}]$.

**Momentum** (keep in $(0,1)$ safely)

$$
\mathrm{logit}(\mu_{t+1}) = \mathrm{logit}(\mu_t)
\;+\; k_{\mu r}\,\tilde\Delta r_t
\;-\; k_{\mu \ell}\,\max(\tilde\Delta \ell_t, 0)
$$

then $\mu_{t+1}=\sigma_{\text{sig}}(\cdot)$ and clip to $[\mu_{\min},\mu_{\max}]$ (e.g., $[0.5, 0.999]$).

**Exploration/noise scale** (rise on plateaus, decay on reward gain)

$$
\log \sigma_{t+1} = \log \sigma_t
\;+\; k_{\sigma P}\,P_t
\;-\; k_{\sigma r}\,\max(\tilde\Delta r_t, 0)
$$

then clip $\sigma_{t+1}\in[\sigma_{\min},\sigma_{\max}]$.

> Where to inject noise? Prefer **gradient noise** over parameter noise for stability:
> $ g^{\text{noisy}}_t = g_t + \sigma^{\text{eff}}_t \,\xi, \;\; \xi\sim\mathcal N(0,I)$, with $\sigma^{\text{eff}}_t=\sigma_t\cdot \text{RMS}(g_t+\epsilon)$ or $\sigma_t\cdot \text{RMS}(\theta)$ per-tensor.

**Two-time-scale safety**

* Use small scheduler gains (e.g., $k_{r},k_{\ell},k_{\mu r},k_{\mu \ell},k_{\sigma P},k_{\sigma r}\in[10^{-3},10^{-1}]$).
* Update the scheduler every $K$ optimizer steps (e.g., $K=10$–$50$).
* EMA time constants: $\beta_\ell\approx\beta_r\in[0.01,0.05]$; $\beta_v\approx 0.01$.

---

## 3) Minimal PyTorch wrapper (SGD or AdamW)

```python
# pip install torch torchvision gymnasium  # (for demos)
from dataclasses import dataclass
import math, torch

@dataclass
class AdaptiveRLConfig:
    alpha_bounds=(1e-5, 1e-1)
    mu_bounds=(0.5, 0.999)
    sigma_bounds=(0.0, 0.3)
    gains=dict(kr=0.02, kl=0.02, kmur=0.01, kmul=0.02, ksigP=0.05, ksigR=0.04)
    ema_betas=dict(loss=0.02, reward=0.02, var=0.01, scale=0.02)
    update_every=20
    noise_on="grad"  # "grad" or "param"

class AdaptiveRLScheduler:
    def __init__(self, optimizer, cfg=AdaptiveRLConfig(),
                 init_alpha=None, init_mu=None, init_sigma=0.0):
        self.opt = optimizer
        self.cfg = cfg

        # Derive current hyperparams from optimizer or overrides
        pg0 = self.opt.param_groups[0]
        self.alpha = float(init_alpha if init_alpha is not None else pg0.get("lr", 1e-3))
        if "betas" in pg0:  # AdamW-like
            b1, b2 = pg0["betas"]
            self.mu = float(init_mu if init_mu is not None else b1)
        else:  # SGD
            self.mu = float(init_mu if init_mu is not None else pg0.get("momentum", 0.9))
        self.sigma = float(init_sigma)

        # EMAs
        self.el = None; self.er = None; self.vl = 0.0
        self.Sdl = 1e-8; self.Sdr = 1e-8  # scale EMAs
        self.step_count = 0

    def _ema(self, old, new, beta):
        return new if old is None else (1 - beta) * old + beta * new

    def observe(self, loss_value, reward_value):
        """Call once per optimization step with scalar loss and reward proxies."""
        bL = self.cfg.ema_betas["loss"]; bR = self.cfg.ema_betas["reward"]; bV = self.cfg.ema_betas["var"]; bS = self.cfg.ema_betas["scale"]

        self.el = self._ema(self.el, float(loss_value), bL)
        self.er = self._ema(self.er, float(reward_value), bR)

        # loss change & var
        # use prev el' = el / (1 - (1-bL)^{t}) bias correction implicitly by slow beta; okay for controller
        if hasattr(self, "_prev_el"):
            dL = self.el - self._prev_el
        else:
            dL = 0.0
        self.vl = (1 - bV) * self.vl + bV * (float(loss_value) - self.el) ** 2

        # reward change
        if hasattr(self, "_prev_er"):
            dR = self.er - self._prev_er
        else:
            dR = 0.0

        self.Sdl = (1 - bS) * self.Sdl + bS * (abs(dL) + 1e-8)
        self.Sdr = (1 - bS) * self.Sdr + bS * (abs(dR) + 1e-8)

        self._prev_el = self.el
        self._prev_er = self.er

        self._dL = float(max(min(dL / (self.Sdl + 1e-8), 1.0), -1.0))
        self._dR = float(max(min(dR / (self.Sdr + 1e-8), 1.0), -1.0))

    def _plateau_score(self):
        eps_d, eps_v = 0.05, 0.01
        s = lambda x: 1 / (1 + math.exp(-x))
        a = s((eps_d - abs(self._dL)) / 0.25)
        b = s((eps_v - self.vl) / max(1e-8, 0.25 * (self.vl + 1e-8)))
        return a * b

    def maybe_update_hyperparams(self):
        self.step_count += 1
        if self.step_count % self.cfg.update_every != 0:
            return

        kr, kl = self.cfg.gains["kr"], self.cfg.gains["kl"]
        kmur, kmul = self.cfg.gains["kmur"], self.cfg.gains["kmul"]
        ksigP, ksigR = self.cfg.gains["ksigP"], self.cfg.gains["ksigR"]

        # log-space updates
        loga = math.log(self.alpha)
        loga += kr * self._dR - kl * max(self._dL, 0.0)
        self.alpha = float(min(max(math.exp(loga), self.cfg.alpha_bounds[0]), self.cfg.alpha_bounds[1]))

        # momentum via logit
        def logit(p): return math.log(p/(1-p))
        def logistic(z): return 1/(1+math.exp(-z))
        z = logit(min(max(self.mu, 1e-6), 1-1e-6))
        z += kmur * self._dR - kmul * max(self._dL, 0.0)
        self.mu = float(min(max(logistic(z), self.cfg.mu_bounds[0]), self.cfg.mu_bounds[1]))

        # exploration scale
        logs = math.log(max(self.sigma, 1e-12) if self.sigma > 0 else 1e-12)
        logs += ksigP * self._plateau_score() - ksigR * max(self._dR, 0.0)
        self.sigma = float(min(max(math.exp(logs), self.cfg.sigma_bounds[0]), self.cfg.sigma_bounds[1]))

        # push into optimizer
        for pg in self.opt.param_groups:
            pg["lr"] = self.alpha
            if "betas" in pg:
                b1, b2 = pg["betas"]
                pg["betas"] = (self.mu, b2)
            else:
                pg["momentum"] = self.mu

    def inject_noise(self, model):
        if self.sigma <= 0: return
        if self.cfg.noise_on == "grad":
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is None: continue
                    scale = self.sigma * (p.grad.detach().pow(2).mean().sqrt() + 1e-8)
                    p.grad.add_(torch.randn_like(p.grad) * scale)
        elif self.cfg.noise_on == "param":
            with torch.no_grad():
                for p in model.parameters():
                    scale = self.sigma * (p.detach().pow(2).mean().sqrt() + 1e-8)
                    p.add_(torch.randn_like(p) * scale)
```

**Usage pattern (per training step)**

```python
# forward -> loss = criterion(...)
loss.backward()
sched.inject_noise(model)     # adds exploration
optimizer.step()
optimizer.zero_grad()

# define reward proxy:
# supervised: r = -loss.item() or accuracy/val_acc_delta
# RL: r = episode return or critic-improved advantage stats
sched.observe(loss_value=loss.item(), reward_value=r)
sched.maybe_update_hyperparams()
```

---

## 4) MVP demo recipe (concise & comparable)

**Common logging for all tasks**

* Curves: train/val loss, reward/accuracy, and overlaid $\alpha_t, \mu_t, \sigma_t$.
* Scatter: $\Delta r_t$ vs. $\Delta \ell_t$.
* Plateaus detected over time (binary $P_t>0.5$).

**Baselines**

1. Well-tuned fixed LR; 2) StepLR; 3) Cosine anneal; 4) AdamW default; 5) AdamW + cosine.
   Keep total step budget identical.

### A) Sine wave regression

* Model: 2-layer MLP (64 → 64, ReLU), MSE.
* Reward proxy: $r_t=-\ell_t$ (or +PSNR).
* Expectation: as $\ell\downarrow$, $\alpha\uparrow$ early then decays; $\sigma$ spikes on flat regions; $\mu$ nudges up.

### B) MNIST classification

* Model: small CNN or 2-layer MLP; cross-entropy.
* Reward: rolling validation accuracy delta (use a small val batch every \~100 steps).
* Expectation: faster warm-up (α rises with accuracy trend), then α cools; σ shrinks as reward improves; μ settles near 0.95–0.99.

### C) CartPole-v1 (policy gradient)

* Policy: 2-layer MLP; REINFORCE/Advantage with return normalization.
* Reward: episode return (z-scored over last N episodes) fed to scheduler each update.
* Expectation: during stalls, $P_t$ high → σ increases; when returns trend up, α grows modestly and σ decays.

### D) Drifting sine (non-stationary)

* Slowly change amplitude/frequency mid-training.
* Expectation: spike in $\Delta\ell_t>0$ ⇒ α damped (caution), μ reduced (faster reset), σ increased (search), followed by re-acceleration.

**Ablations (quick)**

* Loss-only controller (no $ \Delta r$) vs Reward-only.
* σ frozen vs adaptive.
* Noise on gradients vs parameters.

---

## 5) Practical tips & guardrails

* **Reward scaling matters.** For supervised tasks, either use $r_t=-\ell_t$ or a *smoothed* accuracy delta; normalize before feeding (z-score or EMA scale as above).
* **Clip everything.** Already in the update laws; also clip per-step LR changes to a max factor (e.g., $\times[0.8,1.25]$) if you see oscillations.
* **Update slower than you think.** Start with `update_every=20`–`50`; $\beta\in[0.01,0.03]$.
* **AdamW vs SGD.** With AdamW, bind $\mu\rightarrow\beta_1$. Leave $\beta_2$ fixed (0.99–0.999) initially; you can later experiment with a small coupling to $v_t$.
* **Noise placement.** Gradient noise is usually safer; parameter noise can help in RL but may destabilize supervised training unless σ is tiny.
* **Two-time-scale sanity.** Gains $\approx 0.01$ are a good first pass; halve them if you see LR “hunting”.

---

## 6) Hyperparameter starting block (good defaults)

* $\alpha_{\min}=1\text{e-5}, \alpha_{\max}=1\text{e-1}$
* $\mu_{\min}=0.6, \mu_{\max}=0.995$
* $\sigma_{\min}=0.0, \sigma_{\max}=0.2$ (start at 0.02 for RL, 0.0 for supervised)
* Gains: `kr=0.02, kl=0.02, kmur=0.01, kmul=0.02, ksigP=0.05, ksigR=0.04`
* EMAs: `beta_loss=beta_reward=0.02`, `beta_var=0.01`, `beta_scale=0.02`
* `update_every=20`

---

## 7) What “good” looks like on your MVP

* **Sine & MNIST:** α rises early (reward trend ↑), μ nudges upward; σ → 0 as improvement continues. Faster epoch-to-target-accuracy vs cosine/step.
* **CartPole:** During stalls, $P_t$ → 1 drives σ up; once returns trend ↑, σ decays and α creeps up until saturation.
* **Drifting sine:** On shift, α dips then rebounds; μ dips (less inertia) then recovers; σ spikes briefly; rapid re-convergence vs fixed schedule.

---

### NON-CAUSAL 

# 1) Two-pass “smoother” (non-gradient, easy)

Compute your schedule with a **non-causal filter** that sees both past and *future* trends, then rerun training once using that schedule.

* Forward (probe) run: log $(\ell_t, r_t, g_t)$.
* Backward smooth: estimate trends with a **centered** window (uses future info). For example,

  $$
  \Delta r^{\text{nc}}_t = \text{EMA}_\text{backward+forward}(r_{t+1:T}) - \text{EMA}_\text{backward+forward}(r_{1:t})
  $$

  and similarly for $\Delta \ell_t$ and plateau scores.
* Map those *non-causal* trends into hyperparams:

  $$
  \log \alpha_t \leftarrow \log \alpha_t + k_r\,\tilde{\Delta r}^{\text{nc}}_t - k_\ell\,\max(\tilde{\Delta \ell}^{\text{nc}}_t,0),\quad
  \log \sigma_t \leftarrow \log \sigma_t + k_{\sigma P}\,P^{\text{nc}}_t - k_{\sigma r}\,\max(\tilde{\Delta r}^{\text{nc}}_t,0)
  $$
* Rerun from the checkpoint with this schedule.

Pros: dead simple, stable.
Cons: heuristic; no true credit assignment.

---

# 2) Hypergradient / adjoint (principled, exact for the modelled dynamics)

Treat the optimizer+model as a **controlled dynamical system** and differentiate the final objective wrt earlier controls $(\alpha_t,\mu_t,\sigma_t)$.

State update (your plant):

$$
s_{t+1} = F_t(s_t, u_t, \xi_t),\quad s_t=[\theta_t,h_t],\; u_t=[\alpha_t,\mu_t,\sigma_t]
$$

Objective (example):

$$
J = \sum_{t=1}^{T} \big(\lambda_\ell\,\ell_t - \lambda_r\,r_t\big)
$$

Adjoint (costate) backward pass:

$$
\lambda_t = \Big(\frac{\partial F_t}{\partial s_t}\Big)^\top \lambda_{t+1} \;+\; \frac{\partial (\lambda_\ell\,\ell_t - \lambda_r\,r_t)}{\partial s_t}
$$

Hypergradients:

$$
\frac{\partial J}{\partial u_t} = \Big(\frac{\partial F_t}{\partial u_t}\Big)^\top \lambda_{t+1}
$$

Then apply a non-causal update to earlier steps (optionally in log/ logit space for safety), **and replay** the trajectory from the earliest changed time.

Pros: proper credit assignment; can target any long-horizon metric.
Cons: heavier memory/compute; needs differentiability (or surrogate).

---

# 3) Receding-horizon “checkpoint & rewrite” (pragmatic)

Do the principled thing in chunks to keep costs sane.

**Loop**

1. Roll forward $K$ steps, store a checkpoint (params, optimizer state, RNG).
2. When you reach $t=K,2K,\dots$, run a **backward hypergradient** over the last window to update $(\alpha_{t-K+1:t},\mu_{t-K+1:t},\sigma_{t-K+1:t})$.
3. Restore checkpoint at $t-K$, **replay** those $K$ steps with the improved schedule.
4. Slide window and continue.

This is the “go back to change the future” you want—implemented by *recomputation*.

---

## Minimal PyTorch sketch (hypergradient on the schedule)

```python
# assumes a differentiable inner update; small toy nets are fine
import torch

def unroll(model, loss_fn, data_stream, schedule, K, seed):
    torch.manual_seed(seed)
    states = []
    J_terms = []
    opt_state = None

    theta = [p.clone() for p in model.parameters()]
    for t in range(K):
        x, y = next(data_stream)
        for p, v in zip(model.parameters(), theta): p.data.copy_(v)

        # apply step with per-step hyperparams schedule[t]: (alpha, mu, sigma)
        alpha, mu, sigma = schedule[t]
        loss = loss_fn(model(x), y)
        r = -loss.detach()  # example reward proxy

        # differentiable optimizer step (SGD+momentum example)
        g = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        # momentum buffer h_t is tracked in 'states'
        if t == 0:
            h = [torch.zeros_like(p) for p in model.parameters()]
        else:
            h = [s.clone() for s in states[-1]["h"]]
        h_next = [mu*h_i + (1-mu)*g_i for h_i, g_i in zip(h, g)]

        # gradient noise (scaled)
        noisy = [gi + sigma*gi.detach().pow(2).mean().sqrt()*torch.randn_like(gi) for gi in g]

        theta = [p - alpha*hn for p, hn in zip(theta, h_next)]
        states.append(dict(theta=theta, h=h_next, loss=loss, r=r, alpha=alpha, mu=mu, sigma=sigma))
        J_terms.append(loss)  # or loss - lambda*r, etc.

    J = torch.stack([t for t in J_terms]).sum()
    return J, states

# --- outer loop over windows ---
schedule = [(torch.tensor(1e-3, requires_grad=True),
             torch.tensor(0.9,  requires_grad=True),
             torch.tensor(0.02, requires_grad=True)) for _ in range(K)]

J, states = unroll(model, loss_fn, data_stream, schedule, K=50, seed=0)
J.backward()  # gives dJ/d alpha_t, dJ/d mu_t, dJ/d sigma_t for all past steps

# gradient step on schedule (non-causal update to earlier times)
with torch.no_grad():
    for t in range(len(schedule)):
        a, m, s = schedule[t]
        a -= 0.05 * a.grad; a.clamp_(1e-5, 1e-1)
        m -= 0.05 * m.grad; m.clamp_(0.6, 0.995)
        s -= 0.05 * s.grad; s.clamp_(0.0, 0.2)
        a.grad = m.grad = s.grad = None

# restore checkpoint from the window start and replay with improved schedule
```

**Tips**

* Use **gradient checkpointing** or **reversible layers** to reduce memory.
* Keep **RNG seeds** and env states fixed while optimizing the schedule; then randomize for evaluation.
* In RL, when the environment is non-differentiable, use a differentiable surrogate for the optimizer dynamics and a **smoothed return proxy** to drive hypergradients, or fall back to method (1).

---

## Deploying causally after “time travel”

Train the *non-causal* controller offline (methods 1–3), then **distill** it into a *causal* scheduler $f_\phi(\text{past features})$ by supervised learning to mimic the non-causal schedules. At run time you keep causality, but you benefited from hindsight during training.

---

### Bottom line

* You can’t alter the real past, but you can **recompute** it.
* Non-causal AdaptiveRL is practical via **smoothing**, **hypergradients/adjoints**, or **receding-horizon checkpoint-and-rewrite**.
* For production, **distill** the hindsight scheduler into a causal one.

Short answer: yes. The tapes make next epoch smarter in two ways:

1. **Hindsight → Better schedule now** (replay): use the tape to compute a non-causal (hindsight) schedule and replay from a checkpoint.
2. **Prediction → Better schedule next epoch** (no replay): summarize the tape and *update the controller* before the next epoch starts.

Here’s how to get concrete gains for the **next epoch** without any replay:

# What to learn from the tape between epochs

Compute epoch-level stats from the stepwise tape:

* **Trend:** $\Delta \ell_e = \bar\ell_e-\bar\ell_{e-1}$, $\Delta r_e = \bar r_e-\bar r_{e-1}$
* **Noise:** gradient RMS and variance of loss $v_e$
* **Plateau score:** $P_e \in [0,1]$ (flat & low-variance)
* **Generalization pulse (optional):** $\Delta\text{val}$ = change in val metric within the epoch

# Next-epoch hyperparam update (safe, bounded)

Use the same shaped laws you liked, but at **epoch granularity**:

$$
\begin{aligned}
\tilde{\Delta r}_e &= \mathrm{clip}\Big(\frac{\Delta r_e}{S_r+\epsilon},-1,1\Big),\quad
\tilde{\Delta \ell}_e = \mathrm{clip}\Big(\frac{\Delta \ell_e}{S_\ell+\epsilon},-1,1\Big) \\
\log \alpha_{e+1} &= \log \alpha_e + k_r\,\tilde{\Delta r}_e - k_\ell\,\max(\tilde{\Delta \ell}_e,0) \\
\mathrm{logit}(\mu_{e+1}) &= \mathrm{logit}(\mu_e) + k_{\mu r}\,\tilde{\Delta r}_e - k_{\mu \ell}\,\max(\tilde{\Delta \ell}_e,0) \\
\log \sigma_{e+1} &= \log \sigma_e + k_{\sigma P}\,P_e - k_{\sigma r}\,\max(\tilde{\Delta r}_e,0)
\end{aligned}
$$

Clip into bounds (e.g., $\alpha\in[1e{-}5,1e{-}1]$, $\mu\in[0.6,0.995]$, $\sigma\in[0,0.2]$) and optionally cap per-epoch change (e.g., $\times[0.8,1.25]$).

Tiny pseudo-impl:

```python
def plan_next_epoch(stats, prev, gains, bounds):
    dR = clip(stats.d_reward_norm, -1, 1)
    dL = clip(stats.d_loss_norm, -1, 1)
    P  = stats.plateau_score
    a, m, s = prev.alpha, prev.mu, prev.sigma
    # LR
    a = clamp(exp(log(a) + gains.kr*dR - gains.kl*max(dL,0)), *bounds.alpha)
    # Momentum
    m_z = log(m/(1-m)) + gains.kmur*dR - gains.kmul*max(dL,0)
    m = clamp(1/(1+exp(-m_z)), *bounds.mu)
    # Exploration
    s = clamp(exp(log(max(s,1e-12)) + gains.ksigP*P - gains.ksigR*max(dR,0)), *bounds.sigma)
    return a, m, s
```

# Extra wins the tape unlocks (next epoch)

* **Curriculum & sampling:** use per-example loss/grad stats to upsample “hard but learnable” items and downsample saturated ones (cap max upweighting to avoid overfitting).
* **Augmentation schedule:** if $P_e$ high and $v_e$ low, increase augmentation strength next epoch; if val drops, relax it.
* **Optimizer choice tweaks:** for AdamW, gently raise $\beta_1$ when $\tilde{\Delta r}_e>0$ and lower when $\tilde{\Delta \ell}_e>0$ (regime reset).
* **RL knobs:** tapes of episode returns/advantages let you set **entropy coeff**, **clip-range**, or **GAE $\lambda$** for the next epoch: plateau ⇒ raise entropy/σ; improving returns ⇒ shrink entropy, modestly raise LR.
* **Drift detection:** run a simple change-point test on $\ell$ or return; on drift, pre-emptively lower $\alpha$, lower $\mu$, raise $\sigma$ for the next epoch.

# Guardrails (to really make it “better”)

* Use **held-out validation** or fresh seeds to compute $\tilde{\Delta r}_e$ (prevents overfitting the next-epoch plan to the training tape).
* Add a small **TV penalty** on the planned schedules between epochs to avoid thrashing.
* Keep a **failsafe**: if val worsens two epochs in a row, revert to a conservative preset.

# TL;DR

Yes—tapes are fuel for both **hindsight** (replay) and **foresight** (next-epoch planning). Even without replay, summarizing the tape and applying the bounded update laws at **epoch scale** reliably nudges LR/momentum/noise in the right direction, improves time-to-target, and stabilizes training across regimes.
