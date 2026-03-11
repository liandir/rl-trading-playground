Rainbow is best understood as **DQN plus six orthogonal modifications**.

---

# 1. Base MDP setup

We assume a discounted Markov decision process
[
(\mathcal S,\mathcal A,P,R,\gamma),
]
with:

* state (s \in \mathcal S),
* discrete action (a \in \mathcal A),
* transition kernel (P(s' \mid s,a)),
* reward (r \sim R(\cdot \mid s,a)),
* discount (\gamma \in [0,1)).

The optimal action-value function is
[
Q^*(s,a) = \mathbb E\Big[\sum_{t=0}^{\infty}\gamma^t r_t ,\Big|, s_0=s, a_0=a, \pi=\pi^*\Big].
]

It satisfies the Bellman optimality equation
[
Q^*(s,a)
========

\mathbb E\big[r + \gamma \max_{a'} Q^*(s',a') \mid s,a\big].
]

---

# 2. Vanilla DQN

DQN approximates (Q(s,a)) by a neural network (Q_\theta(s,a)), and uses a target network (Q_{\bar\theta}).

For a sampled transition ((s,a,r,s',d)), where (d \in {0,1}) is terminal, the target is
[
y^{\mathrm{DQN}}
================

r + \gamma (1-d)\max_{a'} Q_{\bar\theta}(s',a').
]

The TD error is
[
\delta = y^{\mathrm{DQN}} - Q_\theta(s,a),
]
and the standard loss is
[
L(\theta)=\mathbb E\left[\ell(\delta)\right],
]
usually with squared loss or Huber loss.

Rainbow modifies this template in six places.

---

# 3. The six Rainbow components

Rainbow combines:

1. **Double DQN**
2. **Prioritized Experience Replay**
3. **Dueling architecture**
4. **Multi-step returns**
5. **Distributional RL (C51)**
6. **Noisy Networks**

I’ll define each carefully.

---

## 3.1 Double DQN

Vanilla DQN uses
[
\max_{a'} Q_{\bar\theta}(s',a')
]
for both **selection** and **evaluation**, which creates overestimation bias.

Double DQN separates them:

* choose the next greedy action using the online network:
  [
  a^* = \arg\max_{a'} Q_\theta(s',a'),
  ]

* evaluate that action using the target network:
  [
  y^{\mathrm{DDQN}}
  =
  r + \gamma(1-d),Q_{\bar\theta}(s', a^*).
  ]

So:
[
y^{\mathrm{DDQN}}
=================

r + \gamma(1-d),Q_{\bar\theta}\Big(s', \arg\max_{a'}Q_\theta(s',a')\Big).
]

This reduces positive bias from the max operator.

---

## 3.2 Prioritized Experience Replay (PER)

Instead of sampling replay uniformly, transitions are sampled according to priority.

For transition (i), define priority
[
p_i > 0,
]
typically
[
p_i = |\delta_i| + \varepsilon.
]

Sampling probability:
[
P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha},
]
where (\alpha \in [0,1]) controls how strongly prioritization is used.

Because this changes the sampling distribution, importance weights are used:
[
w_i
===

\left(\frac{1}{N}\cdot \frac{1}{P(i)}\right)^\beta,
]
with replay size (N), and usually normalized as
[
\tilde w_i = \frac{w_i}{\max_j w_j}.
]

The loss becomes
[
L(\theta) = \mathbb E_{i \sim P}\big[\tilde w_i ,\ell(\delta_i)\big].
]

In Rainbow with C51, priorities are usually based on the magnitude of the distributional loss or an implied scalar TD error.

---

## 3.3 Dueling network architecture

Instead of directly parameterizing (Q(s,a)), dueling networks decompose it into:

* a state-value stream (V(s)),
* an advantage stream (A(s,a)).

A naive decomposition is not identifiable since
[
Q(s,a) = V(s)+A(s,a)
]
is unchanged if one adds a constant to (A) and subtracts it from (V).

So the dueling combination is
[
Q(s,a)
======

V(s)
+
\left(
A(s,a) - \frac{1}{|\mathcal A|}\sum_{a'} A(s,a')
\right).
]

This centers the advantage stream and makes the decomposition identifiable enough for training.

In Rainbow+C51, this decomposition is often applied **per atom** of the return distribution.

---

## 3.4 Multi-step returns

Instead of a 1-step target, Rainbow uses (n)-step returns.

For a trajectory fragment
[
(s_t,a_t,r_t,s_{t+1},r_{t+1},\dots,s_{t+n}),
]
define the (n)-step return:
[
R_t^{(n)}
=========

\sum_{k=0}^{n-1}\gamma^k r_{t+k}.
]

The (n)-step target is
[
y_t^{(n)}
=========

R_t^{(n)} + \gamma^n (1-d_{t:t+n-1}), \text{bootstrap}(s_{t+n}),
]
where (d_{t:t+n-1}) means “terminated before or at step (n).”

In scalar Double DQN this becomes
[
y_t^{(n)}
=========

\sum_{k=0}^{n-1}\gamma^k r_{t+k}
+
\gamma^n (1-d_{t:t+n-1})
Q_{\bar\theta}\Big(s_{t+n}, \arg\max_{a'}Q_\theta(s_{t+n},a')\Big).
]

This usually propagates reward information faster.

---

## 3.5 Distributional RL (C51)

This is the most mathematically distinctive part of Rainbow.

Instead of learning only the expectation (Q(s,a)), we learn the full **distribution of returns**.

Define the return random variable
[
Z^\pi(s,a)
==========

\sum_{t=0}^{\infty}\gamma^t r_t.
]

Then
[
Q^\pi(s,a) = \mathbb E[Z^\pi(s,a)].
]

C51 approximates the distribution (Z(s,a)) with a categorical distribution supported on a fixed grid:
[
{z_i}*{i=0}^{N-1}, \qquad
z_i = V*{\min} + i\Delta z,
]
where
[
\Delta z = \frac{V_{\max}-V_{\min}}{N-1}.
]

The network outputs probabilities
[
p_i(s,a) \approx \Pr\big(Z(s,a)=z_i\big),
]
with
[
\sum_i p_i(s,a)=1.
]

Thus the predicted distribution is
[
Z_\theta(s,a) = \sum_{i=0}^{N-1} p_i(s,a),\delta_{z_i}.
]

The corresponding scalar Q-value is its expectation:
[
Q_\theta(s,a)
=============

# \mathbb E[Z_\theta(s,a)]

\sum_{i=0}^{N-1} z_i, p_i(s,a).
]

### Distributional Bellman target

For a next-state action (a^*), define the target distribution before projection:
[
T Z(s,a)
========

R^{(n)} + \gamma^n Z_{\bar\theta}(s',a^*).
]

This shifts and scales the support:
[
\hat z_i = \mathrm{clip}\big(R^{(n)} + \gamma^n z_i,; V_{\min}, V_{\max}\big).
]

Because the shifted support (\hat z_i) does not generally lie exactly on the fixed grid ({z_j}), C51 projects the target distribution back onto the fixed support using the projection operator (\Phi).

### Categorical projection

Let
[
b_i = \frac{\hat z_i - V_{\min}}{\Delta z},
\qquad
l_i = \lfloor b_i \rfloor,
\qquad
u_i = \lceil b_i \rceil.
]

Then the mass (p_i^{\text{next}}) from atom (i) is redistributed linearly onto neighboring support atoms:
[
m_{l_i} \mathrel{+}= p_i^{\text{next}}(u_i-b_i),
]
[
m_{u_i} \mathrel{+}= p_i^{\text{next}}(b_i-l_i).
]

If (l_i=u_i), all mass goes there.

The projected target is
[
m = \Phi(TZ).
]

### C51 loss

If the network predicts probabilities (p_\theta(s,a)) for the chosen action, the loss is the cross-entropy:
[
L_{\mathrm{C51}}(\theta)
========================

-\sum_{i=0}^{N-1} m_i \log p_{\theta,i}(s,a).
]

Over a minibatch:
[
L(\theta)=\mathbb E[L_{\mathrm{C51}}(\theta)].
]

With PER:
[
L(\theta)=\mathbb E_{i\sim P}\big[\tilde w_i,L_{\mathrm{C51},i}(\theta)\big].
]

---

## 3.6 Noisy Networks

Instead of (\varepsilon)-greedy exploration, Rainbow uses parameterized stochasticity in linear layers.

A noisy linear layer is
[
y = (W_\mu + W_\sigma \odot \varepsilon_W)x + b_\mu + b_\sigma \odot \varepsilon_b.
]

So the effective parameters are random:

* (W = W_\mu + W_\sigma \odot \varepsilon_W),
* (b = b_\mu + b_\sigma \odot \varepsilon_b).

This induces state-dependent exploration through the Q-values themselves.

In practice, Rainbow uses **factorized Gaussian noise**, where
[
\varepsilon_W = f(\epsilon_{\text{out}}) f(\epsilon_{\text{in}})^\top
]
for random vectors (\epsilon_{\text{out}}, \epsilon_{\text{in}}), with
[
f(x)=\operatorname{sign}(x)\sqrt{|x|}.
]

This reduces noise generation cost.

At action selection time:
[
a_t = \arg\max_a Q_\theta(s_t,a),
]
but the network itself is stochastic because of noisy weights.

---

# 4. Putting Rainbow together

Now combine all six pieces.

The network outputs a categorical distribution over returns for every action:
[
p_\theta(s,a) \in \Delta^{N-1}.
]

The implied Q-values are
[
Q_\theta(s,a)=\sum_{i=0}^{N-1} z_i,p_{\theta,i}(s,a).
]

For a sampled (n)-step transition
[
(s_t,a_t,R_t^{(n)}, s_{t+n}, d),
]
Double DQN action selection chooses
[
a^*
===

\arg\max_a Q_\theta(s_{t+n},a).
]

The target distribution is taken from the target network:
[
p_{\bar\theta}(s_{t+n}, a^*).
]

This distribution is shifted by the (n)-step return and discount:
[
\hat z_i
========

\mathrm{clip}\big(R_t^{(n)} + \gamma^n(1-d)z_i,; V_{\min},V_{\max}\big).
]

Project onto the fixed support:
[
m = \Phi\left(R_t^{(n)} + \gamma^n(1-d) Z_{\bar\theta}(s_{t+n},a^*)\right).
]

For the chosen action (a_t), the online network predicts
[
p_\theta(s_t,a_t).
]

The per-sample Rainbow loss is
[
\ell_t(\theta)
==============

-\sum_{i=0}^{N-1} m_i \log p_{\theta,i}(s_t,a_t).
]

With prioritized replay and importance weights:
[
L(\theta)
=========

\mathbb E_{t\sim P}\left[\tilde w_t,\ell_t(\theta)\right].
]

The target network is periodically or softly updated:
[
\bar\theta \leftarrow \theta
\quad\text{(hard update every (K) steps)}
]
or
[
\bar\theta \leftarrow \tau \theta + (1-\tau)\bar\theta.
]

---

# 5. Full algorithm in equations

Let replay contain (n)-step prioritized transitions.

At training step (k):

### Step 1: sample minibatch

Sample indices (i) with
[
P(i)=\frac{p_i^\alpha}{\sum_j p_j^\alpha}.
]

Compute importance weights
[
\tilde w_i
==========

\frac{\left(\frac{1}{N}\frac{1}{P(i)}\right)^\beta}
{\max_j \left(\frac{1}{N}\frac{1}{P(j)}\right)^\beta}.
]

### Step 2: online action selection

For each sampled transition, compute
[
Q_\theta(s'*i,a)=\sum*{j=0}^{N-1} z_j,p_{\theta,j}(s'*i,a),
]
then
[
a_i^*=\arg\max_a Q*\theta(s'_i,a).
]

### Step 3: target distribution

Obtain target probabilities
[
p_{\bar\theta}(s'_i,a_i^*).
]

Form shifted atoms
[
\hat z_{ij}
===========

\mathrm{clip}\big(r_i^{(n)} + \gamma^n(1-d_i)z_j,;V_{\min},V_{\max}\big).
]

Project to the fixed support:
[
m_i = \Phi\left(r_i^{(n)}+\gamma^n(1-d_i)Z_{\bar\theta}(s'_i,a_i^*)\right).
]

### Step 4: prediction

Get predicted action distribution for sampled action (a_i):
[
p_\theta(s_i,a_i).
]

### Step 5: loss

[
\ell_i(\theta)
==============

-\sum_{j=0}^{N-1} m_{ij}\log p_{\theta,j}(s_i,a_i).
]

Weighted minibatch loss:
[
L(\theta)=\frac{1}{B}\sum_{i=1}^B \tilde w_i \ell_i(\theta).
]

### Step 6: gradient update

[
\theta \leftarrow \theta - \eta \nabla_\theta L(\theta).
]

### Step 7: priority update

Set new priority, for example
[
p_i \leftarrow |\delta_i| + \varepsilon
]
or in C51 practice often
[
p_i \leftarrow \ell_i(\theta) + \varepsilon.
]

---

# 6. Where the dueling architecture fits in C51

In Rainbow, the network often predicts logits per atom using a dueling decomposition.

For each atom (z_j):

* value stream:
  [
  v_j(s)
  ]

* advantage stream:
  [
  a_j(s,a)
  ]

Combined logits:
[
\ell_j(s,a)
===========

v_j(s)
+
\left(
a_j(s,a)-\frac{1}{|\mathcal A|}\sum_{a'}a_j(s,a')
\right).
]

Then probabilities are
[
p_j(s,a)=\frac{\exp(\ell_j(s,a))}{\sum_{k=0}^{N-1}\exp(\ell_k(s,a))}.
]

So dueling operates at the **logit level**, before softmax over atoms.

---

# 7. Why Rainbow works

Each piece addresses a different weakness of vanilla DQN:

* **Double DQN**: reduces overestimation bias.
* **PER**: spends more updates on informative transitions.
* **Dueling**: better representation learning when many actions have similar value.
* **n-step returns**: faster propagation of reward information.
* **C51**: richer learning target than a scalar expectation.
* **NoisyNet**: learned exploration instead of hand-designed (\varepsilon)-greedy.

Rainbow’s strength is not a single theorem saying “all six together are optimal,” but that the pieces are largely complementary.

---

# 8. Subtle point: Rainbow is still value-based, not policy-based

Rainbow learns
[
Q(s,a)=\mathbb E[Z(s,a)]
]
and acts greedily:
[
\pi(s)=\arg\max_a Q(s,a).
]

There is no separate actor network. Exploration comes from the noisy parameters, not from sampling a policy distribution in the actor-critic sense.

---

# 9. Recurrent Rainbow variant

If you add recurrence, the network becomes something like
[
h_t = f_\theta(h_{t-1}, s_t),
]
and all quantities depend on hidden state:
[
p_\theta(z \mid h_t, a), \qquad
Q_\theta(h_t,a)=\sum_i z_i p_{\theta,i}(h_t,a).
]

Then the Bellman-style target is conceptually the same, but training becomes off-policy sequence learning rather than simple independent transition replay. In practice that means:

* sequence replay,
* burn-in,
* hidden-state reconstruction,
* masking across episode boundaries.

That is where recurrent DQN becomes much more delicate than recurrent PPO.

---

# 10. Compact “one-line” Rainbow objective

If I compress Rainbow into one mathematical object, it is:

[
L(\theta)
=========

\mathbb E_{i\sim P_{\text{PER}}}
\left[
\tilde w_i,
\mathrm{CE}
\Big(
\Phi\big(R_i^{(n)}+\gamma^n(1-d_i)Z_{\bar\theta}(s_i', \arg\max_a \mathbb E Z_\theta(s_i',a))\big),
,
Z_\theta(s_i,a_i)
\Big)
\right]
]

where:

* (\mathrm{CE}) is cross-entropy between target categorical distribution and predicted categorical distribution,
* (\Phi) is the C51 projection operator,
* replay sampling is prioritized,
* Q-values are represented with a dueling noisy network.

That is Rainbow in one sentence.

If you want, I can also give you:

1. a **derivation of the C51 projection step**, or
2. a **line-by-line mapping from the Rainbow math to your PyTorch implementation**.
