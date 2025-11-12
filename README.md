# Multi-Currency Trading Environment — Formal Specification

## 1. Overview

We consider a portfolio of $N$ tradable assets (currencies).
The agent maintains:

* a **cash balance** $C \in \mathbb{R}_+$ (USD capital),
* a **wallet** $w_i \in \mathbb{R}_+$ for each asset $i = 1,\dots,N$,
* and observes time-varying market features.

At each discrete time step $t_k$, prices $p_i(t_k)$ and traded volumes $v_i(t_k)$ are provided by an external data source.
The environment is **Markovian** with state

$$
s_t = f(C_t, w_t, p_t, v_t,\text{history})
$$

and continuous action

$$
a_t \in (-1,1)^{N+1}.
$$

---

## 2. State Space

The agent observes a *feature vector* containing only **relative**, scale-free quantities to avoid overfitting to absolute price levels.

### 2.1. Time Encoding

A cyclic embedding of the current timestamp $t$ encodes daily, weekly, and yearly periodicity:

$$
t_{\text{vec}} =
\begin{bmatrix}
\sin(2\pi \phi_{\text{day}}) & \cos(2\pi \phi_{\text{day}})\\
\sin(2\pi \phi_{\text{week}}) & \cos(2\pi \phi_{\text{week}})\\
\sin(2\pi \phi_{\text{year}}) & \cos(2\pi \phi_{\text{year}})
\end{bmatrix}
\in \mathbb{R}^{3\times 2}.
$$

Flattened dimension: **6**.

---

### 2.2. Relative Price Deviation

For each asset $i$ and each exponential smoother with time constant $\tau_m$,

$$
s_{m,i}(t) = s_{m,i}(t-\Delta t) + \left(1 - e^{-\frac{\Delta t}{\tau_m}}\right) \big(p_i(t) - s_{m,i}(t-\Delta t)\big),
$$

the **relative deviation** is

$$
p_{\text{rel},m,i}(t) =
\frac{p_i(t) - s_{m,i}(t)}{s_{m,i}(t)}.
$$

Shape: $[M,N]$.

---

### 2.3. Relative Portfolio Exposures

$$
x_{\text{rel}}(t) =
\begin{bmatrix}
\frac{C_t}{V_t}, &
\frac{w_1(t)p_1(t)}{V_t}, \dots, \frac{w_N(t)p_N(t)}{V_t}
\end{bmatrix},
\qquad
V_t = C_t + \sum_i w_i(t)p_i(t).
$$

Shape: $[N+1]$.
Interpretation: current *exposure weights* (cash + asset value fractions).

---

### 2.4. Relative Invested Capital (Stable Commitment)

Let the **cost basis** (dollar amount committed) be $I_i(t)$, updated only on trades:

$$
I_i(t+1) =
\begin{cases}
I_i(t) + \text{buy\_notional}_i, & \text{on buy},\\
I_i(t) - \text{sold\_cost}_i, & \text{on sell.}
\end{cases}
$$

Define total invested capital $S_t = \sum_i I_i(t)$, then

$$
c_{\text{rel},i}(t) = \frac{I_i(t)}{S_t+\varepsilon},\qquad
\rho_t = \frac{S_t}{S_t + C_t}.
$$

* $c_{\text{rel}}$ : per-asset committed fractions, shape $[N]$
* $\rho$ : scalar degree of total investment, shape $[1]$

These quantities are *slow* features—unchanged by pure market moves.

---

### 2.5. Moneyness vs. Entry (Unrealized PnL)

Using average entry cost

$$
\bar c_i(t) = \frac{I_i(t)}{\max(w_i(t),\varepsilon)},
$$

define per-asset **moneyness**

$$
m_i(t) = \frac{p_i(t) - \bar c_i(t)}{\bar c_i(t) + \varepsilon}.
$$

Shape: $[N]$.
Interpretation: fractional gain/loss on each open position.

---

### 2.6. Relative Volume Dynamics

Given raw volume $v_i(t)$ (or dollar-volume $p_i v_i$),

$$
v_{\text{rel},i}(t)
= \log\left(\frac{v_i(t)}{v_i(t-\Delta t)+\varepsilon}\right),
$$

which encodes multiplicative changes in market activity.
Shape: $[N]$.

---

### 2.7. State Vector Dimension

Flattening all components gives:

$$
\boxed{
\dim(\mathcal{S})
= 6 + M N + (N+1) + N + 1 + N + N
= M N + 4N + 8.
}
$$

Hence the state vector has **$M N + 4N + 8$** real-valued components.

---

## 3. Action Space

The agent outputs:

$$
a = [a_0, a_1, \dots, a_N] \in (-1,1)^{N+1},
$$

typically via a final $\tanh$ activation layer.

### 3.1. Buy/Sell Semantics

* **Sell signals**: for $i>0$ with $a_i < -\epsilon$

  $$
  f_i = |a_i|, \quad
  w_i \leftarrow (1 - f_i)w_i.
  $$

  Proceeds credited to $C$ after deducting percentage and flat fees.

* **Buy signals**: for $i>0$ with $a_i > \epsilon$
  The fraction of *buying budget* $I = \bar a_0 C$ allocated to asset $i$ is

  $$
  \bar a_i = \frac{a_i}{\sum_{j\in\mathcal{B}} a_j},
  \qquad
  \bar a_0 = \frac{a_0 + 1}{2}.
  $$

  USD spent on asset $i$:

  $$
  A_i = \bar a_i,\bar a_0,C.
  $$

  Units bought:

  $$
  \Delta w_i = \frac{A_i - \text{fees}_i}{p_i}.
  $$

All trades are subject to buy/sell fees:

$$
\text{fee}^{(\text{pct})}_i = b_i,A_i, \qquad
\text{fee}^{(\text{flat})}_i = b_i^{(\text{flat})}.
$$

---

### 3.2. Action Dimensionality

$$
\boxed{
\dim(\mathcal{A}) = N + 1.
}
$$

Each step produces one scalar action per asset plus one global capital-allocation control.

---

### 3.3. Summary of Trading Logic

| Symbol  | Range    | Meaning                                                  |
| :------ | :------- | :------------------------------------------------------- |
| $a_0$   | $(-1,1)$ | maps to capital-allocation fraction $\bar a_0 \in [0,1)$ |
| $a_i<0$ |          | sell fraction of units                                   |
| $a_i>0$ |          | participate in buying budget proportionally to $a_i$     |
| $N$     |          | number of tradable assets                                |

---

## 4. Reward Modes

1. **Absolute PnL**

   $$
   r_t = V_{t+1} - V_t.
   $$

2. **Log Return**

   $$
   r_t = \log\left(\frac{V_{t+1}+\varepsilon}{V_t+\varepsilon}\right).
   $$

3. **Realized ROI** (default for stable training)
   Only when sells occur:

   $$
   r_t =
   \sum_i
   \operatorname{clip}
   \left(
   \frac{\text{PnL}^{(\text{real})}*i}
   {\text{Cost}^{(\text{real})}*i + \varepsilon},
   \text{roi}*{\min}, \text{roi}*{\max}
   \right),
   $$

   where realized PnL and cost are computed from the agent’s average-cost basis.

---

## 5. Dimensional Summary

| Component        | Symbol           | Shape              | Description                            |
| :--------------- | :--------------- | :----------------- | :------------------------------------- |
| Time cycles      | $t_{\text{vec}}$ | [3, 2] → 6         | daily, weekly, yearly sin/cos encoding |
| Relative price   | $p_{\text{rel}}$ | [M, N]             | normalized deviations from EWMA        |
| Exposure weights | $x_{\text{rel}}$ | [N+1]              | cash + per-asset value fractions       |
| Cost weights     | $c_{\text{rel}}$ | [N]                | relative invested amounts              |
| Commitment ratio | $\rho$           | [1]                | total invested vs. cash                |
| Moneyness        | $m$              | [N]                | unrealized ROI per position            |
| Volume change    | $v_{\text{rel}}$ | [N]                | log relative volume                    |
| **Total**        | –                | **$M N + 4N + 8$** | flattened state dimension              |
| **Actions**      | –                | **$N + 1$**        | buy/sell controls                      |

---

✅ **State size:** $M N + 4N + 8$
✅ **Action size:** $N + 1$

---

