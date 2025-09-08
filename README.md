# Multi-Currency-Environment

## State Space

We have \(N\) currencies to trade. Each currency \(i\) has a wallet with balance \(w_i\).  
The price in USD of currency \(i\) is denoted by \(p_i\).  

We also have capital \(C\), representing USD available for investment.

The portfolio value in USD for currency \(i\) is
\[
v_i = w_i p_i.
\]
The total portfolio value is
\[
V = C + \sum_{i=1}^N v_i = C + \sum_{i=1}^N w_i p_i.
\]

---

## Action Space

For \(N\) currencies, the agent outputs \(N+1\) values \(\{a_0, a_1, \dots, a_N\}\), each in \((-1,1)\).  
Indices \(1,\dots,N\) map to currencies. \(a_0\) controls the budget fraction for buying.  
A \(\tanh\) output layer naturally produces values in \((-1,1)\).

---

### Sell

Define
\[
\mathcal{S} = \{k \in \{1,\dots,N\} : a_k < 0\}.
\]

For each \(k \in \mathcal{S}\), we sell a fraction
\[
f_k = \max(0, -a_k).
\]

Wallet update:
\[
w_k \mapsto (1 - f_k) w_k.
\]

Capital update:
\[
C \mapsto C + \sum_{k \in \mathcal{S}} p_k w_k f_k (1 - s_k),
\]
where \(s_k \ll 1\) is the sell fee (fraction of notional traded).

---

### Buy

Define
\[
\mathcal{B} = \{l \in \{1,\dots,N\} : a_l > 0\}.
\]

Normalize allocation weights:
\[
\bar{a}_l = \frac{a_l}{\sum_{j \in \mathcal{B}} a_j}, \quad l \in \mathcal{B}.
\]

Investment budget fraction:
\[
\bar{a}_0 = \frac{a_0 + 1}{2} \in (0,1).
\]

Total investable USD:
\[
I = \bar{a}_0 C.
\]

Per-currency USD allocation:
\[
A_l = \bar{a}_l I.
\]

Units purchased:
\[
u_l = \frac{A_l (1 - b_l)}{p_l},
\]
where \(b_l\) is the buy fee fraction.

Wallet update:
\[
w_l \mapsto w_l + u_l.
\]

Capital update:
\[
C \mapsto C - I.
\]

---

## Improvements vs. Original

- Corrected **sell fees**: fee is proportional to notional traded, not full wallet.  
- Corrected **buy update**: wallet increases in **units**, not USD.  
- Fees applied **before adding to wallet**.  
- Capital updates subtract only the allocated budget \(I\), since fees are included in \(A_l(1-b_l)\).  
- Guardrails: skip trades if allocations are negligible, ensure no negative \(C\), and clip fractions to \([0,1]\).  
- Supports slippage or minimum trade thresholds if needed.

---
