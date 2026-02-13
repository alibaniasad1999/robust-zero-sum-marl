# Convergence to Nash Equilibrium (Tabular, Discounted, Two-Player Zero-Sum Markov Game)

This section states and proves the standard *tabular* convergence results used to justify minimax/Nash language in
discounted zero-sum Markov games.

> **Scope:** Finite state/action spaces (tabular case), discount factor $\gamma\in[0,1)$, bounded rewards.  
> These results do **not** automatically extend to nonlinear function approximation (deep networks).

---

## 1) Problem setup

Let $\mathcal S$ be a finite state space. Player 1 (maximizer) chooses $a\in\mathcal A$ and player 2 (minimizer)
chooses $b\in\mathcal B$, where $\mathcal A,\mathcal B$ are finite. Rewards are bounded:
$r(s,a,b)\in[-R_{\max},R_{\max}]$. Transitions are $P(\cdot\mid s,a,b)$ and discount factor is $\gamma\in[0,1)$.

A stationary mixed strategy for player 1 is $\pi(\cdot\mid s)\in\Delta(\mathcal A)$ and for player 2
$\sigma(\cdot\mid s)\in\Delta(\mathcal B)$.

Define the discounted value under $(\pi,\sigma)$:
$$
J^{\pi,\sigma}(s)=\mathbb E\Big[\sum_{t=0}^\infty \gamma^t r(s_t,a_t,b_t)\,\Big|\,s_0=s,\ a_t\sim\pi(\cdot|s_t),\ b_t\sim\sigma(\cdot|s_t)\Big].
$$

The *game value* is
$$
V^\star(s)=\max_{\pi}\min_{\sigma}J^{\pi,\sigma}(s) \;=\; \min_{\sigma}\max_{\pi}J^{\pi,\sigma}(s).
$$

A stationary Nash equilibrium $(\pi^\star,\sigma^\star)$ satisfies for all $s$ and all stationary $\pi,\sigma$:
$$
J^{\pi,\sigma^\star}(s)\le J^{\pi^\star,\sigma^\star}(s)\le J^{\pi^\star,\sigma}(s).
$$

Existence of stationary equilibria for finite discounted stochastic games is classical.  
**Refs:** Shapley-style operator approach and stationary equilibrium results. :contentReference[oaicite:0]{index=0}

---

## 2) Shapley (Bellman–minimax) operator is a contraction

For any $V\in\mathbb R^{|\mathcal S|}$ define the Shapley operator $T$:
$$
(TV)(s)=\max_{x\in\Delta(\mathcal A)}\min_{y\in\Delta(\mathcal B)}
\sum_{a\in\mathcal A}\sum_{b\in\mathcal B} x(a)\,y(b)\Big(r(s,a,b)+\gamma\sum_{s'}P(s'|s,a,b)V(s')\Big).
$$

### Lemma 1 (Contraction)
$T$ is a $\gamma$-contraction in $\|\cdot\|_\infty$:
$$
\|TV-TW\|_\infty \le \gamma\|V-W\|_\infty \quad \forall V,W.
$$

**Proof.** Fix $s$. For any $x\in\Delta(\mathcal A)$ and $y\in\Delta(\mathcal B)$,
\begin{align*}
&\Big|\sum_{a,b}x(a)y(b)\big(r+\gamma\sum_{s'}P(s'|s,a,b)V(s')\big)
-\sum_{a,b}x(a)y(b)\big(r+\gamma\sum_{s'}P(s'|s,a,b)W(s')\big)\Big|\\
&= \gamma\Big|\sum_{a,b}x(a)y(b)\sum_{s'}P(s'|s,a,b)\big(V(s')-W(s')\big)\Big|\\
&\le \gamma\sum_{a,b}x(a)y(b)\sum_{s'}P(s'|s,a,b)\,\|V-W\|_\infty
= \gamma\|V-W\|_\infty.
\end{align*}
Taking $\max_x\min_y$ cannot increase the bound, hence
$|(TV)(s)-(TW)(s)|\le \gamma\|V-W\|_\infty$. Taking $\max_s$ yields the result. $\square$

**Refs:** Contraction/fixed-point statement for Shapley operator is standard. :contentReference[oaicite:1]{index=1}

### Corollary 1 (Unique fixed point and value-iteration convergence)
Since $T$ is a contraction on $(\mathbb R^{|\mathcal S|},\|\cdot\|_\infty)$, it has a unique fixed point $V^\star$
with $V^\star=TV^\star$, and the iteration $V_{k+1}=TV_k$ converges geometrically:
$$
\|V_k-V^\star\|_\infty \le \gamma^k\|V_0-V^\star\|_\infty.
$$
$\square$

---

## 3) Fixed point induces a stationary Nash equilibrium

Define the optimal state–action value
$$
Q^\star(s,a,b)=r(s,a,b)+\gamma\sum_{s'}P(s'|s,a,b)V^\star(s').
$$

For each state $s$, consider the *matrix game* with payoff matrix $Q^\star(s,\cdot,\cdot)$.
Let $(x_s^\star,y_s^\star)$ be a saddle point of this matrix game:
$$
x_s^\star \in \arg\max_{x\in\Delta(\mathcal A)}\min_{y\in\Delta(\mathcal B)} \sum_{a,b}x(a)y(b)Q^\star(s,a,b),
\quad
y_s^\star \in \arg\min_{y}\max_{x} \sum_{a,b}x(a)y(b)Q^\star(s,a,b).
$$
Define stationary policies $\pi^\star(a|s)=x_s^\star(a)$ and $\sigma^\star(b|s)=y_s^\star(b)$.

### Lemma 2 (One-step saddle inequality)
For all $s$ and all mixed actions $x\in\Delta(\mathcal A)$, $y\in\Delta(\mathcal B)$,
$$
\sum_{a,b}x(a)\sigma^\star(b|s)Q^\star(s,a,b)\le V^\star(s)\le \sum_{a,b}\pi^\star(a|s)y(b)Q^\star(s,a,b).
$$

**Proof.** By the minimax theorem, $(x_s^\star,y_s^\star)$ is a saddle point of the finite zero-sum matrix game
$Q^\star(s,\cdot,\cdot)$, hence it satisfies the saddle inequalities with value
$V^\star(s)=\max_x\min_y \sum_{a,b}x(a)y(b)Q^\star(s,a,b)$. $\square$

### Theorem 1 (Stationary Nash equilibrium)
$(\pi^\star,\sigma^\star)$ is a Nash equilibrium in stationary strategies and achieves value $V^\star$:
$$
J^{\pi,\sigma^\star}(s)\le V^\star(s)\le J^{\pi^\star,\sigma}(s)\quad \forall s,\ \forall \pi,\sigma.
$$

**Proof sketch (dynamic programming).**
Define the Bellman operator under fixed stationary policies $(\pi,\sigma)$:
$$
(T^{\pi,\sigma}V)(s)=\sum_{a,b}\pi(a|s)\sigma(b|s)\Big(r(s,a,b)+\gamma\sum_{s'}P(s'|s,a,b)V(s')\Big).
$$
$T^{\pi,\sigma}$ is also a $\gamma$-contraction, so it has a unique fixed point $V^{\pi,\sigma}$.

Using Lemma 2 and $Q^\star(s,a,b)=r(s,a,b)+\gamma\sum_{s'}P(\cdot)V^\star(s')$ yields
$$
T^{\pi,\sigma^\star}V^\star \le V^\star \le T^{\pi^\star,\sigma}V^\star \quad \forall \pi,\sigma.
$$
Apply $(T^{\pi,\sigma^\star})^k$ to the left inequality and $(T^{\pi^\star,\sigma})^k$ to the right inequality.
Taking $k\to\infty$ (contraction) gives
$V^{\pi,\sigma^\star}\le V^\star \le V^{\pi^\star,\sigma}$, i.e., Nash inequalities. $\square$

**Refs:** Stationary equilibrium for finite discounted stochastic games; operator approach. :contentReference[oaicite:2]{index=2}

---

## 4) Minimax-Q: contraction + stochastic approximation ⇒ almost sure convergence

Minimax-Q (introduced for zero-sum Markov games) updates a tabular $Q(s,a,b)$. :contentReference[oaicite:3]{index=3}

Define the minimax value induced by any $Q$:
$$
V_Q(s)=\max_{x\in\Delta(\mathcal A)}\min_{y\in\Delta(\mathcal B)}\sum_{a,b}x(a)y(b)Q(s,a,b).
$$
Define the Bellman–minimax operator $H$ on $Q$:
$$
(HQ)(s,a,b)=r(s,a,b)+\gamma\sum_{s'}P(s'|s,a,b)\,V_Q(s').
$$

### Lemma 3 ($V_Q$ is 1-Lipschitz in $\|\cdot\|_\infty$)
$$
|V_{Q_1}(s)-V_{Q_2}(s)|\le \|Q_1-Q_2\|_\infty \quad \forall s.
$$

**Proof.**
For any $x,y$,
$
\left|\sum_{a,b}x(a)y(b)\big(Q_1(s,a,b)-Q_2(s,a,b)\big)\right|
\le \|Q_1-Q_2\|_\infty.
$
Taking $\max_x\min_y$ preserves the inequality. $\square$

### Lemma 4 ($H$ is a $\gamma$-contraction)
$$
\|HQ_1-HQ_2\|_\infty \le \gamma\|Q_1-Q_2\|_\infty.
$$

**Proof.**
Using Lemma 3,
\begin{align*}
|(HQ_1-HQ_2)(s,a,b)|
&=\gamma\left|\sum_{s'}P(s'|s,a,b)\big(V_{Q_1}(s')-V_{Q_2}(s')\big)\right|\\
&\le \gamma\sum_{s'}P(s'|s,a,b)\|Q_1-Q_2\|_\infty
=\gamma\|Q_1-Q_2\|_\infty.
\end{align*}
Taking $\sup_{s,a,b}$ yields the result. $\square$

Therefore, $H$ has a unique fixed point $Q^\star$.

### Theorem 2 (Almost sure convergence of tabular Minimax-Q)
Consider the asynchronous stochastic update:
$$
Q_{t+1}(s_t,a_t,b_t)
=(1-\alpha_t)\,Q_t(s_t,a_t,b_t)+\alpha_t\Big(r_t+\gamma V_{Q_t}(s_{t+1})\Big),
$$
and $Q_{t+1}(s,a,b)=Q_t(s,a,b)$ for all other $(s,a,b)$.

Assume:
1. Every $(s,a,b)$ is visited infinitely often;
2. Step sizes satisfy Robbins–Monro: $\sum_t\alpha_t=\infty$ and $\sum_t\alpha_t^2<\infty$;
3. Rewards are bounded.

Then $Q_t \to Q^\star$ almost surely, and stationary saddle policies extracted from $Q^\star$
form a Nash equilibrium achieving the game value.

**Proof idea (stochastic approximation).**
The update is an asynchronous stochastic approximation of the fixed-point iteration $Q\leftarrow HQ$.
Since $H$ is a $\gamma$-contraction (Lemma 4), the ODE $\dot Q = H(Q)-Q$ has a globally asymptotically stable
equilibrium at $Q^\star$. Under Assumptions 1–3, standard asynchronous stochastic approximation results imply
$Q_t\to Q^\star$ almost surely. $\square$

**Refs:** Asynchronous stochastic approximation / Q-learning convergence machinery. :contentReference[oaicite:4]{index=4}  
**Ref (algorithm context):** Minimax-Q for Markov games. :contentReference[oaicite:5]{index=5}  
(Recent work also provides finite-time analyses of minimax Q-learning under additional conditions.) :contentReference[oaicite:6]{index=6}

---

## References (links)

- Shapley operator contraction / stochastic game value iteration: :contentReference[oaicite:7]{index=7}  
- Stationary equilibrium existence (finite discounted stochastic games): :contentReference[oaicite:8]{index=8}  
- Minimax-Q / Markov games framework: :contentReference[oaicite:9]{index=9}  
- Asynchronous stochastic approximation and Q-learning convergence: :contentReference[oaicite:10]{index=10}
