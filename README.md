# Gambler's Problem — Value Iteration

Solves the classic Gambler's Problem from Sutton & Barto (Example 4.3):
a gambler flips a biased coin and bets on heads to reach a goal capital of $100.

The optimal policy $\pi(s)$ and value function $V(s)$ are computed via
Gauss-Seidel value iteration — in each state $s$, the action
$a \in [1, \min(s, 100-s)]$ that maximizes the Bellman equation is chosen:

$$V(s) = \max_a\ \bigl[\, p_h\,(r_{\text{win}} + V(s+a)) \;+\; (1-p_h)\,(r_{\text{step}} + V(s-a)) \,\bigr]$$

## Interactive Presentation Mode

The app has four views, switchable via the top navigation:

- **Guessing** — shows the casino scenario ($50 → $100, $p_h = 0.4$) and collects
  audience bets before revealing the optimal strategy; reveals a probability table
  (win % + method note per bet) and a log-scale bar chart
- **Micro-World ($3)** — step-by-step walkthrough of value iteration on a
  3-state toy problem (Sweeps 1–16 annotated), ideal for explaining the Bellman
  updates live
- **Analysis** — full value function $V(s)$ across sweep snapshots, optimal policy
  chart $\pi(s)$, and a convergence log; sidebar controls $p_h$, rewards, and snapshots
- **Lab & Dev** — an interactive $\gamma$ experiment (discount factor sweep with
  auto-generated plain-English summary) plus two implementation issues not covered
  in the book: floating-point convergence and tie-breaking in argmax
