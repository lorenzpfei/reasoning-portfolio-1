# Gambler's Problem — Value Iteration

Solves the classic Gambler's Problem from Sutton & Barto (Example 4.3):
a gambler flips a biased coin and bets on heads to reach a goal capital of $100.

The optimal policy $\pi(s)$ and value function $V(s)$ are computed via
Gauss-Seidel value iteration — in each state $s$, the action
$a \in [1, \min(s, 100-s)]$ that maximizes the Bellman equation is chosen:

$$V(s) = \max_a\ \bigl[\, p_h\,(r_{\text{win}} + V(s+a)) \;+\; (1-p_h)\,(r_{\text{step}} + V(s-a)) \,\bigr]$$
