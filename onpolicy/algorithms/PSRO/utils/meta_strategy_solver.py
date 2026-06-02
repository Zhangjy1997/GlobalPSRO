import warnings

import numpy as np
from scipy.optimize import linprog
from open_spiel.python.egt import alpharank
from open_spiel.python.algorithms.projected_replicator_dynamics import projected_replicator_dynamics



def _alpharank_compute_checked(payoff_mats, **alpharank_kwargs):
    with warnings.catch_warnings():
        # AlphaRank can emit benign overflow warnings while still returning a
        # valid stationary distribution. Validate the result instead of failing
        # on the warning itself.
        warnings.simplefilter("ignore", RuntimeWarning)
        rhos, rho_m, pi, mass, transition_matrix = alpharank.compute(
            payoff_mats, **alpharank_kwargs
        )

    pi = np.asarray(pi, dtype=float)
    if not np.all(np.isfinite(pi)):
        raise FloatingPointError("AlphaRank returned non-finite probabilities")
    pi_sum = np.sum(pi)
    if not np.isfinite(pi_sum) or pi_sum <= 0:
        raise FloatingPointError("AlphaRank returned invalid probability mass")

    return rhos, rho_m, pi, mass, transition_matrix


def _alpharank_compute_with_param_fallback(payoff_mats):
    try:
        return _alpharank_compute_checked(payoff_mats)
    except Exception as exc:
        print(f"[alpha_rank_MSS] AlphaRank failed on raw payoff matrix ({exc}); retry with alpha staircase.")

    last_exc = None
    for alpha in (80, 60, 40, 20, 10, 5, 1):
        try:
            return _alpharank_compute_checked(payoff_mats, alpha=alpha)
        except Exception as exc:
            last_exc = exc
            print(f"[alpha_rank_MSS] AlphaRank failed with alpha={alpha} ({exc}).")

    raise last_exc


def zero_sum_2p_game(U:np.array):
    # U: payoff matrix of player 1 (-U^T for player 2)
    p1_action_dim, p2_action_dim = U.shape
    # print(p1_action_dim, p2_action_dim)

    c = np.zeros(p2_action_dim + 1)
    c[-1] = 1 

    # A_ub * x <= b_ub
    A_ub = np.hstack([U, -np.ones((p1_action_dim, 1))])
    b_ub = np.zeros(p1_action_dim)

    A_eq = np.ones((1, p2_action_dim + 1))
    A_eq[0, -1] = 0
    b_eq = np.array([1])

    bounds = [(0, None) for _ in range(p2_action_dim)] + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    s2 = res.x[:-1]
    U1_star = res.x[-1]

    # print(s2, U1_star)

    c = np.zeros(p1_action_dim + 1)
    c[-1] = -1

    # A_ub * x <= b_ub
    A_ub = np.hstack([-U.T, np.ones((p2_action_dim, 1))])
    b_ub = np.zeros(p2_action_dim)

    A_eq = np.ones((1, p1_action_dim + 1))
    A_eq[0, -1] = 0
    b_eq = np.array([1])

    bounds = [(0, None) for _ in range(p1_action_dim)] + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    s1 = res.x[:-1]
    U1_star_ = res.x[-1]

    # print(s1, U1_star_)

    return [s1, s2], [U1_star, -U1_star]

def alpha_rank_MSS(payoff_mat:np.array):
    if payoff_mat.shape[0] == payoff_mat.shape[1]:
        res_mat = payoff_mat + payoff_mat.T
    else:
        res_mat = np.ones((1,1))
    if np.max(np.abs(res_mat)) < 1e-5:
        U_mat = payoff_mat
        rhos, rho_m, pi, _, _ = _alpharank_compute_with_param_fallback([U_mat])
        probs = [np.array(pi), np.array(pi).copy()]
        U_s = [0.0, 0.0]
    else:
        payoff_mats = [payoff_mat, -payoff_mat]
        rhos, rho_m, pi, _, _ = _alpharank_compute_with_param_fallback(payoff_mats)
        pi = pi.reshape(payoff_mat.shape)
        pi_row = np.sum(pi, axis = 1)
        pi_col = np.sum(pi, axis = 0)
        probs = [pi_row, pi_col]
        Uq = np.dot(payoff_mat, pi_col)
        result = np.dot(pi_row, Uq)
        U_s = [result, -result]

    return probs, U_s

def loading_MSS(probs = None):
    if probs is None:
        probs = [np.ones(1), np.ones(1)]
    def static_MSS(payoff_mat):
        U_s = [0.0, 0.0]
        return probs.copy(), U_s
    
    return static_MSS

def Uniform_MSS(payoff_mat:np.array):
    rows, cols = payoff_mat.shape
    probs_rows = np.ones(rows)
    probs_rows /= np.sum(probs_rows)
    probs_cols = np.ones(cols)
    probs_cols /= np.sum(probs_cols)

    payoff_p1 = np.dot(probs_rows.T, np.dot(payoff_mat, probs_cols))

    return [probs_rows, probs_cols], [payoff_p1, -payoff_p1]

def PRD_MSS(payoff_mat:np.array):
    probs = projected_replicator_dynamics([payoff_mat, -payoff_mat.T])

    probs_rows, probs_cols = probs[0], probs[1]

    payoff_p1 = np.dot(probs_rows.T, np.dot(payoff_mat, probs_cols))

    return probs, [payoff_p1, -payoff_p1]


def Random_MSS(payoff_mat:np.array):
    m, n = payoff_mat.shape[0], payoff_mat.shape[1]
    alpha_row = np.ones(m)
    alpha_col = np.ones(n)
    probs_row = np.random.dirichlet(alpha_row)
    probs_col = np.random.dirichlet(alpha_col)

    payoff_p1 = np.dot(probs_row.T, np.dot(payoff_mat, probs_col))

    return [probs_row, probs_col], [payoff_p1, -payoff_p1]


class hack_mss:
    def __init__(self, mss, global_payoff_mat):
        self.mss = mss
        self.global_payoff_mat = global_payoff_mat

    def real_mss(self, payoff_mat):
        if len(payoff_mat) == 3:
            probs_, payoff_ = zero_sum_2p_game(self.global_payoff_mat[:3, :4])
            return [probs_[0], probs_[0].copy()], [0.0, -0.0]
        else:
            return self.mss(payoff_mat)
