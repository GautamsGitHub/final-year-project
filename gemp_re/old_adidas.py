"""
Copyright 2020 ADIDAS Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from scipy import special

def simplex_project_grad(g):
	"""Project gradient onto tangent space of simplex."""
	return g - g.sum() / g.size

def gradients_qre_nonsym(dist, y, anneal_steps, payoff_matrices,
                         num_players, temp=0., proj_grad=True,
                         exp_thresh=1e-3, lrs=(1e-2, 1e-2),
                         logit_clip=-1e5):
    """Computes exploitablity gradient and aux variable gradients.

    Args:
    dist: list of 1-d np.arrays, current estimate of nash
    y: list of 1-d np.arrays, current est. of payoff gradient
    anneal_steps: int, elapsed num steps since last anneal
    payoff_matrices: dict with keys as tuples of agents (i, j) and
    values of (2 x A x A) arrays, payoffs for each joint action.
    keys are sorted and arrays are indexed in the same order.
    num_players: int, number of players
    temp: non-negative float, default 0.
    proj_grad: bool, if True, projects dist gradient onto simplex
    exp_thresh: ADI threshold at which temp is annealed
    lrs: tuple of learning rates (lr_x, lr_y)
    logit_clip: float, minimum allowable logit
    Returns:
    gradient of ADI w.r.t. (dist, y, anneal_steps)
    temperature (possibly annealed, i.e., reduced)
    unregularized ADI (stochastic estimate)
    shannon regularized ADI (stochastic estimate)
    """
    # first compute policy gradients and player effects (fx)
    policy_gradient = []
    other_player_fx = []
    grad_y = []
    unreg_exp = []
    reg_exp = []
    for i in range(num_players):
        nabla_i = np.zeros_like(dist[i])
        for j in range(num_players):
            if j == i:
                continue
            if i < j:
                hess_i_ij = payoff_matrices[(i, j)][0]
            else:
                hess_i_ij = payoff_matrices[(j, i)][1].T

            nabla_ij = hess_i_ij.dot(dist[j])
            nabla_i += nabla_ij / float(num_players - 1)

        grad_y.append(y[i] - nabla_i)

        if temp >= 1e-3:  # numerical under/overflow for temp < 1e-3
            br_i = special.softmax(y[i] / temp)
            br_i_mat = (np.diag(br_i) - np.outer(br_i, br_i)) / temp
            log_br_i_safe = np.clip(np.log(br_i), logit_clip, 0)
            br_i_policy_gradient = nabla_i - temp * (log_br_i_safe + 1)
        else:
            power = np.inf
            s_i = np.linalg.norm(y[i], ord=power)
            br_i = np.zeros_like(dist[i])
            maxima_i = (y[i] == s_i)
            br_i[maxima_i] = 1. / maxima_i.sum()
            br_i_mat = np.zeros((br_i.size, br_i.size))
            br_i_policy_gradient = np.zeros_like(br_i)

        policy_gradient_i = np.array(nabla_i)
        if temp > 0:
            log_dist_i_safe = np.clip(np.log(dist[i]), logit_clip, 0)
            policy_gradient_i -= temp * (log_dist_i_safe + 1)
        policy_gradient.append(policy_gradient_i)

        unreg_exp_i = np.max(y[i]) - y[i].dot(dist[i])
        unreg_exp.append(unreg_exp_i)

        entr_br_i = temp * special.entr(br_i).sum()
        entr_dist_i = temp * special.entr(dist[i]).sum()

        reg_exp_i = y[i].dot(br_i - dist[i]) + entr_br_i - entr_dist_i
        reg_exp.append(reg_exp_i)

        other_player_fx_i = (br_i - dist[i])
        other_player_fx_i += br_i_mat.dot(br_i_policy_gradient)
        other_player_fx.append(other_player_fx_i)

    # then construct ADI gradient
    grad_dist = []
    for i in range(num_players):

        grad_dist_i = -policy_gradient[i]
        for j in range(num_players):
            if j == i:
                continue
            if i < j:
                hess_j_ij = payoff_matrices[(i, j)][1]
            else:
                hess_j_ij = payoff_matrices[(j, i)][0].T

            grad_dist_i += hess_j_ij.dot(other_player_fx[j])

        if proj_grad:
            grad_dist_i = simplex_project_grad(grad_dist_i)

        grad_dist.append(grad_dist_i)

    unreg_exp_mean = np.mean(unreg_exp)
    reg_exp_mean = np.mean(reg_exp)

    _, lr_y = lrs
    if (reg_exp_mean < exp_thresh) and (anneal_steps >= 1 / lr_y):
        temp = np.clip(temp / 2., 0., 1.)
        if temp < 1e-3:  # consistent with numerical issue above
            temp = 0.
        grad_anneal_steps = 0
        # originally was
        # grad_anneal_steps = -anneal_steps
    else:
        grad_anneal_steps = anneal_steps + 1
        # originally was
        # grad_anneal_steps = 1

    return ((grad_dist, grad_y, grad_anneal_steps), temp,
            unreg_exp_mean, reg_exp_mean)

def gradients_ate_sym(dist, y, anneal_steps, payoff_matrices,
                      num_players, p=1, proj_grad=True,
                      exp_thresh=1e-3, lrs=(1e-2, 1e-2)):
    """Computes ADI gradient and aux variable gradients.

    Args:
    dist: list of 1-d np.arrays, current estimate of nash
    y: list of 1-d np.arrays, current est. of payoff gradient
    anneal_steps: int, elapsed num steps since last anneal
    payoff_matrices: dict with keys as tuples of agents (i, j) and
    values of (2 x A x A) arrays, payoffs for each joint action.
    keys are sorted and arrays are indexed in the same order.
    num_players: int, number of players
    p: float in [0, 1], Tsallis entropy-regularization
    proj_grad: bool, if True, projects dist gradient onto simplex
    exp_thresh: ADI threshold at which p is annealed
    lrs: tuple of learning rates (lr_x, lr_y)
    Returns:
    gradient of ADI w.r.t. (dist, y, anneal_steps)
    temperature, p (possibly annealed, i.e., reduced)
    unregularized ADI (stochastic estimate)
    tsallis regularized ADI (stochastic estimate)
    """
    nabla = payoff_matrices[0].dot(dist)
    if p >= 1e-2:  # numerical under/overflow when power > 100.
        power = 1. / float(p)
        s = np.linalg.norm(y, ord=power)
        if s == 0:
            br = np.ones_like(y) / float(y.size)  # uniform dist
        else:
            br = (y / s)**power
    else:
        power = np.inf
        s = np.linalg.norm(y, ord=power)
        br = np.zeros_like(dist)
        maxima = (y == s)
        br[maxima] = 1. / maxima.sum()

    unreg_exp = np.max(y) - y.dot(dist)
    br_inv_sparse = 1 - np.sum(br**(p + 1))
    dist_inv_sparse = 1 - np.sum(dist**(p + 1))
    entr_br = s / (p + 1) * br_inv_sparse
    entr_dist = s / (p + 1) * dist_inv_sparse
    reg_exp = y.dot(br - dist) + entr_br - entr_dist

    entr_br_vec = br_inv_sparse * br**(1 - p)
    entr_dist_vec = dist_inv_sparse * dist**(1 - p)
    policy_gradient = nabla - s * dist**p
    other_player_fx = (br - dist)
    other_player_fx += 1 / (p + 1) * (entr_br_vec - entr_dist_vec)
    other_player_fx_translated = payoff_matrices[1].dot(other_player_fx)
    grad_dist = -policy_gradient
    grad_dist += (num_players - 1) * other_player_fx_translated
    if proj_grad:
        grad_dist = simplex_project_grad(grad_dist)
    grad_y = y - nabla

    _, lr_y = lrs
    if (reg_exp < exp_thresh) and (anneal_steps >= 1 / lr_y):
        p = np.clip(p / 2., 0., 1.)
        if p < 1e-2:  # consistent with numerical issue above
            p = 0.
        grad_anneal_steps = -anneal_steps
    else:
        grad_anneal_steps = 1

    return ((grad_dist, grad_y, grad_anneal_steps), p, unreg_exp, reg_exp)
