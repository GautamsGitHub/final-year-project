import jax.numpy as np
from jax.random import PRNGKey
from jax import grad
from polym_lcp_jax import polym_lcp_solver
from nf_and_polymatrix_jax import NormalFormGame, PolymatrixGame
import flax.linen as nn
import optax

rng_key = PRNGKey(0)

number_of_players = 2
number_of_actions = 3

def matrices_solve(flattened_polym):
    pmg = PolymatrixGame.from_flattened(
        flattened_polym,
        number_of_players,
        number_of_actions
        )

    sol = polym_lcp_solver(pmg)
    return(sol)

class PolymatrixLayer(nn.Module):
    @nn.compact
    def __call__(self, x):
        return matrices_solve(x)
    

class GamePolymNeuralNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        n_inputs = int(pow(number_of_actions, number_of_players)) * number_of_players
        n_polym_layer = number_of_players * (number_of_players - 1) * number_of_actions * number_of_actions
        n_outputs = number_of_actions * number_of_players
        x = nn.Dense(features=16)(x)
        x = nn.relu(x)
        x = nn.Dense(features=16)(x)
        x = nn.relu(x)
        x = nn.Dense(features=n_polym_layer)(x)
        x = PolymatrixLayer()(x)
        return x


def judge_polym(nfg: NormalFormGame, polym: PolymatrixGame):

    polym_eq = polym_lcp_solver(polym)
    ppeq = nfg.payoffs_of_actions(polym_eq)
    pvs = nfg.payoff_vectors(polym_eq)
    brs, pbrs = nfg.best_responses_and_payoffs(polym_eq)
    print("proposed eq actions:", polym_eq)
    print("in the normal form game")
    print("payoff vectors:", pvs)
    print("best responses:", brs)
    print("payoffs of best responses", pbrs)
    print("payoffs at our attempt", ppeq)

    return 0

def judgement_of_proposed_eq(nfg : NormalFormGame, eq):
    return 0


filename = "../rps.gam"
# filename = "../gemp_re/games/handmade.gam"
# filename = "../gemp_re/games/polym4.gam"
normal_form_game = NormalFormGame.from_gam_file(filename)
number_of_players = normal_form_game.players
number_of_actions = normal_form_game.actions[0]
flattened_nfg = normal_form_game.flatten()

polymatrix_game = PolymatrixGame.from_nf(normal_form_game)
flattened_polym = polymatrix_game.flatten()

model = GamePolymNeuralNetwork()
n_inputs = int(pow(number_of_actions, number_of_players)) * number_of_players
x0 = np.empty(n_inputs)
params = model.init(rng_key, x0)
# params = model.init(rng_key, flattened_nfg)

# output = model.apply(params, flattened_nfg)

# print(output)

# actions = output
# # actions = [all_actions[sum(normal_form_game.actions[:p]):sum(normal_form_game.actions[:p + 1])] for p in range(normal_form_game.players)]

# best_payoffs = normal_form_game.best_responses_and_payoffs(actions)[1]
# our_payoffs = normal_form_game.payoffs_of_actions(actions)
# payoff_vectors = normal_form_game.payoff_vectors(actions)
# payoff_grad = np.concatenate([
#     # np.mean(pv) - pv
#     pv - np.mean(pv)
#     for pv in payoff_vectors
#     ])

# loss = l2_loss(our_payoffs, best_payoffs)

def compute_loss(params, flattened_nfg):
    nfg = NormalFormGame.from_flattened(flattened_nfg, number_of_players, number_of_actions)
    output = model.apply(params, flattened_nfg)
    actions = output
    best_payoffs = nfg.best_responses_and_payoffs(actions)[1]
    our_payoffs = nfg.payoffs_of_actions(actions)
    loss = sum(best_payoffs) - sum(our_payoffs)
    return(loss)



optimiser = optax.sgd(0.1)

opt_state = optimiser.init(params)

for i in range(200):
    g = grad(compute_loss)(params, flattened_nfg)
    updates, opt_state = optimiser.update(g, opt_state)
    params = optax.apply_updates(params, updates)
    if i % 10 == 0:
        l = compute_loss(params, flattened_nfg)
        print("current loss:", l)


print("done")
