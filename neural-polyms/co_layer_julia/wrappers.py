import numpy as np
from polym_lcp import polym_lcp_solver
from nf_and_polymatrix import PolymatrixGame, NormalFormGame
import pickle

print("successfully imported my wrappers!")


def afunction(x): return x + 2

def flat_lcp_solve(
        flattened_polym,
        number_of_players,
        number_of_actions
):
    pmg = PolymatrixGame.from_flattened(
        flattened_polym,
        number_of_players,
        number_of_actions
    )

    sol = polym_lcp_solver(pmg)
    flat_sol = np.concatenate(sol)
    return (flat_sol)

def flat_judge_eq(
        flattened_eq,
        flattened_nfg,
        number_of_players,
        number_of_actions
):
    nfg = NormalFormGame.from_flattened(
        flattened_nfg, number_of_players, number_of_actions)
    actions = [
        flattened_eq[sum(nfg.actions[:p]
                         ):sum(nfg.actions[:p + 1])]
        for p in range(nfg.players)
        ]
    
    best_payoffs = nfg.best_responses_and_payoffs(actions)[1]
    our_payoffs = nfg.payoffs_of_actions(actions)
    loss = sum(best_payoffs) - sum(our_payoffs)
    return(loss)

def flat_judge_eq_grad(
        flattened_eq,
        flattened_nfg,
        number_of_players,
        number_of_actions
):
    nfg = NormalFormGame.from_flattened(
        flattened_nfg, number_of_players, number_of_actions)
    actions = [
        flattened_eq[sum(nfg.actions[:p]
                         ):sum(nfg.actions[:p + 1])]
        for p in range(nfg.players)
        ]
    
    payoff_vectors = nfg.payoff_vectors(actions)
    payoff_grad = np.concatenate([
        np.mean(pv) - pv
        # ^ seems to be the right way around
        # v is the direction of loss decrease
        # pv - np.mean(pv)
        for pv in payoff_vectors
        ])
    return(payoff_grad)

def flat_from_file(filename):
    nfg = NormalFormGame.from_gam_file(filename)
    return nfg.flatten()

def pickle_wrapper_flat(filename):
    with open(filename, 'rb') as file:
        games: NormalFormGame = pickle.load(file)
    flat_games = [
        game.flatten()
        for game in games
    ]
    return flat_games
