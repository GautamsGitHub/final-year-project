from adidas_simplified import simplex_project_grad, gradients_qre_nonsym, gradients_ate_sym
import numpy as np
from scipy.special import softmax
from itertools import product

def get_game(filename):
    entries = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        combined = [
            token
            for line in lines
            for token in line.split()
            ]
        
        i = iter(combined)
        players = int(next(i))
        actions = [int(next(i)) for _ in range(players)]
        entries = [
            {
                tuple(reversed(action_combination)): float(next(i))
                for action_combination in product(*[range(a) for a in actions])
            }
            for _ in range(players)
        ]

    return (players, actions, entries)

def nf_to_polymatrix(nf):
    (players, actions, entries) = nf

    polymatrix = {
        (p1, p2): np.full((actions[p1], actions[p2]), -np.inf)
        for p1 in range(players)
        for p2 in range(players)
        if p1 != p2
    }
    for p1 in range(players):
        for a1 in range(actions[p1]):
            payoffs = hh_payoff_player(nf, p1, a1)
            for ((p2, a2), payoff) in payoffs.items():
                polymatrix[(p1, p2)][a1][a2] = payoff
    
    return polymatrix


def polymatrix_to_paired_polym(players, polymatrix):
    return {
        (p1, p2) : (polymatrix[(p1, p2)], polymatrix[(p2, p1)])
        for p1 in range(players - 1)
        for p2 in range(p1 + 1, players)
    }


def hh_payoff_player(nf, my_player, my_action):
    (players, actions, entries) = nf
    # np.zeros((np.prod(actions), sum(actions)))
    action_combinations = product(*(
        [range(actions[p]) if p != my_player else [my_action] for p in range(players)] 
        ))
    hh_actions_and_payoffs = np.vstack([
        np.hstack(
            [
                np.eye(actions[p])[action_combination[p]]
                for p in range(players) if p != my_player
            ] + [entries[my_player][action_combination]]
        )
        for action_combination in action_combinations
    ])
    hh_actions = hh_actions_and_payoffs[:, :-1]
    combined_payoffs = hh_actions_and_payoffs[:, -1]

    # hh_payoffs_array = np.linalg.solve(hh_actions[:sum(actions)], combined_payoffs[:sum(actions)])
    hh_payoffs_array = np.dot(np.linalg.pinv(hh_actions), combined_payoffs)
    # hh_payoffs_array = np.linalg.lstsq(hh_actions, combined_payoffs, rcond=1.0)
    
    payoff_labels = [
        (p, a)
        for p in range(players) if p != my_player
        for a in range(actions[p])
        ]
    
    payoffs = {label : payoff for label, payoff in zip(payoff_labels, hh_payoffs_array)}
    return payoffs

filename = "compound.gam"
nf = get_game(filename)
polymatrix = nf_to_polymatrix(nf)
paired_polym = polymatrix_to_paired_polym(nf[0], polymatrix)
print(paired_polym)