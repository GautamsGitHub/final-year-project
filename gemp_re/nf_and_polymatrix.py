import numpy as np
from itertools import product

class NormalFormGame:

    def __init__(self, players, actions, entries) -> None:
        self.players = players
        self.actions = actions
        self.entries = entries

    @classmethod
    def from_gam_file(cls, filename) -> None:
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

        return cls(players, actions, entries)

    def hh_payoff_player(self, my_player, my_action):
        action_combinations = product(*(
            [range(self.actions[p]) if p != my_player else [my_action] for p in range(self.players)] 
            ))
        hh_actions_and_payoffs = np.vstack([
            np.hstack(
                [
                    np.eye(self.actions[p])[action_combination[p]]
                    for p in range(self.players) if p != my_player
                ] + [self.entries[my_player][action_combination]]
            )
            for action_combination in action_combinations
        ])
        hh_actions = hh_actions_and_payoffs[:, :-1]
        combined_payoffs = hh_actions_and_payoffs[:, -1]

        # different ways to solve the simultaneous equations
        hh_payoffs_array = np.dot(np.linalg.pinv(hh_actions), combined_payoffs)
        # hh_payoffs_array = np.linalg.solve(hh_actions[:sum(actions)], combined_payoffs[:sum(actions)])
        # hh_payoffs_array = np.linalg.lstsq(hh_actions, combined_payoffs, rcond=1.0)
        
        payoff_labels = [
            (p, a)
            for p in range(self.players) if p != my_player
            for a in range(self.actions[p])
            ]
        
        payoffs = {label : payoff for label, payoff in zip(payoff_labels, hh_payoffs_array)}

        return payoffs


class PolymatrixGame:

    def __init__(self, nf: NormalFormGame) -> None:
        polymatrix_builder = {
            (p1, p2): np.full((nf.actions[p1], nf.actions[p2]), -np.inf)
            for p1 in range(nf.players)
            for p2 in range(nf.players)
            if p1 != p2
        }
        for p1 in range(nf.players):
            for a1 in range(nf.actions[p1]):
                payoffs = nf.hh_payoff_player(p1, a1)
                for ((p2, a2), payoff) in payoffs.items():
                    polymatrix_builder[(p1, p2)][a1][a2] = payoff

        self.players = nf.players
        self.actions = nf.actions
        self.polymatrix = polymatrix_builder

    def to_paired_polymatrix(self):
        # for paired, in the second matrix, the second player is still the column player
        return {
            (p1, p2) : (
                self.polymatrix[(p1, p2)],
                self.polymatrix[(p2, p1)].T
                )
            for p1 in range(self.players - 1)
            for p2 in range(p1 + 1, self.players)
        }
    
    def payoffs_of_actions(self, actions):
        return [
            sum(
                [
                    actions[p1].T @ self.polymatrix[(p1, p2)] @ actions[p2]
                    for p2 in range(self.players)
                    if p1 != p2
                ]
            )
            for p1 in range(self.players)
        ]
    
    def payoff_vectors(self, actions):
        return [
            sum(
                [
                    self.polymatrix[(p1, p2)] @ actions[p2]
                    for p2 in range(self.players)
                    if p1 != p2
                ]
            )
            for p1 in range(self.players)
        ]
    
    def best_responses_and_payoffs(self, actions):
        pvs = self.payoff_vectors(actions)
        return (
            [np.argmax(pv) for pv in pvs],
            [np.max(pv) for pv in pvs]
        )
    
    def to_nfg(self):
        return NormalFormGame(
            self.players,
            self.actions,
            [
                {
                    comb : sum([
                        self.polymatrix[(p1, p2)][comb[p1]][comb[p2]]
                        for p2 in range(self.players)
                        if p1 != p2
                        ])
                    for comb in product(*[range(a) for a in self.actions])
                }
                for p1 in range(self.players)
            ]
        )
