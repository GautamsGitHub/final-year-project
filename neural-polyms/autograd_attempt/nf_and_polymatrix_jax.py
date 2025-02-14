import jax.numpy as np
from itertools import product
from abc import ABC, abstractmethod
from math import isclose


class Game(ABC):
    def __init__(self, players, actions) -> None:
        self.players = players
        self.actions = actions

    @abstractmethod
    def payoff_pure(self, player, pure_actions):
        return 0.0

    def two_player_deviations(self, p1, p2, pure_actions):
        return {
            (a1, a2): tuple([
                a1 if p3 == p1 else a2 if p3 == p2 else pure_actions[p3]
                for p3 in range(self.players)
            ])
            for a1 in range(self.actions[p1])
            for a2 in range(self.actions[p2])
        }

    def two_player_deviation_payoffs(self, player, p2, pure_actions):
        deviations = self.two_player_deviations(player, p2, pure_actions)
        return np.array([
            [
                self.payoff_pure(player, deviations[(a1, a2)])
                for a2 in range(self.actions[p2])
            ]
            for a1 in range(self.actions[player])
        ])
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Game): return False
        if self.players != value.players: return False
        for player in range(self.players):
            if self.actions[player] != value.actions[player]: return False
        for player in range(self.players):
            for action_combination in product(*[range(a) for a in self.actions]):
                if not isclose(
                    self.payoff_pure(player, action_combination),
                    value.payoff_pure(player, action_combination)
                ): return False
        return True


class NormalFormGame(Game):

    def __init__(self, players, actions, entries) -> None:
        super().__init__(players, actions)
        self.entries = entries

    @classmethod
    def from_gam_file(cls, filename):
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
    
    @classmethod
    def from_flattened(cls, flattened_nfg, number_of_players, number_of_actions):
        combs_pp = pow(number_of_actions, number_of_players)
        entries = [
            dict(zip(product(
                *[range(number_of_actions) for _ in range(number_of_players)]),
                flattened_nfg[player * combs_pp : (player + 1) *combs_pp]))
            for player in range(number_of_players)
        ]
        return cls(
            number_of_players,
            [number_of_actions for _ in range(number_of_players)],
            entries
        )
    
    def flatten(self):
        flattened_nfg = np.concatenate([
            np.array(list(m.values()))
            for m in self.entries
        ])
        return flattened_nfg

    def payoff_pure(self, player, pure_actions):
        return self.entries[player][pure_actions]

    def hh_payoff_player(self, my_player, my_action):
        action_combinations = product(*(
            [range(self.actions[p]) if p != my_player else [my_action]
             for p in range(self.players)]
        ))
        hh_actions_and_payoffs = np.vstack([
            np.hstack(
                [
                    np.eye(self.actions[p])[action_combination[p]]
                    for p in range(self.players) if p != my_player
                ] + [self.payoff_pure(my_player, action_combination)]
                # ] + [self.entries[my_player][action_combination]]
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

        payoffs = {label: payoff for label, payoff in zip(
            payoff_labels, hh_payoffs_array)}

        return payoffs
    
    def check_if_polymatrix(self):
        return self == PolymatrixGame.from_nf(self)
    
    def __str__(self) -> str:
        str_builder =""
        for player in range(self.players):
            for action_combination in product(*[range(a) for a in self.actions]):
                str_builder += str(action_combination) + " : "
                str_builder += str(self.entries[player][(action_combination)]) + "\n\n"
        return str_builder
    
    def payoff_vectors(self, actions):
        return [
            [
                sum([
                    np.prod(
                        np.array([actions[p][action_combination[p]] if p != my_player else 1
                        for p in range(self.players)])
                    ) * self.entries[my_player][action_combination]
                    for action_combination in product(*([
                        range(self.actions[other_player]) if other_player != my_player else [my_action]
                        for other_player in range(self.players)
                    ]))
                ])
                for my_action in range(self.actions[my_player])
            ]
            for my_player in range(self.players)
        ]

    def best_responses_and_payoffs(self, actions):
        pvs = self.payoff_vectors(actions)
        return (
            [np.argmax(np.array(pv)) for pv in pvs],
            [np.max(np.array(pv)) for pv in pvs]
        )
    
    def payoffs_of_actions(self, actions):
        return [
            sum([
                np.prod(
                    np.array([actions[p][action_combination[p]]
                    for p in range(self.players)])
                ) * self.entries[my_player][action_combination]
                for action_combination in product(*([
                    range(self.actions[p])
                    for p in range(self.players)
                ]))
            ])
            for my_player in range(self.players)
        ]


class PolymatrixGame(Game):

    def __str__(self) -> str:
        print("hello world")
        str_builder = ""
        for k, v in self.polymatrix.items():
            str_builder += str(k) + ":\n"
            str_builder += str(v) + "\n\n"
        return str_builder

    def __init__(self, players, actions, polymatrix) -> None:
        super().__init__(players, actions)
        self.polymatrix = polymatrix

    @classmethod
    def from_nf(cls, nf: NormalFormGame):
        polymatrix_pre_builder = {
            (p1, p2, a1, a2): payoffs.get((p2, a2), -np.inf) 
            for p1 in range(nf.players)
            for a1 in range(nf.actions[p1])
            for payoffs in [nf.hh_payoff_player(p1, a1)]
            for p2 in range(nf.players)
            if p1 != p2
            for a2 in range(nf.actions[p2])
        }
        polymatrix_builder = {
            (p1, p2) : np.array([
                [
                    polymatrix_pre_builder[(p1, p2, a1, a2)]
                    for a2 in range(nf.actions[p2])
                ]
                for a1 in range(nf.actions[p1])
            ])
            for p1 in range(nf.players)
            for p2 in range(nf.players)
            if p1 != p2
        }

        return cls(nf.players, nf.actions, polymatrix_builder)
    
    @classmethod
    def from_flattened(cls, flattened_polym, number_of_players, number_of_actions):
        flat_mats_mat = np.reshape(
            flattened_polym,
                (
                    number_of_players * (number_of_players - 1),
                    number_of_actions*number_of_actions
                    )
                )
        mats_list = [
            np.reshape(flat_ar, (number_of_actions, number_of_actions))
            for flat_ar in flat_mats_mat
        ]
        pm_keys = [
            (p1, p2)
            for p1 in range(number_of_players)
            for p2 in range(number_of_players)
            if p1 != p2
        ]
        pm_entries = dict(zip(pm_keys, mats_list))
        return(cls(
            number_of_players, 
            [number_of_actions for _ in range(number_of_players)], 
            pm_entries
            ))
    
    def flatten(self):
        flattened_polym = np.concatenate([
            np.ravel(self.polymatrix[p1, p2])
            for p1 in range(self.players)
            for p2 in range(self.players)
            if p1 != p2
        ])
        return flattened_polym

    def payoff_pure(self, player, pure_actions):
        return sum(
            [
                self.polymatrix[(player, p2)
                                ][pure_actions[player]][pure_actions[p2]]
                for p2 in range(self.players)
                if player != p2
            ]
        )

    def to_paired_polymatrix(self):
        # for paired, in the second matrix, the second player is still the column player
        return {
            (p1, p2): (
                self.polymatrix[(p1, p2)],
                self.polymatrix[(p2, p1)].T
            )
            for p1 in range(self.players - 1)
            for p2 in range(p1 + 1, self.players)
        }

    def payoffs_of_actions(self, actions):
        payoff_vectors = self.payoff_vectors(actions)
        return [
            np.inner(actions[p1], payoff_vectors[p1])
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
                    comb: sum([
                        self.polymatrix[(p1, p2)][comb[p1]][comb[p2]]
                        for p2 in range(self.players)
                        if p1 != p2
                    ])
                    for comb in product(*[range(a) for a in self.actions])
                }
                for p1 in range(self.players)
            ]
        )
    
    def range_of_payoffs(self):
        min_p = min([np.min(M) for M in self.polymatrix.values()])
        max_p = max([np.max(M) for M in self.polymatrix.values()])
        return (min_p, max_p)
    
    def to_one_big_matrix(self, costs=True, low_avoider=2.0):
        positive_cost_maker = self.range_of_payoffs()[1] + low_avoider
        sign = -1 if costs else 1
        M = np.vstack([
            np.hstack([
                np.zeros((self.actions[player], self.actions[player])) if p2 == player
                else (positive_cost_maker + sign * self.polymatrix[(player, p2)])
                for p2 in range(self.players)
                ])
            for player in range(self.players)
            ])
        return M