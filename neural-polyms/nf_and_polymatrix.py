import numpy as np
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
    
    # in development. doesn't work. contains falsehoods
    def check_if_player_action_polymatrixable(self, my_player, my_action):
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

        x, res, rank, s = np.linalg.lstsq(hh_actions, combined_payoffs, rcond=1e-5)

        # print("linalg pinv:\n", np.dot(np.linalg.pinv(hh_actions), combined_payoffs))
        # print("linalg lstsq:\n", np.linalg.lstsq(hh_actions, combined_payoffs, rcond=1e-5))

        print("stats")
        # print(hh_actions @ x - combined_payoffs)
        # print(hh_actions @ np.dot(np.linalg.pinv(hh_actions), combined_payoffs) - combined_payoffs)
        print((hh_actions @ x - hh_actions @ np.dot(np.linalg.pinv(hh_actions), combined_payoffs)))
        # print(combined_payoffs)
        print(res)

        return False

    # in development. doesn't work. contains falsehoods
    def check_if_polymatrix(self):
        polymatrixable_so_far = True
        for p1 in range(self.players):
            for a1 in range(self.actions[p1]):
                polymatrixable_so_far = self.check_if_player_action_polymatrixable(
                    p1, a1
                )
        return polymatrixable_so_far
    
    def __str__(self) -> str:
        str_builder =""
        for player in range(self.players):
            for action_combination in product(*[range(a) for a in self.actions]):
                str_builder += str(action_combination) + " : "
                str_builder += str(self.entries[player][(action_combination)]) + "\n\n"
        return str_builder


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

        return cls(nf.players, nf.actions, polymatrix_builder)

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
            # sum(
            #     [
            #         actions[p1].T @ self.polymatrix[(p1, p2)] @ actions[p2]
            #         for p2 in range(self.players)
            #         if p1 != p2
            #     ]
            # )
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
