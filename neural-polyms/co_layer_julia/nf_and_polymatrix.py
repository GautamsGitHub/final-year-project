import numpy as np
from itertools import product
from abc import ABC, abstractmethod
from math import isclose


class Game(ABC):
    
    def __init__(self, players, actions) -> None:
        """Creates a game.

        Args:
            players : Number of players in game.
            actions : List of how many actions each player has.
        """        
        self.players = players
        self.actions = actions

    @abstractmethod
    def payoff_pure(self, player, pure_actions):
        """Payoff to player at pure action combination.

        Args:
            player : Player to get payoff of.
            pure_actions : Pure action combination being played.

        Returns:
            _type_: Payoff to player.
        """        
        return 0.0

    def two_player_deviations(self, p1, p2, pure_actions):
        """
        Action combinations that result from two players
        deviating.

        Args:
            p1 : First deviating player.
            p2 : Second deviating player.
            pure_actions : 
                Pure action combination orginially being played.

        Returns:
            _type_: Action combination at each 2 player deviation.
        """        
        return {
            (a1, a2): tuple([
                a1 if p3 == p1 else a2 if p3 == p2 else pure_actions[p3]
                for p3 in range(self.players)
            ])
            for a1 in range(self.actions[p1])
            for a2 in range(self.actions[p2])
        }

    def two_player_deviation_payoffs(self, player, p2, pure_actions):
        """Payoff matrix to player at 2 player deviations.

        Args:
            player : Deviating player whose payoffs are given.
            p2 : Other deviating player
            pure_actions : Pure action combination orginially being played.

        Returns:
            _type_: Payoff matrix to player at 2 player deviations.
        """        
        deviations = self.two_player_deviations(player, p2, pure_actions)
        return np.array([
            [
                self.payoff_pure(player, deviations[(a1, a2)])
                for a2 in range(self.actions[p2])
            ]
            for a1 in range(self.actions[player])
        ])

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Game):
            return False
        if self.players != value.players:
            return False
        for player in range(self.players):
            if self.actions[player] != value.actions[player]:
                return False
        for player in range(self.players):
            for action_combination in product(*[range(a) for a in self.actions]):
                if not isclose(
                    self.payoff_pure(player, action_combination),
                    value.payoff_pure(player, action_combination),
                    abs_tol=1e-5
                ):
                    return False
        return True


class NormalFormGame(Game):

    def __init__(self, players, actions, entries) -> None:
        """Creates a Normal Form Game.

        Args:
            players (_type_): Number of players.
            actions (_type_): 
                List of how many actions each player has.
            entries (_type_): 
                Payoff to player at each pure action
                combination for each player.
        """        
        super().__init__(players, actions)
        self.entries = entries

    @classmethod
    def from_gam_file(cls, filename):
        """
        Creates a Normal Form Game from a .gam file with format
        as specified by GameTracer.

        Args:
            filename (_type_): Name of gam file.

        Returns:
            _type_: New Normal Form Game.
        """        
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
        """
        Creates a Normal Form Game
        from a flattened Normal Form Game.

        Args:
            flattened_nfg (_type_): 
                Array of all of the entries in a Normal Form Game.
                With all of the payoffs of one player before the
                next. With the player actions going up starting at
                the last player.
            number_of_players (_type_):
                Number of players in the flattened game.
            number_of_actions (_type_):
                Number of actions per player.
                All players must have the same number of actions.

        Returns:
            _type_: New Normal Form Game.
        """        
        combs_pp = pow(number_of_actions, number_of_players)
        entries = [
            dict(zip(product(
                *[range(number_of_actions) for _ in range(number_of_players)]),
                flattened_nfg[player * combs_pp: (player + 1) * combs_pp]))
            for player in range(number_of_players)
        ]
        return cls(
            number_of_players,
            [number_of_actions for _ in range(number_of_players)],
            entries
        )

    def flatten(self):
        """
        Flatten this into an array.

        Returns:
            _type_: 
                Array of all of the entries in a Normal Form Game.
                With all of the payoffs of one player before the
                next. With the player actions going up starting at
                the last player.
        """        
        self.sort_entries()
        flattened_nfg = np.concatenate([
            np.array(list(m.values()))
            for m in self.entries
        ])
        return flattened_nfg

    def payoff_pure(self, player, pure_actions):
        return self.entries[player][pure_actions]

    def hh_payoff_player(self, my_player, my_action):
        """
        Approximates the payoffs to a player when they play
        an action with values at the actions of other players
        that try to sum to the payoff.
        Precise when the game can be represented with a polymatrix.

        Args:
            my_player (_type_): _description_
            my_action (_type_): _description_

        Returns:
            _type_:
                Dictionary giving an approximate component
                of the payoff at each action for each other player.
        """        
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

    def check_if_polymatrix(self) -> bool:
        """
        Check if this game can be represented by a Polymatrix
        without loss of

        Returns:
            bool: True if it can.
        """
        return self == PolymatrixGame.from_nf(self)

    def __str__(self) -> str:
        str_builder = ""
        for player in range(self.players):
            for action_combination in product(*[range(a) for a in self.actions]):
                str_builder += str(action_combination) + " : "
                str_builder += str(self.entries[player]
                                   [(action_combination)]) + "\n\n"
        return str_builder

    def payoff_vectors(self, actions):
        """
        Payoffs to each action for each player if the other
        players stay at their mixed actions.

        Args:
            actions (_type_):
                Mixed action combination to deviate from.

        Returns:
            _type_: Payoffs to deviations for each player.
        """        
        return [
            [
                sum([
                    np.prod(
                        [actions[p][action_combination[p]] if p != my_player else 1
                         for p in range(self.players)]
                    ) * self.entries[my_player][action_combination]
                    for action_combination in product(*([
                        range(self.actions[other_player]) if other_player != my_player else [
                            my_action]
                        for other_player in range(self.players)
                    ]))
                ])
                for my_action in range(self.actions[my_player])
            ]
            for my_player in range(self.players)
        ]

    def best_responses_and_payoffs(self, actions):
        """
        Best Responses to a mixed action combination
        as the action numbers of each player and
        the payoff that player gets if they unilaterally
        deviate to that Best Response.

        Args:
            actions (_type_): Mixed action combination.

        Returns:
            _type_: 
                Tuple with first entry best responses
                and second entry their payoffs.
        """        
        pvs = self.payoff_vectors(actions)
        return (
            [np.argmax(pv) for pv in pvs],
            [np.max(pv) for pv in pvs]
        )

    def payoffs_of_actions(self, actions):
        """
        Payoff to each player at a mixed action combination.

        Args:
            actions (_type_): Mixed action combination

        Returns:
            _type_: Payoffs to each player.
        """        
        return [
            sum([
                np.prod(
                    [actions[p][action_combination[p]]
                     for p in range(self.players)]
                ) * self.entries[my_player][action_combination]
                for action_combination in product(*([
                    range(self.actions[p])
                    for p in range(self.players)
                ]))
            ])
            for my_player in range(self.players)
        ]
    
    def sort_entries(self):
        """
        Puts the payoffs at the entries in order
        so the player action numbers go up, starting at
        the last player.
        """
        sorted_entries = [
            {
                action_combination : d[action_combination] 
                for action_combination in product(*[range(a) for a in self.actions])
            }
            for d in self.entries
        ]
        self.entries = sorted_entries
    
    def __add__(self, other):
        assert(isinstance(other, NormalFormGame))
        combined_flat = self.flatten() + other.flatten()
        return NormalFormGame.from_flattened(combined_flat, self.players, self.actions[0])


class PolymatrixGame(Game):

    def __str__(self) -> str:
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
        """
        Creates a Polymatrix approximation to a
        Normal Form Game. Precise if possible.

        Args:
            nf (NormalFormGame): Normal Form Game to approximate.

        Returns:
            _type_: New Polymatrix Game.
        """        
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

    @classmethod
    def from_flattened(cls, flattened_polym, number_of_players, number_of_actions):
        """
        Creates a Polymatrix Game
        from a flattened Polymatrix Game.

        Args:
            flattened_nfg (_type_): 
                Array of all of the entries in each matrix
                of the Polymatrix.
                With all of the payoffs of one player before the
                next. With the opponent going up.
            number_of_players (_type_):
                Number of players in the flattened game.
            number_of_actions (_type_):
                Number of actions per player.
                All players must have the same number of actions.

        Returns:
            _type_: New Polymatrix Game.
        """    
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
        return (cls(
            number_of_players,
            [number_of_actions for _ in range(number_of_players)],
            pm_entries
        ))

    def flatten(self):
        """
        Flattens Polymatrix into array. Not equivalent
        to turning into Normal Form Game then flattening.

        Returns:
            _type_:
                Array of all of the entries in each matrix
                of the Polymatrix.
                With all of the payoffs of one player before the
                next. With the opponent going up. 
        """        
        flattened_polym = np.concatenate([
            np.ravel(self.polymatrix[p1, p2])
            for p1 in range(self.players)
            for p2 in range(self.players)
            if p1 != p2
        ])
        return flattened_polym

    def payoff_pure(self, player, pure_actions):
        """
        Payoff to a player at a pure action combination.

        Args:
            player (_type_): _description_
            pure_actions (_type_): _description_

        Returns:
            _type_: Payoff
        """        
        return sum(
            [
                self.polymatrix[(player, p2)
                                ][pure_actions[player]][pure_actions[p2]]
                for p2 in range(self.players)
                if player != p2
            ]
        )

    def to_paired_polymatrix(self):
        """
        Represent the head to head games with bimatrix games with
        a row player and column player.

        Returns:
            _type_: Bimatrix games.
        """
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
        """
        Payoff to each player at a mixed action combination.

        Args:
            actions (_type_): Mixed action combination.

        Returns:
            _type_: Payoff to each player.
        """        
        payoff_vectors = self.payoff_vectors(actions)
        return [
            np.inner(actions[p1], payoff_vectors[p1])
            for p1 in range(self.players)
        ]

    def payoff_vectors(self, actions):
        """
        Payoffs to each action for each player if the other
        players stay at their mixed actions.

        Args:
            actions (_type_):
                Mixed action combination to deviate from.

        Returns:
            _type_: Payoffs to deviations for each player.
        """
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
        """
        Best Responses to a mixed action combination
        as the action numbers of each player and
        the payoff that player gets if they unilaterally
        deviate to that Best Response.

        Args:
            actions (_type_): Mixed action combination.

        Returns:
            _type_: 
                Tuple with first entry best responses
                and second entry their payoffs.
        """
        pvs = self.payoff_vectors(actions)
        return (
            [np.argmax(pv) for pv in pvs],
            [np.max(pv) for pv in pvs]
        )

    def to_nfg(self):
        """Gives the equivalent Normal Form Game.

        Returns:
            _type_: New, equivalent Normal Form Game.
        """
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
        """
        The lowest and highest components of payoff from
        head to head games.

        Returns:
            _type_: Tuple of minimum and maximum.
        """
        min_p = min([np.min(M) for M in self.polymatrix.values()])
        max_p = max([np.max(M) for M in self.polymatrix.values()])
        return (min_p, max_p)

    def to_one_big_matrix(self, costs=True, low_avoider=2.0):
        """
        Puts all of the matrices of the Polymatrix Game into
        One Big Matrix where the submatrices on the diagonal
        are zero and the submatrices at (i, j) are the
        component payoffs to player i in the head to head game
        between players i and j.

        Args:
            costs (bool, optional):
                Should the payoffs be turned into costs.
            low_avoider (float, optional): 
                A number to keep the entries positive.
                Defaults to 2.0.

        Returns:
            _type_: One Big Matrix.
        """
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
