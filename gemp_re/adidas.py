from adidas_simplified import gradients_qre_nonsym
import numpy as np
from nf_and_polymatrix import NormalFormGame, PolymatrixGame, Game
from abc import ABC, abstractmethod
from scipy.special import softmax
from scipy.stats import entropy as shannon_entropy
from itertools import product


class HeadToHeadCalculator(ABC):

    def __init__(self, game: Game, hh_estimates) -> None:
        self.game = game
        self.hh_estimates = hh_estimates
        return None

    @abstractmethod
    def update_hh(self, dists):
        return None


class SamplingHHC(HeadToHeadCalculator):

    def __init__(self, game: Game, hh_samples_per_player=1) -> None:
        hh_zeros = {
            (player, p2): np.zeros((game.actions[player], game.actions[p2]))
            for player in range(game.players)
            for p2 in range(game.players) if p2 != player
        }
        super().__init__(game, hh_zeros)
        self.hh_samples_per_player = hh_samples_per_player
        return None

    def update_hh(self, dists):
        sampled_actions = [
            np.random.choice(self.game.actions[player], p=dists[player])
            for player in range(self.game.players)
        ]
        for player in range(self.game.players):
            for p2 in np.random.choice(
                [op for op in range(self.game.players) if op != player],
                size=self.hh_samples_per_player
            ):
                self.hh_estimates[(player, p2)] = self.game.two_player_deviation_payoffs(
                    player, p2, sampled_actions)
                # print("hh estimate", player, p2, self.hh_estimates[(player, p2)])

        return None


class FullHHC(HeadToHeadCalculator):

    def __init__(self, game: Game) -> None:
        hh_zeros = {
            (player, p2): np.zeros((game.actions[player], game.actions[p2]))
            for player in range(game.players)
            for p2 in range(game.players) if p2 != player
        }
        super().__init__(game, hh_zeros)
        return None

    def update_hh(self, dists):
        fresh_hh = {
            (player, p2): sum([
                np.prod([
                        [1.0 if p3 == player or p3 == p2 else dists[p3][action_combination[p3]]
                         for p3 in range(self.game.players)]
                        ]) * self.game.two_player_deviation_payoffs(player, p2, action_combination)
                for action_combination in product(*(
                    [[0] if p3 == player or p3 == p2 else range(self.game.actions[p3])
                     for p3 in range(self.game.players)]
                ))
            ])

            for player in range(self.game.players)
            for p2 in range(self.game.players) if player != p2
        }
        self.hh_estimates = fresh_hh
        return None


class DeviationPayoffCalculator(ABC):

    def __init__(self, game: Game, deviation_payoffs) -> None:
        self.game = game
        self.deviation_payoffs = deviation_payoffs
        return None

    @abstractmethod
    def update_deviation_payoffs(self, hh_calculator, dists, timestep):
        return None


class ExponentiallyWeightedDPC(DeviationPayoffCalculator):

    def __init__(self, game: Game, di_learning_rate) -> None:
        deviation_payoffs = [np.zeros(game.actions[p])
                             for p in range(game.players)]
        super().__init__(game, deviation_payoffs)
        self.di_learning_rate = di_learning_rate
        return None

    def update_deviation_payoffs(self, hh_calculator: HeadToHeadCalculator, dists, timestep):
        fresh_deviation_payoffs = [
            np.mean(
                [
                    hh_calculator.hh_estimates[(player, p2)] @ dists[p2]
                    for p2 in range(self.game.players) if p2 != player
                ],
                axis=0
            )
            for player in range(self.game.players)
        ]
        print("fresh deviation payoffs:", fresh_deviation_payoffs)
        # According to Gemp:
        # self.deviation_payoffs = [
        #     self.deviation_payoffs[player] - max(1/timestep, self.di_learning_rate) * (
        #         fresh_deviation_payoffs[player] - self.deviation_payoffs[player])
        #     for player in range(self.game.players)
        # ]
        # But I think:
        self.deviation_payoffs = [
            self.deviation_payoffs[player] + max(1/timestep, self.di_learning_rate) * (
                fresh_deviation_payoffs[player] - self.deviation_payoffs[player])
            for player in range(self.game.players)
        ]
        return None


def initials(game: Game):
    dists = [np.ones(game.actions[p]) / game.actions[p]
             for p in range(game.players)]
    y = [np.zeros(game.actions[p]) for p in range(game.players)]
    return dists, y


def keep_dist_in_simplex(
    distribution,
    min_prob=1e-3,
    max_prob=1 - 1e-4
):
    outs_removed = np.clip(distribution, min_prob, max_prob)
    re_summing_to_one = outs_removed / np.sum(outs_removed)
    return re_summing_to_one


def grad_adi_estimate(game: Game, x, y, hh_calculator: HeadToHeadCalculator, temperature):
    best_responses = [softmax(y[player] / temperature)
                      for player in range(game.players)]
    policy_gradient = [
        - (y[player] - temperature * (np.log(np.clip(x[player], 1e-5, 1)) + 1))
        for player in range(game.players)
    ]
    br_jacobians = {
        (player, p2): (np.diag(best_responses[p2]) - np.outer(best_responses[p2], best_responses[p2])
                       ) @ hh_calculator.hh_estimates[(p2, player)] / temperature
        for player in range(game.players)
        for p2 in range(game.players) if p2 != player
    }
    grad_l = [
        policy_gradient[player] + sum([
            np.inner(br_jacobians[(player, p2)], (y[p2] - temperature * (np.log(np.clip(best_responses[p2], 1e-5, 1)) + 1))) +
            hh_calculator.hh_estimates[(
                p2, player)].T @ (best_responses[p2] - x[player])
            for p2 in range(game.players) if p2 != player
        ])
        for player in range(game.players)
    ]
    return grad_l


def adi_estimate(game: Game, x, y, temperature):
    best_responses = [softmax(y[player] / temperature)
                      for player in range(game.players)]
    print("softmaxed BRs", best_responses)
    br_entropies = [temperature *
                    shannon_entropy(best_responses[player]) for player in range(game.players)]
    x_entropies = [temperature *
                   shannon_entropy(x[player]) for player in range(game.players)]
    l = sum([
        np.inner(y[player], (best_responses[player] - x[player])) +
        br_entropies[player] - x_entropies[player]
        for player in range(game.players)
    ])
    return l


def adidas(
    game: Game,
    deviation_payoff_calculator: DeviationPayoffCalculator,
    hh_calculator: HeadToHeadCalculator,
    learning_rate=0.01,
    initial_temperature=1.0,
    adi_threshold=0.001,
    max_iters=100
):

    # dists = [np.ones(game.actions[p]) / game.actions[p]
    #          for p in range(game.players)]
    dists = [
        np.eye(game.actions[p])[0]
        for p in range(game.players)
    ]
    temperature = initial_temperature

    for step in range(1, max_iters+1):
        hh_calculator.update_hh(dists)
        deviation_payoff_calculator.update_deviation_payoffs(
            hh_calculator, dists, step)
        y = deviation_payoff_calculator.deviation_payoffs
        grad_adi = grad_adi_estimate(
            game, dists, y, hh_calculator, temperature)
        possibly_out_of_range_dists = [
            # dists[player] - learning_rate * (grad_adi[player] - np.mean(grad_adi[player])) / np.mean(grad_adi[player])
            dists[player] - learning_rate * \
            (grad_adi[player] - np.mean(grad_adi[player]))
            for player in range(game.players)
        ]
        # the following works in some cases
        # dists = dists - learning_rate * (grad_adi - np.mean(grad_adi, axis=1))

        dists = [
            keep_dist_in_simplex(d)
            for d in possibly_out_of_range_dists
        ]

        current_adi_estimate = adi_estimate(game, dists, y, temperature)

        if current_adi_estimate < adi_threshold:
            input()
            temperature = temperature / 2

        print("step number:", step)
        print("adi gradient:", grad_adi)
        print("deviation payoff estimates:", y)
        print("temperature of", temperature)
        print("dists:", dists)
        print("ADI estimate:", current_adi_estimate)
        # print("polym approx", hh_calculator.hh_estimates)
        print("")

        if temperature < 0.001:
            break


# java -jar .\gamut.jar -g RandomCompoundGame -players 3 -output GTOutput -f compound.gam

# filename = "games/handmade.gam"
filename = "games/compound1.gam"
nf = NormalFormGame.from_gam_file(filename)
polymatrix_game = PolymatrixGame.from_nf(nf)
paired_polym = polymatrix_game.to_paired_polymatrix()

print("starting adidas")

# For the 1 vs 1 game:
# hh_calc = SamplingHHC(nf, 1)
# dpc = ExponentiallyWeightedDPC(nf, di_learning_rate=0.8)
# adidas(
#     nf,
#     dpc,
#     hh_calc,
#     learning_rate=1e-3,
#     adi_threshold=1e-3,
#     initial_temperature=8,
#     max_iters=400
# )

# For the compound game:
# hh_calc = SamplingHHC(nf, 1)
hh_calc = FullHHC(nf)
dpc = ExponentiallyWeightedDPC(nf, di_learning_rate=1.0)
adidas(
    nf,
    dpc,
    hh_calc,
    learning_rate=1e-4,
    adi_threshold=0.5,
    initial_temperature=0.01,
    max_iters=1000
)

# adidas(polymatrix_game, aux_learning_rate=0.2, adi_threshold=0.1,
#        initial_temperature=100.0, learning_rate=0.0001, max_iters=8000)
