from adidas_simplified import gradients_qre_nonsym
import numpy as np
from nf_and_polymatrix import NormalFormGame, PolymatrixGame, Game
from abc import ABC, abstractmethod
from scipy.special import softmax


class DeviationPayoffCalculator(ABC):

    def __init__(self, game: Game, deviation_payoffs) -> None:
        self.game = game
        self.deviation_payoffs = deviation_payoffs
        return ()

    @abstractmethod
    def update_deviation_payoffs(self, dists, timestep):
        return ()


class ExponentiallyWeightedDPC(DeviationPayoffCalculator):

    def __init__(self, game: Game, deviation_payoffs, di_learning_rate) -> None:
        super().__init__(game, deviation_payoffs)
        self.hh_samples_per_player = 1
        self.di_learning_rate = di_learning_rate
        return ()

    def update_deviation_payoffs(self, dists, timestep):
        sampled_actions = [
            np.random.choice(self.game.actions, p=dists[player])
            for player in range(self.game.players)
        ]
        fresh_deviation_payoffs = [
            np.mean(
                [
                    self.game.two_player_deviation_payoffs(
                        player, p2, sampled_actions) @ dists[p2]
                    for p2 in np.random.choice(
                        self.game.players[:player] +
                        self.game.players[player + 1:],
                        size=self.hh_samples_per_player
                    )
                ],
                axis=0
            )
            for player in self.game.players
        ]
        self.deviation_payoffs = [
            self.deviation_payoffs[player] - max(1/timestep, self.di_learning_rate) * (
                fresh_deviation_payoffs[player] - self.deviation_payoffs[player])
            for player in self.game.players
        ]
        return ()


def initials(game: Game):
    dists = [np.ones(game.actions[p]) / game.actions[p]
             for p in range(game.players)]
    y = [np.zeros(game.actions[p]) for p in range(game.players)]
    return dists, y


def keep_dist_in_simplex(
    distribution,
    min_prob=1e-5,
    max_prob=1 - 1e-6
):
    outs_removed = np.clip(distribution, min_prob, max_prob)
    re_summing_to_one = outs_removed / np.sum(outs_removed)
    return re_summing_to_one


def grad_adi(x, y, temperature):
    best_responses = [softmax(yi / temperature) for yi in y]
    return ()


def adidas(
    game: Game,
    deviation_payoff_calculator: DeviationPayoffCalculator,
    learning_rate=0.01,
    aux_learning_rate=0.01,
    initial_temp=1.0,
    adi_threshold=0.001,
    max_iters=100
):

    first_dists, first_y = initials(game)

    dists = first_dists
    y = first_y
    grad_anneal_steps = 1
    temp = initial_temp

    for t in range(1, max_iters+1):
        deviation_payoff_calculator.update_deviation_payoffs(dists, t)
        y = deviation_payoff_calculator.deviation_payoffs

        (grad_dist, grad_y, grad_anneal_steps), temp, unreg_exp_mean, reg_exp_mean = gradients_qre_nonsym(
            dists,
            y,
            grad_anneal_steps,
            polymatrix_game,
            polymatrix_game.players,
            temp=temp,
            exp_thresh=adi_threshold,
            lrs=(learning_rate, aux_learning_rate)
        )
        y = [y[p] - max(1/t, aux_learning_rate) * grad_y[p]
             for p in range(polymatrix_game.players)]
        # dist = [dist[p] - learning_rate * grad_dist[p] for p in range(polymatrix_game.players)]
        dists = [keep_dist_in_simplex(
            dists[p] - learning_rate * grad_dist[p]) for p in range(polymatrix_game.players)]
        # print("grad dist of", grad_dist)
        # print("makes a payoff of", polymatrix_game.payoffs_of_actions(dist))
        # print("compared to a possible best of", polymatrix_game.best_responses_and_payoffs(dist)[1])
        # print("BR:", polymatrix_game.best_responses_and_payoffs(dist)[0])
        # print("dist:", dist)
        # print(y)
        # print("temperature of", temp)
        # print(grad_anneal_steps)
        # print("ADI of orginial:", unreg_exp_mean)
        # print("ADI regularized:", reg_exp_mean)
        if temp < 0.01:
            break


# java -jar .\gamut.jar -g RandomCompoundGame -players 3 -output GTOutput -f compound.gam

filename = "games/polym_big1.gam"
nf = NormalFormGame.from_gam_file(filename)
polymatrix_game = PolymatrixGame.from_nf(nf)
paired_polym = polymatrix_game.to_paired_polymatrix()

print("starting adidas")
adidas(polymatrix_game, aux_learning_rate=0.2, adi_threshold=0.1,
       initial_temp=100.0, learning_rate=0.0001, max_iters=8000)
