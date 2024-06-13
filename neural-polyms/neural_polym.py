import numpy as np
from nf_and_polymatrix import NormalFormGame, PolymatrixGame, Game
from voting_game_generator import generate_games
from math import pow
import torch
from torch import nn
from polym_lcp import polym_lcp_solver

# java -jar gamut.jar -g MajorityVoting -players 5
# -actions 4 -output GTOutput -f majority_voting.gam

filename = "./poly_rps.gam"
# filename = "../gemp_re/games/handmade.gam"
# filename = "../gemp_re/games/polym4.gam"
nf = NormalFormGame.from_gam_file(filename)

polymatrix_game = PolymatrixGame.from_nf(nf)
paired_polym = polymatrix_game.to_paired_polymatrix()
renf = polymatrix_game.to_nfg()
# print(nf.check_if_polymatrix())

number_of_players = 5
number_of_actions = 4


def judge_polym(nfg: NormalFormGame, polym: PolymatrixGame):

    polym_eq = polym_lcp_solver(polym)
    ppeq = nfg.payoffs_of_actions(polym_eq)
    pvs = nfg.payoff_vectors(polym_eq)
    brs, pbrs = nfg.best_responses_and_payoffs(polym_eq)
    print("payoff vectors:", pvs)
    print("best responses:", brs)
    print("payoffs of best responses", pbrs)
    print("payoffs at our attempt", ppeq)

    return 0

class GamePolymNeuralNetwork(nn.Module):

    def __init__(self, players, actions) -> None:
        super().__init__()
        n_inputs = int(pow(actions, players)) * players
        n_outputs = players * (players - 1) * actions * actions
        # this describes all the matrices entirely; a 
        # polymatrix game with the same payoffs
        # can be described with fewer values
        # because only the differences in apyoffs of actions matters
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_inputs, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_outputs)
        )

    def forward(self, nf_game: NormalFormGame):
        x = torch.flatten(nf_game.entries)
        y = self.linear_relu_stack(x)
        return y
    

print("LCP starting")
judge_polym(nf, polymatrix_game)
print("LCP done")
