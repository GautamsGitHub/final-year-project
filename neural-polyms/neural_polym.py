import numpy as np
from nf_and_polymatrix import NormalFormGame, PolymatrixGame, Game
from voting_game_generator import generate_games
from math import pow
import torch
from torch import nn
from quantecon.optimize import lcp_lemke
# from lemkelcp.lemkelcp import lemkelcp as lcp_solver

# java -jar gamut.jar -g MajorityVoting -players 5
# -actions 4 -output GTOutput -f majority_voting.gam

# filename = "./majority_voting.gam"
filename = "../gemp_re/games/handmade.gam"
nf = NormalFormGame.from_gam_file(filename)

polymatrix_game = PolymatrixGame.from_nf(nf)
paired_polym = polymatrix_game.to_paired_polymatrix()
renf = polymatrix_game.to_nfg()
# print(nf.check_if_polymatrix())

number_of_players = 5
number_of_actions = 4

def initiate_nn(players, actions):
    n_inputs = pow(players, actions)
    n_outputs = players * (players - 1) * actions * actions


def solve_polym_via_lcp(polym: PolymatrixGame):
    makes_costs_positive = 20
    M = np.vstack([
        np.hstack([
            np.zeros((polym.actions[player], polym.actions[player])) if p2 == player
            else makes_costs_positive - polym.polymatrix[(player, p2)]
            for p2 in range(polym.players)
        ])
        for player in range(polym.players)
    ])
    q = -np.ones(sum(polym.actions))
    result = lcp_lemke(M, q)
    # result = lcp_solver(M, q)
    print(M, q)
    return result

class GameNeuralNetwork(nn.Module):

    def __init__(self, players, actions) -> None:
        super().__init__()
        n_inputs = pow(players, actions)#
        n_outputs = players * (players - 1) * actions * actions
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
print(solve_polym_via_lcp(polymatrix_game))
print("LCP done")
