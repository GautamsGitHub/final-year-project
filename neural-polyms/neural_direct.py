import numpy as np
from nf_and_polymatrix import NormalFormGame, PolymatrixGame, Game
from voting_game_generator import generate_temp_games
from math import pow
import torch
from torch import nn
from polym_lcp import polym_lcp_solver
from itertools import product

# java -jar gamut.jar -g MajorityVoting -players 5
# -actions 4 -output GTOutput -f majority_voting.gam

filename = "./rps.gam"
# filename = "../gemp_re/games/handmade.gam"
# filename = "../gemp_re/games/polym4.gam"
nfg = NormalFormGame.from_gam_file(filename)

polymatrix_game = PolymatrixGame.from_nf(nfg)
paired_polym = polymatrix_game.to_paired_polymatrix()
renf = polymatrix_game.to_nfg()
# print(nf.check_if_polymatrix())

number_of_players = 3
number_of_actions = 2

def entries_tensor(nf: NormalFormGame):
    t = torch.zeros([nf.players] + nf.actions)
    for player in range(nf.players):
        for k, v in nf.entries[player].items():
            t[(player,) + k] = v
    return t

class GameEqNeuralNetwork(nn.Module):

    def __init__(self, players, actions) -> None:
        super().__init__()
        n_inputs = int(pow(actions, players)) * players
        n_outputs = actions * players
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_outputs)
        )

    def forward(self, nf_game: NormalFormGame):
        x = torch.flatten(entries_tensor(nf_game))
        y = self.linear_relu_stack(x)
        return y


print("begin")

model = GameEqNeuralNetwork(number_of_players, number_of_actions)
optimiser = torch.optim.SGD(model.parameters(), lr=0.01)

model.train()

for epoch in range(5):
    batch_size = 10
    games = generate_temp_games(
        number_of_players, number_of_actions, seeds=range(epoch*batch_size, (epoch+1)*batch_size))
    for nfg in games:
        inputs = nfg

        outputs = model(inputs)
        all_actions = np.squeeze(outputs.detach().numpy())
        actions = [all_actions[sum(nfg.actions[:p]):sum(nfg.actions[:p + 1])] for p in range(nfg.players)]

        best_payoffs = torch.tensor(nfg.best_responses_and_payoffs(actions)[1])
        our_payoffs = torch.tensor(nfg.payoffs_of_actions(actions))
        payoff_vectors = nfg.payoff_vectors(actions)
        payoff_grad = -torch.tensor(np.concatenate([
            pv - np.mean(pv)
            for pv in payoff_vectors
            ]))

        loss = nn.functional.l1_loss(our_payoffs, best_payoffs)

        optimiser.zero_grad()
        outputs.backward(gradient=payoff_grad, inputs=outputs)
        optimiser.step()

        print(loss)