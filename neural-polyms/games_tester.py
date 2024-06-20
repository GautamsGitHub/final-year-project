from voting_game_generator import pickle_wrapper
from polym_lcp import polym_lcp_solver
from typing import List
from nf_and_polymatrix import PolymatrixGame, NormalFormGame
import numpy as np

np.set_printoptions(precision=2, suppress=True, linewidth=np.inf)


def polym_non_polym():
    nfg = NormalFormGame.from_gam_file("generated_games/majority_voting_game.gam")
    polym_approx = PolymatrixGame.from_nf(nfg)
    nfg_approx = polym_approx.to_nfg()
    print("values in original normal form game:", nfg.flatten())
    print("values in apprximation of normal form game:", nfg_approx.flatten())
    print("difference:", nfg.flatten() - nfg_approx.flatten())

# games: List[NormalFormGame] = pickle_wrapper("generated_games/voting_3_3.pkl")

# game = games[0]

# for game in games:
#     polym = PolymatrixGame.from_nf(game)
#     # print(polym)
#     solution = polym_lcp_solver(polym)
#     print(solution)

# me_game = NormalFormGame.from_gam_file("generated_games/majority_voting_game.gam")
# print("is it a polym", me_game.check_if_polymatrix())
# polym_approx = PolymatrixGame.from_nf(me_game)
# print(polym_approx)
# polysol = polym_lcp_solver(polym_approx)
# print("polysol", polysol)

# print("BRs and Ps", me_game.best_responses_and_payoffs(polysol))


polym_non_polym()