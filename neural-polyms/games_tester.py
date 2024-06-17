from voting_game_generator import pickle_wrapper
from polym_lcp import polym_lcp_solver
from typing import List
from nf_and_polymatrix import PolymatrixGame, NormalFormGame
import numpy as np

np.set_printoptions(precision=2, suppress=True, linewidth=np.inf)


# games: List[NormalFormGame] = pickle_wrapper("generated_games/rps_2_3_precise.pkl")

# game = games[0]

game2 = NormalFormGame.from_gam_file("generated_games/temp_generated_polym.gam")
polym2 = PolymatrixGame.from_nf(game2)
print("what is this")

print(polym2)

# game3 = NormalFormGame.from_gam_file("../gemp_re/games/polym3.gam")
# polym3 = PolymatrixGame.from_nf(game3)
# print(polym3)

# assert(game2.check_if_polymatrix())
# polym = PolymatrixGame.from_nf(game)
# print(polym)
solution = polym_lcp_solver(polym2)
print(solution)