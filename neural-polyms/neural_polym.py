import numpy as np
from nf_and_polymatrix import NormalFormGame, PolymatrixGame, Game

# java -jar gamut.jar -g MajorityVoting -players 5 -actions 4 -output GTOutput -f majority_voting.gam

filename = "./majority_voting.gam"
# filename = "../gemp_re/games/compound1.gam"
nf = NormalFormGame.from_gam_file(filename)

polymatrix_game = PolymatrixGame.from_nf(nf)
paired_polym = polymatrix_game.to_paired_polymatrix()
renf = polymatrix_game.to_nfg()

print(nf.check_if_polymatrix())
# print(polymatrix_game)

# print(PolymatrixGame.from_nf(polymatrix_game).entries)