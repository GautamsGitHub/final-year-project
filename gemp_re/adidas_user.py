from adidas import *

# java -jar .\gamut.jar -g RandomCompoundGame -players 3 -output GTOutput -f compound.gam

# filename = "games/handmade.gam"
# filename = "games/compound1.gam"
# filename = "games/majority_voting_game.gam"
filename = "games/voting_8_3.gam"
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
#     adi_threshold=1e-2,
#     initial_temperature=8,
#     max_iters=400
# )

# For the compound game:
# hh_calc = SamplingHHC(nf, 3)
# hh_calc = FullHHC(nf)
# dpc = ExponentiallyWeightedDPC(nf, di_learning_rate=0.4)
# adidas(
#     nf,
#     dpc,
#     hh_calc,
#     learning_rate=1e-4,
#     adi_threshold=0.5,
#     initial_temperature=0.1,
#     max_iters=1000
# )

# For a large game:
# hh_calc = FullHHC(nf)
hh_calc = SamplingHHC(nf, 15)
dpc = ExponentiallyWeightedDPC(nf, di_learning_rate=0.4)
adidas(
    nf,
    dpc,
    hh_calc,
    learning_rate=1e-2,
    adi_threshold=5.0,
    initial_temperature=8.0,
    max_iters=1000
)
