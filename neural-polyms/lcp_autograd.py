import autograd.numpy as np
from polym_lcp import polym_lcp_solver
from autograd import grad
from nf_and_polymatrix import NormalFormGame, PolymatrixGame

def matrices_solve(matrices):
    pm_entries = {
        (0, 1) : matrices[0],
        (1, 0) : matrices[1]
    }
    pmg = PolymatrixGame(2, [3, 3], pm_entries)

    sol = polym_lcp_solver(pmg)
    print(sol)
    return(sol)

pg = grad(matrices_solve)

filename = "./rps.gam"
# filename = "../gemp_re/games/handmade.gam"
# filename = "../gemp_re/games/polym4.gam"
nf = NormalFormGame.from_gam_file(filename)

polymatrix_game = PolymatrixGame.from_nf(nf)

sol = matrices_solve(list(polymatrix_game.polymatrix.values()))

d = pg(list(polymatrix_game.polymatrix.values()))
print(d)
