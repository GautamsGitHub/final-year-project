import numpy as np
from nf_and_polymatrix import NormalFormGame, PolymatrixGame
import cvxpy as cp
import numpy as np

filename = "./rps.gam"

normal_form_game = NormalFormGame.from_gam_file(filename)
polymatrix_game = PolymatrixGame.from_nf(normal_form_game)

def zero_sum_polym_qp_solver(polym: PolymatrixGame):
    obm = polym.to_one_big_matrix(low_avoider=0)
    M = (obm - obm.T) / 2
    print(M)
    print(obm)
    total_actions = sum(polym.actions)
    q = np.zeros(total_actions)
    n = np.shape(M)[0]

    z = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(z, M) + q.T @ z),
        [z >= 0, M @ z + q >= 0]
    )

    prob.solve()

    return z.value

sol = zero_sum_polym_qp_solver(polymatrix_game)
print(sol)


