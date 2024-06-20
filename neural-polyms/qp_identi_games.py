import numpy as np
from nf_and_polymatrix import NormalFormGame, PolymatrixGame
import cvxpy as cp
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=np.inf)

filename = "./poly_rps.gam"

# We utilise the fact that GAMUT polymatrix Rock, Paper, Scissors
# games in fact have identical payoff head to head games.

# Howeer sadly the games are not PSD.

normal_form_game = NormalFormGame.from_gam_file(filename)
polymatrix_game = PolymatrixGame.from_nf(normal_form_game)

def identi_polym_qp_solver(polym: PolymatrixGame):
    """
    Finds a Nash Equilibrium for a Polymatrix Game where all
    the head to head games are identical payoff. Meaning the
    payoff to both players is the same.

    Args:
        polym (PolymatrixGame): _description_

    Returns:
        _type_: _description_
    """    
    obm = polym.to_one_big_matrix(low_avoider=200.0)
    M = np.round(obm, 5)

    total_actions = sum(polym.actions)
    q = np.zeros(total_actions)
    n = np.shape(M)[0]

    z = cp.Variable(n)

    for i in range(n):
        subm = M[:i][:, :i]
        print(i, np.linalg.det(subm))

    qf = cp.quad_form(z, M, assume_PSD=False)
    if not qf.is_psd():
        print("Matrix is not Positive semi Definite")

    prob = cp.Problem(
        cp.Minimize(qf + q.T @ z),
        [z >= 0, M @ z + q >= 0]
    )

    # this problem solver has a differentiable version
    # giving rise to the possibility of using it as a layer
    # in a neural netwrok easily.

    prob.solve()

    return z.value

sol = identi_polym_qp_solver(polymatrix_game)
print(sol)


