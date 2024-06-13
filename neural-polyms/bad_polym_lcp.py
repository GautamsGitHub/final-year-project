"""
By Gautam based on QuantEcon
"""
import numpy as np
from quantecon.optimize.pivoting import _pivoting, _lex_min_ratio_test
from quantecon.optimize.lcp_lemke import _get_solution
from nf_and_polymatrix import PolymatrixGame

def bad_polym_lcp_solver(polym: PolymatrixGame):

    positive_cost_maker = 150
    M = np.vstack([
        np.hstack([
            np.zeros((polym.actions[player], polym.actions[player])) if p2 == player
            else (positive_cost_maker - polym.polymatrix[(player, p2)])
            for p2 in range(polym.players)
        ])
        for player in range(polym.players)
    ])/positive_cost_maker
    assert(np.all(M >= 0))
    total_actions = sum(polym.actions)
    q = -np.ones(total_actions)

    n = np.shape(M)[0]
    tableau = np.hstack([
        np.eye(n),
        -M,
        np.reshape(q, (-1, 1))
    ])
    basis = np.array(range(n))
    z = np.empty(n)

    # can be chosen
    starting_col = n
    finishing_col = starting_col - n

    # in the style of quantecon's lcp_lemke
    pivcol = starting_col
    pivrow = 0
    pivcol_vals = tableau[:, pivcol]
    max_but_negative = -np.inf
    for i in range(1, n):
        if pivcol_vals[i] > max_but_negative and pivcol_vals[i] < 0:
            pivrow = i
            max_but_negative = pivcol_vals[i]
    
    print("pivoting. col:", pivcol, "row:", pivrow)
    _pivoting(tableau, pivcol, pivrow)
    basis[pivrow], pivcol = pivcol, pivrow + n

    pivrow = 0
    valcol_vals = tableau[:, 2 * n]
    pivcol_vals = tableau[:, pivcol]
    max_but_negative = -np.inf
    for i in range(1, n):
        if pivcol_vals[i] > max_but_negative and valcol_vals[i] < 0:
            pivrow = i
            max_but_negative = pivcol_vals[i]

    print("pivoting. col:", pivcol, "row:", pivrow)
    _pivoting(tableau, pivcol, pivrow)
    basis[pivrow], pivcol = pivcol, pivrow + n

    # Array to store row indices in lex_min_ratio_test
    argmins = np.empty(n, dtype=np.int_)

    while pivcol != finishing_col:
        _, pivrow = _lex_min_ratio_test(
            tableau, pivcol, 0, argmins,
        )

        print("basis", basis)
        print("pivoting. col:", pivcol, "row:", pivrow)

        _pivoting(tableau, pivcol, pivrow)
        basis[pivrow], leaving_var = pivcol, basis[pivrow]

        if leaving_var < n:
            pivcol = leaving_var + n
        else:
            pivcol = leaving_var - n

    _get_solution(tableau, basis, z)

    print("final tableau:\n", tableau)
    print("z:", z)
    print("basis:", basis)
    
    return(0)