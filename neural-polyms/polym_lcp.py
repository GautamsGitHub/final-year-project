"""
By Gautam based on QuantEcon
"""
import numpy as np
from quantecon.optimize.pivoting import _pivoting, _lex_min_ratio_test
from quantecon.optimize.lcp_lemke import _get_solution
from nf_and_polymatrix import PolymatrixGame

def polym_lcp_solver(polym: PolymatrixGame):
    LOW_AVOIDER = 2.0
    positive_cost_maker = polym.range_of_payoffs()[1] + LOW_AVOIDER
    # Construct the LCP like Howson:
    M = np.vstack([
        np.hstack([
            np.zeros((polym.actions[player], polym.actions[player])) if p2 == player
            else (positive_cost_maker - polym.polymatrix[(player, p2)])
            for p2 in range(polym.players)
        ] + [
            -np.outer(np.ones(polym.actions[player]), np.eye(polym.players)[player])
        ])
        for player in range(polym.players)
    ] + [
        np.hstack([
            np.hstack([
                np.outer(np.eye(polym.players)[player], np.ones(polym.actions[player]))
                for player in range(polym.players)
            ]
            ),
            np.zeros((polym.players, polym.players))
        ])
    ]
    )
    total_actions = sum(polym.actions)
    q = np.hstack([np.zeros(total_actions), -np.ones(polym.players)])

    n = np.shape(M)[0]
    tableau = np.hstack([
        np.eye(n),
        -M,
        np.reshape(q, (-1, 1))
    ])

    basis = np.array(range(n))
    z = np.empty(n)

    starting_player_actions = {
        player : 0
        for player in range(polym.players)
    }

    for player in range(polym.players):
        row = sum(polym.actions) + player
        col = n + sum(polym.actions[:player]) + starting_player_actions[player]
        _pivoting(tableau, col, row)
        basis[row] = col
    
    # Array to store row indices in lex_min_ratio_test
    argmins = np.empty(n + polym.players, dtype=np.int_)
    p = 0
    retro = False
    while p < polym.players:
        finishing_v = sum(polym.actions) + n + p
        finishing_x = n + sum(polym.actions[:p]) + starting_player_actions[p]
        finishing_y = finishing_x - n

        pivcol = finishing_v if not retro else finishing_x if finishing_y in basis else finishing_y

        retro = False

        while True:
            _get_solution(tableau, basis, z)

            _, pivrow = _lex_min_ratio_test(
                tableau, pivcol, 0, argmins,
            )

            _pivoting(tableau, pivcol, pivrow)
            basis[pivrow], leaving_var = pivcol, basis[pivrow]
            
            if leaving_var == finishing_x or leaving_var == finishing_y:
                p += 1
                break
            elif leaving_var == finishing_v:
                print("entering the dodgy dead end case")
                p -= 1
                retro = True
                break
            elif leaving_var < n:
                pivcol = leaving_var + n
            else:
                pivcol = leaving_var - n
    
    combined_solution = _get_solution(tableau, basis, z)

    eq_strategies = [
        combined_solution[sum(polym.actions[:player]) : sum(polym.actions[:player+1])]
        for player in range(polym.players)
    ]

    return eq_strategies