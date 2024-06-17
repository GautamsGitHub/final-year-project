"""
By Gautam based on QuantEcon
"""
import jax.numpy as np
from quantecon_jax import pivoting, lex_min_ratio_test, get_solution
from nf_and_polymatrix_jax import PolymatrixGame

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

    starting_player_actions = {
        player : 0
        for player in range(polym.players)
    }

    for player in range(polym.players):
        row = sum(polym.actions) + player
        col = n + sum(polym.actions[:player]) + starting_player_actions[player]
        tableau = pivoting(tableau, col, row)
        basis = basis.at[row].set(col)
    
    # Array to store row indices in lex_min_ratio_test
    # argmins = np.empty(n + polym.players, dtype=np.int_)
    p = 0
    retro = False
    while p < polym.players:
        finishing_v = sum(polym.actions) + n + p
        finishing_x = n + sum(polym.actions[:p]) + starting_player_actions[p]
        finishing_y = finishing_x - n

        pivcol = finishing_v if not retro else finishing_x if finishing_y in basis else finishing_y

        retro = False

        while True:

            _, pivrow = lex_min_ratio_test(
                tableau, pivcol, 0,
            )

            tableau = pivoting(tableau, pivcol, pivrow)
            leaving_var = basis[pivrow]
            basis = basis.at[pivrow].set(pivcol)
            
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
    
    combined_solution = get_solution(tableau, basis)

    eq_strategies = [
        combined_solution[sum(polym.actions[:player]) : sum(polym.actions[:player+1])]
        for player in range(polym.players)
    ]

    return eq_strategies