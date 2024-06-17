import jax.numpy as np
from jax import lax
# np.set_printoptions(precision=2, suppress=True, linewidth=np.inf)

TOL_PIV = 1e-10
TOL_RATIO_DIFF = 1e-15

def _min_ratio_test_no_tie_breaking(tableau, pivot, test_col,
                                    argmins, num_candidates,
                                    tol_piv, tol_ratio_diff):
    ratio_min = np.inf
    num_argmins = 0

    for k in range(num_candidates):
        i = argmins[k]
        if tableau[i, pivot] <= tol_piv:  # Treated as nonpositive
            continue
        ratio = tableau[i, test_col] / tableau[i, pivot]
        if ratio > ratio_min + tol_ratio_diff:  # Ratio large for i
            continue
        elif ratio < ratio_min - tol_ratio_diff:  # Ratio smaller for i
            ratio_min = ratio
            num_argmins = 1
        else:  # Ratio equal
            num_argmins += 1
        argmins = argmins.at[num_argmins-1].set(i)

    return num_argmins, argmins

def lex_min_ratio_test(tableau, pivot, slack_start,
                        tol_piv=TOL_PIV, tol_ratio_diff=TOL_RATIO_DIFF):
    nrows = tableau.shape[0]
    num_candidates = nrows

    found = False

    argmins = np.arange(nrows)

    num_argmins, argmins = _min_ratio_test_no_tie_breaking(
        tableau, pivot, -1, argmins, num_candidates, tol_piv, tol_ratio_diff
    )
    if num_argmins == 1:
        found = True
    elif num_argmins >= 2:
        for j in range(slack_start, slack_start+nrows):
            if j == pivot:
                continue
            num_argmins, argmins = _min_ratio_test_no_tie_breaking(
                tableau, pivot, j, argmins, num_argmins,
                tol_piv, tol_ratio_diff
            )
            if num_argmins == 1:
                found = True
                break
    return found, argmins[0]


def pivoting(tableau, pivot_col, pivot_row):
    nrows = np.shape(tableau)[0]
    pivot_elt = tableau[pivot_row, pivot_col]
    pivot_row_vals = tableau[pivot_row] / pivot_elt
    def row_operation(idx):
        return lax.select(
            idx == pivot_row,
            pivot_row_vals,
            tableau[idx, :] - pivot_row_vals * tableau[idx, pivot_col]
        )

    new_tableau = np.vstack([row_operation(i) for i in range(nrows)])
    return new_tableau

def get_solution(tableau, basis):
    n = np.shape(tableau)[0]

    z = np.zeros(n)
    for i in range(n):
        if n <= basis[i] < 2*n:
            z = z.at[basis[i]-n].set(tableau[i, -1])

    return z