from adidas_simplified import simplex_project_grad, gradients_qre_nonsym, gradients_ate_sym
import numpy as np
from scipy.special import softmax

def initials(n: int,m):
	"""
    Generate initial distribution and gradient for a game with 'n' players and 'm' actions each.

    Parameters:
    - n: The number of players in the game.
    - m: The number of actions available to each player.

    Returns:
    - initial_distribution: An array representing the initial distribution.
    - gradient: An array representing the gradient.
	"""
	return np.ones((n, m))/m, np.zeros((n,m))


def adidas(learning_rate=0.01, aux_learning_rate=0.01, initial_temp=0.1, adi_threshold=0.001, total_iters=100):

	num_players = 2
	num_actions = 3

	rps_matrix_A = np.array([[0, -1, 2], [-1, 1, 0], [1, 0, -4]])
	rps_matrix = np.stack((rps_matrix_A, -rps_matrix_A))

	payoff_matrices_dict = {
		(0, 1): rps_matrix,
	}

	first_dist, first_y = initials(num_players, num_actions)
	
	dist = first_dist
	y = first_y
	grad_anneal_steps = 1
	temp = initial_temp

	for t in range(1, total_iters+1):
		(grad_dist, grad_y, grad_anneal_steps), temp, unreg_exp_mean, reg_exp_mean = gradients_qre_nonsym(
			dist, y, grad_anneal_steps, payoff_matrices_dict, 2, temp=temp, exp_thresh=adi_threshold,
			lrs=(learning_rate, aux_learning_rate))
		y = y - max((1/t), aux_learning_rate) * np.array(grad_y)
		dist = dist - np.array(learning_rate) * grad_dist
		print(dist)
		print(temp)
		print(grad_anneal_steps)

# first_dist = [np.ones((2))/2, np.ones((2))/2]
# first_y = [np.zeros((2)), np.zeros((2))]
first_dist, first_y = initials(2,3)

rps_matrix_A = np.array([[0, -1, 2], [-1, 1, 0], [1, 0, -4]])
rps_matrix = np.stack((rps_matrix_A, -rps_matrix_A))

chicken_matrix_A = np.array([[0, 7], [2, 6]])
chicken_matrix = np.stack((chicken_matrix_A, chicken_matrix_A.T))

prisoners_dilemma_A = np.array([[1, -5], [0, -8]])
prisoners_dilemma = np.stack((prisoners_dilemma_A, prisoners_dilemma_A.T))

# maps matchups to the 2 x A x A payoffs of the polymatrix game
payoff_matrices_dict = {
	(0, 1): rps_matrix,
}



# for i in range(10):
# 	(grad_dist, grad_y, grad_anneal_steps), temp, unreg_exp_mean, reg_exp_mean = gradients_qre_nonsym(
# 		dist, y, 0, payoff_matrices_dict, 2)
# 	dist = [
# 		softmax(dist[pi] - grad_dist[pi])
# 		for pi in range(len(dist))
# 	]
# 	y = grad_y
# 	print(y)
# 	print(dist)


adidas(aux_learning_rate=0.01, total_iters=300)

