from adidas_simplified import gradients_qre_nonsym
import numpy as np
from nf_and_polymatrix import NormalFormGame, PolymatrixGame

def initials(pg: PolymatrixGame):
	dist = [np.ones(pg.actions[p]) / pg.actions[p] for p in range(pg.players)]
	y = [np.zeros(pg.actions[p]) for p in range(pg.players)]
	return dist, y

def keep_dist_in_simplex(
		distribution,
		min_prob = 1e-5,
		max_prob = 1 - 1e-6
		):
	outs_removed = np.clip(distribution, min_prob, max_prob)
	re_summing_to_one = outs_removed / np.sum(outs_removed)
	return re_summing_to_one

def adidas(
		polymatrix_game: PolymatrixGame, 
		learning_rate=0.01, 
		aux_learning_rate=0.01, 
		initial_temp=1.0, 
		adi_threshold=0.001, 
		max_iters=100
		):

	first_dist, first_y = initials(polymatrix_game)

	dist = first_dist
	y = first_y
	grad_anneal_steps = 1
	temp = initial_temp

	for t in range(1, max_iters+1):
		(grad_dist, grad_y, grad_anneal_steps), temp, unreg_exp_mean, reg_exp_mean = gradients_qre_nonsym(
			dist,
			y, 
			grad_anneal_steps,
			polymatrix_game, 
			polymatrix_game.players, 
			temp=temp, 
			exp_thresh=adi_threshold,
			lrs=(learning_rate, aux_learning_rate)
			)
		y = [y[p] - max(1/t, aux_learning_rate) * grad_y[p] for p in range(polymatrix_game.players)]
		# dist = [dist[p] - learning_rate * grad_dist[p] for p in range(polymatrix_game.players)]
		dist = [keep_dist_in_simplex(dist[p] - learning_rate * grad_dist[p]) for p in range(polymatrix_game.players)]
		# print("grad dist of", grad_dist)
		# print("makes a payoff of", polymatrix_game.payoffs_of_actions(dist))
		# print("compared to a possible best of", polymatrix_game.best_responses_and_payoffs(dist)[1])
		print("BR:", polymatrix_game.best_responses_and_payoffs(dist)[0])
		print("dist:", dist)
		# print(y)
		print("temperature of", temp)
		print(grad_anneal_steps)
		print("ADI of orginial:", unreg_exp_mean)
		print("ADI regularized:", reg_exp_mean)
		if temp < 0.01: break


# java -jar .\gamut.jar -g RandomCompoundGame -players 3 -output GTOutput -f compound.gam

filename = "games/polym4.gam"
nf = NormalFormGame.from_gam_file(filename)
polymatrix_game = PolymatrixGame.from_nf(nf)
paired_polym = polymatrix_game.to_paired_polymatrix()

print("starting adidas")
adidas(polymatrix_game, aux_learning_rate=0.2, adi_threshold=0.1, initial_temp=100.0, learning_rate=0.0001, max_iters=8000)
