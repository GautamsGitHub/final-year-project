from adidas_simplified import gradients_qre_nonsym
import numpy as np
from nf_and_polymatrix import NormalFormGame, PolymatrixGame

def initials(pg: PolymatrixGame):
	dist = [np.ones(pg.actions[p]) / pg.actions[p] for p in range(pg.players)]
	y = [np.zeros(pg.actions[p]) for p in range(pg.players)]
	return dist, y


def adidas(
		polymatrix_game: PolymatrixGame, 
		learning_rate=0.01, 
		aux_learning_rate=0.01, 
		initial_temp=1.0, 
		adi_threshold=0.001, 
		total_iters=100
		):

	first_dist, first_y = initials(polymatrix_game)

	dist = first_dist
	y = first_y
	grad_anneal_steps = 1
	temp = initial_temp

	for t in range(1, total_iters+1):
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
		y = y - max((1/t), aux_learning_rate) * np.array(grad_y)
		# dist = [softmax(p_dist - learning_rate * p_grad_dist) for p_dist, p_grad_dist in zip(dist, grad_dist)]
		dist = dist - np.array(learning_rate) * grad_dist
		print("grad dist of", grad_dist)
		print("makes a payoff of", polymatrix_game.payoffs_of_actions(dist))
		print("compared to a possible best of", polymatrix_game.best_responses_and_payoffs(dist)[1])
		print("by playing", polymatrix_game.best_responses_and_payoffs(dist)[0])
		print("dist of", dist)
		# print(y)
		print("temperature of", temp)
		print(grad_anneal_steps)
		print(unreg_exp_mean, reg_exp_mean)


# java -jar .\gamut.jar -g RandomCompoundGame -players 3 -output GTOutput -f compound.gam

filename = "compound.gam"
nf = NormalFormGame.from_gam_file(filename)
polymatrix_game = PolymatrixGame(nf)
paired_polym = polymatrix_game.to_paired_polymatrix()

print("starting adidas")
adidas(polymatrix_game, aux_learning_rate=0.01, initial_temp=1.0, learning_rate=0.0001, total_iters=800)
print(polymatrix_game.polymatrix[(0,1)])
