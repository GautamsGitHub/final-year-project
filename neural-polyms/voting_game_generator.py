import subprocess
import os
from nf_and_polymatrix import NormalFormGame, PolymatrixGame
import pickle

def gamut_voting_call(players, actions, seed, file_name = None):
    if not file_name: file_name = f"generated_games/majority_voting_{seed}.gam"
    call_str = (
        "java -jar gamut.jar"
        " -g MajorityVoting"
        f" -players {players}"
        f" -actions {actions}"
        " -output GTOutput"
        f" -random_seed {seed}"
        f" -f {file_name}"
        )
    return call_str

def generate_voting_games(players, actions, seeds=range(10)):
    for seed in seeds:
        subprocess.run(gamut_voting_call(players, actions, seed), shell=True)

def temp_voting_games(players, actions, seeds=range(10)):
    games = []
    for seed in seeds:
        file_name = f"generated_games/majority_voting_{seed}.gam"
        subprocess.run(gamut_voting_call(players, actions, seed, file_name), shell=True)
        nfg = NormalFormGame.from_gam_file(file_name)
        games.append(nfg)
        os.remove(file_name)
    return games


def temp_polym_game(players, actions, seed, p_range = 8):
    file_name = f"generated_games/temp_generated_game_{seed}.gam"
    subprocess.run(gamut_random_polym_call(players, actions, seed, p_range=p_range, file_name=file_name), shell=True)
    nfg = NormalFormGame.from_gam_file(file_name)
    os.remove(file_name)
    return nfg

def gamut_random_polym_call(players, actions, seed, p_range = 8, file_name = None):
   if not file_name: file_name = f"generated_games/random_polym_{seed}.gam"
   call_str = (
      "java -jar gamut.jar"
      " -g PolymatrixGame"
      f" -players {players}"
      f" -actions{(" " + str(actions)) * players}"
      f" -graph CompleteGraph -graph_params [-nodes {players} -reflex_ok 0]"
      f" -subgame RandomGame -subgame_params [-actions {actions} {actions} -players {2}]"
      f" -normalize -min_payoff -{p_range} -max_payoff {p_range}"
      " -output GTOutput"
      f" -random_seed {seed}"
      f" -f {file_name}"
      )
   return(call_str)

def temp_rps_polym_game(players, actions, p_range = 40):
    file_name = "generated_games/temp_generated_polym.gam"
    subprocess.run(gamut_rps_polym_call(players, actions, p_range=p_range, file_name=file_name), shell=True)
    nfg = NormalFormGame.from_gam_file(file_name)
    os.remove(file_name)
    return nfg

def gamut_rps_polym_call(players, actions, p_range = 8, file_name = None , seed=0):
   if not file_name: file_name = "generated_games/rps_polym.gam"
   call_str = (
      "java -jar gamut.jar"
      " -g PolymatrixGame"
      f" -players {players}"
      f" -actions{(" " + str(actions)) * players}"
      f" -graph RandomGraph"
      f" -graph_params [-nodes {players} -edges {1} -sym_edges 0]"
      f" -subgame RockPaperScissors -subgame_params []"
      f" -normalize -min_payoff -{p_range} -max_payoff {p_range}"
      " -output GTOutput"
      f" -random_seed {seed}"
      f" -f {file_name}"
      )
   return(call_str)

def gamut_compound_call(players, file_name = None, seed = 0, p_range = 20):
    if not file_name: file_name = f"generated_games/compopund_polym_{seed}.gam"
    call_str = (
        "java -jar gamut.jar"
        " -g RandomCompoundGame"
        f" -players {players}"
        f" -normalize -min_payoff -{p_range} -max_payoff {p_range}"
        " -output GTOutput"
        f" -random_seed {seed}"
        f" -f {file_name}"
        )
    return(call_str)


def temp_compound_game(players, seed = 0, p_range = 20):
    file_name = "generated_games/temp_generated_compound.gam"
    subprocess.run(gamut_compound_call(
        players, p_range=p_range, file_name=file_name, seed=seed
        ), shell=True)
    nfg = NormalFormGame.from_gam_file(file_name)
    os.remove(file_name)
    return nfg


def temp_compound_games(players, seeds=range(10)):
    games = []
    for seed in seeds:
        game = temp_compound_game(players, seed=seed, p_range=8)
        games.append(game)
    return games

def temp_rps_games(players, actions, seeds=range(10)):
    games = []
    base_rps = temp_rps_polym_game(players, actions, p_range=40)
    for seed in seeds:
        perturbation = temp_polym_game(players, actions, seed, p_range=8)
        perturbed_rps = base_rps + perturbation
        perturbed_rps = base_rps
        games.append(perturbed_rps)
    return games

def compound_pickle(file_name, players, number_of_games):
    games = temp_compound_games(players, range(number_of_games))
    with open(file_name, 'wb') as file:
        pickle.dump(games, file)

def rps_pickle(file_name, players, number_of_games):
    games = temp_rps_games(players, 3, range(number_of_games))
    with open(file_name, 'wb') as file:
        pickle.dump(games, file)

    
def voting_pickle(file_name, players, actions, number_of_games):
    games = temp_voting_games(players, actions, range(number_of_games))
    with open(file_name, 'wb') as file:
        pickle.dump(games, file)

def pickle_wrapper(file_name):
    with open(file_name, 'rb') as file:
        games = pickle.load(file)
    return games
