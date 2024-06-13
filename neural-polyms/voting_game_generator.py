import subprocess
import os
from nf_and_polymatrix import NormalFormGame

def gamut_call_generator(players, actions, seed, file_name = None):
   if not file_name: file_name = "generated_voting_games/majority_voting_{seed}.gam"
   return (
      "java -jar gamut.jar"
      " -g MajorityVoting"
      f" -players {players}"
      f" -actions {actions}"
      " -output GTOutput"
      f" -random_seed {seed}"
      f" -f {file_name}"
      )

def generate_games(players, actions, seeds=range(10)):
    for seed in seeds:
        subprocess.run(gamut_call_generator(players, actions, seed), shell=True)

def generate_temp_games(players, actions, seeds=range(10)):
    games = []
    for seed in seeds:
        file_name = f"generated_voting_games/majority_voting_{seed}.gam"
        subprocess.run(gamut_call_generator(players, actions, seed, file_name), shell=True)
        nfg = NormalFormGame.from_gam_file(file_name)
        games.append(nfg)
        os.remove(file_name)
    return games
