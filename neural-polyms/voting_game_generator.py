import subprocess

def gamut_call_generator(number):
   return (
      "java -jar gamut.jar"
      " -g MajorityVoting"
      " -players 5"
      " -actions 4"
      " -output GTOutput"
      f" -random_seed {number}"
      f" -f generated_voting_games/majority_voting_{number}.gam"
      )

seeds = range(10)

for seed in seeds:
    subprocess.call(gamut_call_generator(seed), shell=True)