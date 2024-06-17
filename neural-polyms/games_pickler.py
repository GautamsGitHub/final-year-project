from voting_game_generator import rps_pickle, compound_pickle, voting_pickle

file_name = "generated_games/voting_5_3.pkl"
voting_pickle(file_name, 5, 3, 5000)
print("generated games!")
