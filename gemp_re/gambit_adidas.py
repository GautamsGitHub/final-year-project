import pygambit as gbt

# make the nfg by
# java -jar .\gamut.jar -g Chicken -output GambitOutput -f chicken.nfg

game = gbt.Game.read_game("chicken.nfg")

print("game", game)

result = gbt.nash.simpdiv_solve(game)

print("result", result)

print(type(result[0].mixed_strategies()[0]))