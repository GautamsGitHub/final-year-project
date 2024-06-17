module nnGames

include("lcp.jl")

using  .LCP
using  Flux
using  InferOpt
import DifferentiableFrankWolfe
import StatsBase

n_inputs = Int(number_of_actions^number_of_players) * number_of_players
n_outputs = number_of_players * (number_of_players - 1) * number_of_actions * number_of_actions

nfg_to_polym_model = Chain(
    Dense(n_inputs, 16, relu),
    Dense(16, 16, relu),
    Dense(16, n_outputs)
)

probabilistic_co_layer = PerturbedAdditive(solve_lcp)

println()
gradient_optimiser = Adam(0.1)
parameters = Flux.params(nfg_to_polym_model)
# data = [nfg_from_file("../rps.gam")]
all_data = load_data("rps_2_3.pkl")

function pipeline_loss(x)
    polym = nfg_to_polym_model(x)
    y = probabilistic_co_layer(polym)
    return judge_eq(y, x)
end

for epoch in 0:20
    sample_data = StatsBase.sample(all_data, 16)
    # data = all_data[1+(epoch*40):40+(epoch*40)]
    Flux.train!(pipeline_loss, parameters, sample_data, gradient_optimiser)
    println("epoch", epoch)
    println("loss ", pipeline_loss.(all_data[1:5]))
end

println("all done!")

end