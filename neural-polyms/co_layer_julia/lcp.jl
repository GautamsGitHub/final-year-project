module LCP

using PyCall
using Zygote

pushfirst!(pyimport("sys")."path", "")
@pyimport wrappers

export nfg_from_file, solve_lcp, judge_eq, number_of_players, number_of_actions, say_hello, apythonfun, load_data

number_of_players = 5
number_of_actions = 3

say_hello() = println("Hello!")

apythonfun(x) = wrappers.afunction(x)

nfg_from_file(filename) = wrappers.flat_from_file(filename)

solve_lcp(x) = wrappers.flat_lcp_solve(
    x, 
    number_of_players,
    number_of_actions
    )

judge_eq(eq, fg) = wrappers.flat_judge_eq(
    eq, fg, number_of_players, number_of_actions
)

load_data(filename) = [Float32.(x) for x in (wrappers.pickle_wrapper_flat(filename))]

function judge_eq_grad(Δ, eq, fg)
    grad = wrappers.flat_judge_eq_grad(
        eq, fg, number_of_players, number_of_actions
    )
    return (Δ .* grad, nothing)
end

Zygote.@adjoint judge_eq(eq, fg) = judge_eq(eq, fg), Δ -> judge_eq_grad(Δ, eq, fg)

end