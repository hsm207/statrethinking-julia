module Utils
using HTTP, CSV
using JuMP, Ipopt, ForwardDiff
using Distributions
using LinearAlgebra
using Gen, RCall, Statistics, DataFrames

import StatsBase

export VariableSpecification, get_data, quap, get_posterior_samples, summarize_posterior_samples

build_url(filename) = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/$(filename).csv"
retrieve_file(url) = HTTP.get(url).body |> IOBuffer |> CSV.read
get_data(filename) = build_url(filename) |> retrieve_file

struct VariableSpecification
    lower_bound::Float64
    upper_bound::Float64
    prior::Distributions.Distribution
end

calculate_covariance_matrix(f, optimal_points) = begin
    H(x::Vector) = ForwardDiff.hessian(f, x)

    inv(-1 * H(optimal_points)) .|>
    m->round(m, digits = 4)
end

function quap(objective_fn, vars_specs)
    model = Model(Ipopt.Optimizer)
    nvars = length(vars_specs)
    register(model, :f, nvars, objective_fn, autodiff = true)

    model_vars = map(vars_specs) do var_spec
        @variable(model, lower_bound = var_spec.lower_bound, upper_bound = var_spec.upper_bound, start = rand(var_spec.prior))
    end

    @NLobjective(model, Max, f(model_vars...))

    optimize!(model)

    optimal_points = model_vars .|> value |> collect 
    covar_mat = calculate_covariance_matrix(objective_fn, optimal_points)

    optimal_points, covar_mat
end

StatsBase.cov2cor(C::AbstractMatrix) = StatsBase.cov2cor(C, diag(C) .|> sqrt)

function do_inference(model, X, Y, amount_of_computation, params)
    
    observations = Gen.choicemap()
    for (i, y) in enumerate(Y)
        observations[(:y, i)] = y
    end
    
    trace, = generate(model, (X,), observations)
   
    for i = 1:amount_of_computation
        trace, = mh(trace, Gen.select(params...))
    end
  
    return trace
end

function get_posterior_samples(model, sample_size, computation_budget, X, Y, params)
    results = Array{Array{Float64}}(undef, sample_size)

    Threads.@threads for i in 1:sample_size
        trace = do_inference(model, X, Y, computation_budget, params)
        results[i] = [trace[param] for param in params]
    end
    
    return hcat(results...)'
end

function summarize_posterior_samples(samples, params)
    a = map(eachcol(samples)) do col
        [mean(col), std(col)]
    end |>
    m->hcat(m...)'

    b = R"""
    require(rethinking)
    apply($samples, 2, PI)
    """ |>
    rcopy |>
    m->m'

    m = hcat(params, a, b)
    
    return DataFrame(m, [:param, :mean, :std, Symbol("5%"), Symbol("95%")])
end

end
