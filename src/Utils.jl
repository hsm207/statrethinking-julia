module Utils
using HTTP, CSV
using JuMP, Ipopt, ForwardDiff
using Distributions
using LinearAlgebra

import StatsBase

export VariableSpecification, get_data, quap

build_url(filename) = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/$(filename).csv"
retrieve_file(url) = HTTP.get(url).body |> IOBuffer |> CSV.read
get_data(filename) = build_url(filename) |> retrieve_file

struct VariableSpecification
    lower_bound::Float64
    upper_bound::Float64
    prior::Distribution
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
end