using IterativeSolvers
using SparseArrays
using Arpack
using Statistics

""" Convex Factorization Machine with Hazan's algorithm (squared loss)
    Yamada, M. et. al., http://arxiv.org/abs/1507.01073


Model:

f(x;w,W) = w^t z + 0.5 trace(W (xx^t - diag(x.*x)))

Parameters
----------

num_iter: int
   The number of iteration for Hazan's algorithm: (default: 100)

reg_W: double
   The regularization parameter for interaction term (2-way factors) (default 100)

w: double
   d + 1 dimensional vector the global bias (w[0]) and user-item bias (w[1:])

U: double
   d by T dimensional matrix such that W = U U^t

"""
function train(iter, reg, X, Y)
  X = convert(SparseMatrixCSC{Float64,Int64}, X)
  Y = Array(Y)

  n, d = size(X)

  #Add bias for training data
  Z = hcat(ones(n), X)

  T = iter
  η = reg

  U = zeros(d, T)
  P = zeros(d, T)
  λ = ones(T)
  w = zeros(d+1, 1)
  global fval = zeros(T)

  for t in 1:T
    tmp = X * U[:, 1:t]
    fQ = 0.5 * (vec(sum(tmp .* tmp, dims=2)) - (X .* X) * vec(sum(U[:, 1:t] .* U[:, 1:t], dims=2)))
    
    ZY = Z' * (Y - fQ)

    #Conjugate Gradient: Solve Zw = Y
    wout = cg!(w, Z'*Z, ZY, tol=1e-6, verbose = false, maxiter=100)
    w = vec(wout)

    tr_err = Y - (Z*w + fQ)

    #Frank-Wolfe update: eigs(X diag(tr_err) X^t, 1)
    pout = eigs(X'*spdiagm(0 => sparsevec(tr_err))*X, nev=1, which=:LR, maxiter = 300, tol = 1e-1)[2]
        
    p = vec(real(pout))

    #Optimal step size
    err = η*((X*p).^2 - (X .* X) * (p .* p)) - fQ
    α = ((tr_err' * err) / (err' * err))

    #Update
    P[:, t] = √η * p
    λ[1:t-1] = (1 - α) * λ[1:t-1]
    λ[t] = max(1e-10, α)

    U[:, 1:t] = P[:, 1:t] .* sqrt.(λ[1:t])'

    #Traning RMSE
    global fval[t] = sqrt(mean(tr_err.^2))

    #fval: training RMSE
    println("iteration:", t, " Training RMSE:", fval[t])

    tmp = nothing
    fQ = nothing
    ZY = nothing
    wout = nothing
    tr_err = nothing
    pout = nothing
    p = nothing
    err = nothing
    α= nothing
    end
    
    return w, U
end

function predicter(w, U, X)
    X = convert(SparseMatrixCSC{Float64,Int64}, X)

    #Compute fQ (2-way factor)
    tmp = X * U
    fQ = 0.5 * (vec(sum(tmp .* tmp, dims=2)) - (X .* X) * vec(sum(U .* U, dims=2)))

    Ŷ = w[1] .+ X * w[2:end] + fQ

  return Ŷ
end
