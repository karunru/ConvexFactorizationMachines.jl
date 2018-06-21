using IterativeSolvers

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
mutable struct CFM
  num_iter::Int64
  reg_W::Float64
  w::Array{Float64, 1}
  U::Array{Float64, 2}

  CFM(num_iter, reg_W) = new(num_iter, reg_W, Float64[], Array{Float64}(0,0))
end

function train(model::CFM, X, Y)
  X = convert(SparseMatrixCSC{Float64,Int64}, X)
  Y = Array(Y)

  n, d = size(X)

  #Add bias for training data
  Z = hcat(ones(n), X)

  T = model.num_iter
  η = model.reg_W

  model.U = zeros(d, T)
  P = zeros(d, T)
  λ = ones(T)
  w = zeros(d+1, 1)
  global fval = zeros(T)

  for t in 1:T
    tmp = X * model.U[:, 1:t]
    fQ = 0.5 * (vec(sum(tmp .* tmp, 2)) - (X .* X) * vec(sum(model.U[:, 1:t] .* model.U[:, 1:t], 2)))

    ZY = Z' * (Y - fQ)

    #Conjugate Gradient: Solve Zw = Y
    wout = cg!(w, Z'*Z, ZY, tol=1e-6, verbose = false, maxiter=100)
    model.w = vec(wout)

    tr_err = Y - Z*w -fQ

    #Frank-Wolfe update: eigs(X diag(tr_err) X^t, 1)
    pout = eigs(X'*spdiagm(sparsevec(tr_err))*X, nev=1, which=:LR, maxiter = 300, tol = 1e-1)[2]

    p = vec(real(pout))

    #Optimal step size
    err = η*((X*p).^2 - (X .* X) * (p .* p)) - fQ
    α = ((tr_err' * err) / (err' * err))

    #Update
    P[:, t] = √η * p
    λ[1:t-1] = (1 - α) * λ[1:t-1]
    λ[t] = max(1e-10, α)

    model.U[:, 1:t] = P[:, 1:t] .* sqrt.(λ[1:t])'

    #Traning RMSE
    global fval[t] = sqrt(mean(tr_err.^2))

    #fval: training RMSE
    println("iteration:", t, " Training RMSE:", fval[t])
  end
end

function predicter(model::CFM, X)
  X = convert(SparseMatrixCSC{Float64,Int64}, X)

  #Compute fQ (2-way factor)
  tmp = X * model.U
  fQ = 0.5 * (sum(tmp .* tmp, 2) - (X .* X) * sum(model.U .* model.U, 2))

  Ŷ = model.w[1] + X * model.w[2:end] + fQ

  return Ŷ
end
