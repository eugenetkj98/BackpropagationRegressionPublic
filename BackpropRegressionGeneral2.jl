include("ChaoticNetwork2.jl")
include("MLP2.jl")

using ProgressBars
using Plots
using LinearAlgebra

# %% Guess Network
# Generalised Guess function with no preimposed adjacency matrix
function guessGeneral(S, C_guess, t_in, dt; id_start=1, supersample = 1, model = nothing, RK = false)
    # Guessed Value
    N = size(S)[1]
    dims = size(S)[3]
    S_guess, C_guess = chaotic_network(N, dt = dt, T=t_in, c=C_guess, p=1, reg_mode = true, init=S[:, id_start, :],
                                        weights = false, dims = dims,
                                        supersample = supersample, model = model, RK = RK)
    return S_guess
end

# %% Coupling Function
function g(a, b)
    return -(a-b)
end

# %% Calculate coupling differences
function coupling_matrix(node_vals)
    x = node_vals
    N = length(x)
    D = zeros((N, N))
    # Threads.@threads
    for i in 1:N
        for j in 1:N
            D[i, j] = g(deepcopy(x[i]), deepcopy(x[j]))
        end
    end
    return D
end

# %% Forward Function
function forward(x, C, dt;
                exact = true, model = nothing)

    N = size(x)[1]
    t_in = size(x)[2]
    dims = size(x)[3]

    # Forward Pass Storage
    # (timestep, target_node, (coupling gradients), dims)
    dXdK = zeros((t_in, N, N, N, dims))
    # # (timestep, target_node, (coupling gradients))
    # dYdK = zeros((t_in, N, N, N))
    # # (timestep, target_node, (coupling gradients))
    # dZdK = zeros((t_in, N, N, N))
    # (timestep, target_node, (coupling gradients))
    dFdK = zeros((t_in, N, N, N))
    # (timestep,(pairs adjacency), (coupling gradients))
    dGdK = zeros((t_in, N, N, N, N))
    G = zeros((t_in, N, N))  # (timestep, (pair coupling))

    for t in 1:t_in
        #Initialise
        if t == 1
            for d in 1:dims
                dXdK[t, :, :, :, d] = zeros((N, N, N))
            end
            dFdK[t, :, :, :] = zeros((N, N, N))

        else
            # t step

            dXdK[t, :, :, :, :] = dXdK_calc(
                dXdK, dFdK, dGdK, G, C, x, t, dt, model)

            dFdK[t, :, :, :] = dFdK_calc(
                dXdK, x, t, dt, model = model)

        end

        # Difference X-Coupling Matrix
        G[t, :, :] = coupling_matrix(x[:, t, 1])
        dGdK[t, :, :, :, :] = dGdK_calc(dXdK, t)
    end

    return dXdK#,dYdK,dZdK
end

function dXdK_calc(dXdK, dFdK, dGdK, G, C, pos, t, dt, model = model)
    N = size(dFdK)[2]
    dims = size(dXdK)[5]
    dXdK_t = zeros((N, N, N, dims))

    # X component calculation
    Threads.@threads for i in 1:N
        for j in 1:N
            for k in 1:N
                sum = 0
                for h in 1:N
                    if h != i
                        sum = sum + deepcopy(C[i, h]).*deepcopy(dGdK[t-1, i, h, j, k])
                    end
                    if (i == j) & (h == k)
                        sum = sum + deepcopy(G[t-1, i, h])
                    end
                end
                dXdK_t[i, j, k, 1] = deepcopy(dFdK[t-1, i, j, k])+dt*sum
            end
        end
    end

    Threads.@threads for i in 1:N
        # Extract current state of node i
        state = deepcopy(pos[i, t-1, :])

        #Calculate Partial Derivatives
        ∂ = partial_diff(model, state)

        for dim in 2:dims
            for j in 1:N
                for k in 1:N
                    sum = 0
                    for d in 1:dims
                        sum = sum + deepcopy(∂[dim,d])*deepcopy(dXdK[t-1, i, j, k, d])
                    end
                    dXdK_t[i, j, k, dim] = deepcopy(dXdK[t-1, i, j, k, dim])+dt*sum
                end
            end

        end
    end
    return dXdK_t
end

function dFdK_calc(dXdK, pos, t, dt; model = nothing)
    N = size(dXdK)[2]
    dims = size(dXdK)[5]
    dFdK_t = zeros((N, N, N))

    Threads.@threads for i in 1:N
        # Extract current state of node i

        state = deepcopy(pos[i, t, :])

        #Calculate Partial Derivatives
        ∂ = partial_diff(model, state)

        #Calculate dFdK
        for j in 1:N
            for k in 1:N
                sum = 0
                for d in 1:dims
                    sum = sum + deepcopy(∂[1,d])*deepcopy(dXdK[t, i, j, k, d])
                end
                dFdK_t[i, j, k] = deepcopy(dXdK[t, i, j, k, 1])+dt*sum
            end
        end
    end
    return dFdK_t
end

function dGdK_calc(dXdK, t)
    N = size(dXdK)[2]
    dGdK_t = zeros((N, N, N, N))
    Threads.@threads for i in 1:N
        for h in 1:N
            dg_IHdK = zeros((N, N))
            for j in 1:N
                for k in 1:N
                    # Assuming Difference Coupling in x (1) dimension only
                    dg_IHdK[j, k] = (-1)*deepcopy(dXdK[t, i, j, k, 1])+(1)*deepcopy(dXdK[t, h, j, k, 1])
                end
            end
            dGdK_t[i, h, :, :] = deepcopy(dg_IHdK)
        end
    end
    return dGdK_t
end

# %%

function grad(S_guess, S, C_guess, dt; idx=1, omit = 0,
                exact = true, model = nothing)
    N = size(S_guess)[1]
    t_in = size(S_guess)[2]
    dims = size(S_guess)[3]

    #Calculate prediction gradients
    dXdK = forward(S_guess, C_guess,dt, model = model)

    #Calculate Error gradients
    dEdX = 2. *(S_guess[1:N-omit,:,:]-S[1:N-omit, idx:idx+t_in-1, :])

    # Report error over all terms
    E = sum((S_guess[1:N-omit,:,:]-S[1:N-omit, idx:idx+t_in-1, :]).^2)


    #Sum up all gradients
    dLossdK = zeros((N, N))
    for t in 1:t_in
        for i in 1:N-omit
            for d in 1:dims
                dLossdK = dLossdK + dEdX[i, t, d].*dXdK[t, i, :, :, d]
            end
        end
    end

    return E, dLossdK  # Error matrix, Gradient
end
