using LinearAlgebra
using StatsBase
using Plots
using Random
using ProgressBars
using Statistics

# %% Helper functions

function lorenz!(X)
    sigma = 10
    beta = 8/3
    rho = 28

    x, y, z = X

    dx = sigma*(y-x)
    dy = x*(rho-z)-y
    dz = x*y-beta*z

    output = [dx, dy, dz]
    return output
end

function rossler!(X)
    a = 0.2
    b = 0.2
    c = 5.7

    x, y, z = X

    dx = -y-z
    dy = x+a*y
    dz = b + z*(x-c)

    output = [dx, dy, dz]
    return output
end

function HR_neuron!(X)
    s = 4
    x_rest = -1.6
    I = 3.1
    r = 0.002
    a = 1
    b = 3
    c = 1
    d = 5

    x, y, z = X

    dx = y-a*x^3+b*x^2+I-z
    dy = c-d*x^2-y
    dz = r*(s*(x-x_rest)-z)

    output = [dx, dy, dz]
    return output
end

function FHN_neuron!(X)
    a = 0.25 #Scaled threshold voltage
    ϵ = 0.005 #Magnituide scaling of blocking mechanism
    γ = 2.5
    f = 0.129
    α = 0.1 #input Driving amplitude
    v, w, I = X

    b1 = 10
    b2 = 1

    # dv = v*(v-a)*(1-v) - w + (α/(2*pi*f))*cos(I) #Sodium positive depolarisation channel
    # dw = ϵ*(v-γ*w)
    # dI = 2*pi*f


    dv = v*(v-1)*(1-b1*v) - w + (α/(2*pi*f))*I#cos(I) #Sodium positive depolarisation channel
    dw = b2*v
    dI = -2*pi*f*sqrt(1-I^2)

    output = [dv, dw, dI]

    return output
end

function FHN_neuron2!(X; f = 0.129)
    # f = 0.12643
    α = 0.1 #input Driving amplitude
    b1 = 10
    b2 = 1

    v, w, I, dI = X
    ω = 2*pi*f
    dv = v*(v-1)*(1-b1*v) - w + (α/ω)*I #Sodium positive depolarisation channel
    dw = b2*v
    dI = dI
    ddI = -(ω^2)*I

    output = [dv, dw, dI, ddI]

    return output
end

#Druven FHN neuron with chaotic driver
function FHN_neuron3!(X; f = 0.2)
    α = 0.1 #input Driving amplitude
    b1 = 10
    b2 = 1

    v, w, x, y, z = X
    dv = v*(v-1)*(1-b1*v) - w + α*x #Sodium positive depolarisation channel
    dw = b2*v
    dx, dy, dz = f* rossler!([x,y,z])

    output = [dv, dw, dx, dy, dz]

    return output
end

function chua!(X)
    #Parameters
    k = -1
    β = 53.612186
    γ = -0.75087096
    α = 17

    #Cubic nonlinearity
    a = -0.0375582129
    b = -0.8415410391
    f(x) = a*x^3 + b*x

    x,y,z = X
    dx = k*(y-x+z)
    dy = k*α*(x-y-f(y))
    dz = k*(-β*x-γ*z)

    output = [dx,dy,dz]
end

function chuaDouble!(X)
    #Parameters
    k = -1
    β = 53.612186
    γ = -0.75087096
    α = 20

    #Cubic nonlinearity
    a = -0.0375582129
    b = -0.8415410391
    f(x) = a*x^3 + b*x

    x,y,z = X
    dx = k*(y-x+z)
    dy = k*α*(x-y-f(y))
    dz = k*(-β*x-γ*z)

    output = [dx,dy,dz]
end

function chua2!(X)
    #Parameters
    α = 10
    β = 15.68
    a = -1.2768
    b = -0.6888

    #Nonlinearity
    function ϕ(x)
        if x >= 1
            out = b*x + a - b
        elseif abs(x) < 1
            out = a*x
        else
            out = b*x - a + b
        end
        return out
    end

    x,y,z = X
    dx = y-x+z
    dy = α*(x-y-ϕ(y))
    dz = -β*x

    output = [dx,dy,dz]
end

function chua3!(X)
    #Parameters Xiaogen Yin, YJ Cao, Sychronisation of Chua’s oscillator via the state observer techniqu
    α = 8.5
    β = 14.286
    γ = 0.01
    m0  = -1/7
    m1 = 2/7

    #Nonlinearity
    function ϕ(x)
        out = m1*x+0.5*(m0-m1)*(abs(x+1)-abs(x-1))
        return out
    end

    x,y,z = X
    dx = y-x+z
    dy = α*(x-ϕ(y))
    dz = -β*y-γ*z

    output = [dx,dy,dz]
end

function halvorsen!(X)
    #Parameters Xiaogen Yin, YJ Cao, Sychronisation of Chua’s oscillator via the state observer techniqu
    a = 1.27
    b = 4

    x,y,z = X
    dx = -a*x-b*(y+z)-y^2
    dy = -a*y-b*(z+x)-z^2
    dz = -a*z-b*(x+y)-x^2

    output = [dx,dy,dz]
end

function linearise(X; model = nothing)

    if isnothing(model)
        output = lorenz!(X)
    else
        output = model(X)
    end
    return output
end


# %%
function diff_matrix(X; N=10, c_dim = 1)
    D = zeros((N, N))
    Threads.@threads for i in 1:N
        for j in i:N
            # D[i, j] = sign(X[i, c_dim]-X[j, c_dim])*abs(X[i, c_dim]-X[j, c_dim])^0.5  # Difference Coupling in X only
			D[i, j] = X[i, c_dim]-X[j, c_dim] # Difference Coupling in X only
			# deviation = X[i, c_dim]-X[j, c_dim]
			# D[i, j] = deviation + 0.5.*sin.(8*pi.*deviation)
            D[j, i] = -D[i, j]
        end
    end
    return D
end

# %% Random Network Generators
function random_graph_A(N,p; neg_p = 0, sym = true)
    A = zeros((N,N))
    # For symmetric matrix
    if sym
        for i in 1:N
            for j in (i+1):N
                if (rand()<p)
                    A[i,j] = 1
                    A[j,i] = 1
                end
            end
        end
    else

        #For nonsymmetric matrix
        for i in 1:N
            for j in 1:N
                if (rand()<p)
                    if i==j
                        continue
                    elseif (rand()<neg_p)
                        A[i,j] = -1
                    else
                        A[j,i] = 1
                    end
                end
            end
        end
    end

    return A
end
# %%
function random_weight_C(N,μ,σ)
    # C = mag.*(rand(Float64,(N,N)))

    C = (σ.*randn(Float64, (N,N))) .+ (μ.*ones((N,N)))

    Threads.@threads for i in 1:N
        for j in (i+1):N
            C[j,i] = C[i,j]
        end
    end

    return C
end

# %%

function chaotic_network(N; K = 1,dt=0.02, T=5000, c=nothing,
                        weights = false,p=0.1, A=nothing,reg_mode = false,
                        init=nothing, noise = nothing, model = nothing, supersample = 1, RK = false, dims = 3, c_dim = 1,
                        μ = 0.5, σ = 0.1, sym = true, neg_p = 0)
    # Generate Adjacency network
    if isnothing(A)
        if reg_mode #Check if doing regression task
            A = ones((N,N))-I
        else #Mode to generate a random graph i.e. not regression mode
            A = random_graph_A(N,p, neg_p = neg_p,sym = sym)
        end
    end

    if isnothing(c)
        # C = random_weight_C(N,K) # Uniform from 0
        C = random_weight_C(N,μ,σ) # Uniform from 0
    else
        if weights == false
            C = c.*ones((N, N))
        else
            C = c
        end
    end

    # r = np.linalg.eig(C.*A)[0][0]  # Spectral Radius --> Effective Gain
    # d_max = np.max(np.sum(A, axis=1))  # Maximum Degree
    #
    # # Scaling Value to prevent overflow
    # K = np.maximum(r, d_max)
    # if K == 0:  # Zero degree case
    #     K = 1

    # Generate Empty Matrix
    S = zeros((N, T*supersample, dims))
    dS = zeros((N, dims))
    δt = (dt/supersample)

    # Initialise starting values
    if isnothing(init)
        if model == FHN_neuron2!
            S[:, 1, 1] = 0.5 .*rand(Float64,size(S[:, 1, 1]))
            S[:, 1, 3] = 0.2 .*rand(Float64,size(S[:, 1, 3])) .+0.8
        elseif model == chua!
			# Ensure that for heterogeneous networks, initialise system in the same scroll regime
			S[:,1,1] .= 0.5 .+ 0.05*randn(size(S[:,1,1]))
			S[:,1,2] .= 0
			S[:,1,3] .= 0
            # S[:, 1, :] = 1 .*(rand(Float64,size(S[:, 1, :])).-0.5)
        else
            S[:, 1, :] = 1 .*(rand(Float64,size(S[:, 1, :])).-0.5)
            # S[:, 1, :] = 2 .*(rand(Float64,size(S[:, 1, :])).-0.5)
        end

    else
        S[:, 1, :] = init
    end

    for t in 1:T*supersample
        Threads.@threads for i in 1:N
            if !RK
                dS[i, :] = linearise(S[i, t, :], model = model)
            else
                # Runge Kutta Integration
                k1 = linearise(S[i, t, :], model = model)
                k2 = linearise(S[i, t, :].+δt*(k1/2), model = model)
                k3 = linearise(S[i, t, :].+δt*(k2/2), model = model)
                k4 = linearise(S[i, t, :].+δt*k3, model = model)
                dS[i, :] = (k1+2*(k2+k3)+k4)/6
            end
        end

        D = diff_matrix(S[:, t, :], N=N, c_dim = c_dim)
        # c_vec = np.sum((C*A*D)/K, axis=1)
        c_vec = sum((C.*A.*D), dims=2)
        dS[:, c_dim] = dS[:, c_dim]-c_vec  # Apply coupling forces

        if !isnothing(noise)
            dS = dS + noise.*randn(size(dS))
        end # Apply coupling forces

        if t < T*supersample
            S[:, t+1, :] = S[:, t, :]+δt*dS
        end
    end

    return S[:,1:supersample:(T*supersample),:], A, C
end

# %% Data normalisation
function normalise(data)
    S = deepcopy(data)

    # # 0-1 Normalisation
    # scale = maximum(S,dims = (1,2))[:]-minimum(S,dims = (1,2))[:]
    # loc = minimum(S,dims = (1,2))[:]

    # Gaussian Normalisation
    scale = std(S,dims = (1,2))[:]
    loc = mean(S,dims = (1,2))[:]

    for d in 1:size(S)[3]
        S[:,:,d] = (S[:,:,d].-loc[d])./scale[d]
    end
    return S, scale, loc
end

function unscale(data, scale, loc)
    S = deepcopy(data)

    for d in 1:size(S)[3]
        S[:,:,d] = scale[d].*S[:,:,d] .+ loc[d]
    end
    return S
end

function rescale(data, scale, loc)
    S = deepcopy(data)
    for d in 1:size(S)[3]
        S[:,:,d] = (S[:,:,d].-loc[d])./scale[d]
    end
    return S
end
