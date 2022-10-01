include("ChaoticNetwork2.jl")
include("MeanFieldVF2.jl")
include("MLP2.jl")

using ProgressBars
using LinearAlgebra
# %% Uncouples the system manually and returns a set of mappings from state to VF
function pointify_uncouple_VF(S,V,C_est; node_id = nothing)#, couple_A = [1,0,0])

    dims = size(S)[3]

    # Placeholder variable to store decoupled velocities
    V_decoupled_temp = zeros(size(V))

    # Coupling toggle vector, turns on or off assumed coupling
    #Assign coupling dimensions
    couple_A = zeros(dims)
    couple_A[1] = 1

    #Placeholder variable for decoupling
    c_d = zeros(dims)

    for J in ProgressBar(1:size(S)[1], leave = false)
        for t in 3:size(S)[2]

            for d in 1:dims
                c_d[d] = dot(C_est[J,:],S[:,t,d].-S[J,t,d])
            end
            V_decoupled_temp[J,t,:] = V[J,t,:] - couple_A.*c_d
        end
    end

    # Flatten data
    if isnothing(node_id)
        state = S[1,:,:]
        local_f = V_decoupled_temp[1,:,:]

        for J in 2:size(S)[1]
            state = vcat(state,S[J,:,:])
            local_f = vcat(local_f,V_decoupled_temp[J,:,:])
        end
    else
        state = S[node_id[1],:,:]
        local_f = V_decoupled_temp[node_id[1],:,:]

        for J in 2:length(node_id)
            state = vcat(state,S[node_id[J],:,:])
            local_f = vcat(local_f,V_decoupled_temp[node_id[J],:,:])
        end
    end

    return state, local_f
end

"""
Alternate function to uncouple velocities and collate states if velocities are already
    precomputed
"""
function uncouple_VF(S,C_est,dt; node_id = nothing)#, couple_A = [1,0,0])

    dims = size(S)[3]

    # Remove one/two time step to adjust length of array for decoupling
    S_0 = zeros(size(S).-(0,2,0)) #states zeros(size(S).-(0,1,0)) #states
    f = zeros(size(S_0)) #local derivative for evolution (f dot)

    # Coupling toggle vector, turns on or off assumed coupling
    #Assign coupling dimensions
    couple_A = zeros(dims)
    couple_A[1] = 1

    #Placeholder variable for decoupling
    f_d = zeros(dims)
    c_d = zeros(dims)

    for J in ProgressBar(1:size(S_0)[1], leave = false)
        for t in 3:size(S_0)[2]#2:size(S_0)[2]
            #Numerically compute the estimated derivative Central Difference Approach
            # f_1 = (S[J,t+1,1]-S[J,t-1,1])/2#(S[J,t+1,1]-S[J,t,1])
            # f_2 = (S[J,t+1,2]-S[J,t-1,2])/2#(S[J,t+1,2]-S[J,t,2])
            # f_3 = (S[J,t+1,3]-S[J,t-1,3])/2#(S[J,t+1,3]-S[J,t,3])


            #Numerically compute the estimated derivative 5 Point Stencil Approach

            f_d = (-S[J,t+2,:].+8*S[J,t+1,:].-8*S[J,t-1,:].+S[J,t-2,:])./12
            # f_1 = (-S[J,t+2,1]+8*S[J,t+1,1]-8*S[J,t-1,1]+S[J,t-2,1])/12
            # f_2 = (-S[J,t+2,2]+8*S[J,t+1,2]-8*S[J,t-1,2]+S[J,t-2,2])/12
            # f_3 = (-S[J,t+2,3]+8*S[J,t+1,3]-8*S[J,t-1,3]+S[J,t-2,3])/12

            for d in 1:dims
                c_d[d] = dot(C_est[J,:],S[:,t,d].-S[J,t,d])
            end
            # c_1 = dot(C_est[J,:],S[:,t,1].-S[J,t,1])
            # c_2 = dot(C_est[J,:],S[:,t,2].-S[J,t,2])
            # c_3 = dot(C_est[J,:],S[:,t,3].-S[J,t,3])

            S_0[J,t,:] = S[J,t,:]
            f[J,t,:] = (f_d-dt*couple_A.*c_d)/dt
            # f[J,t,:] = ([f_1,f_2,f_3]-dt*couple_A.*[c_1,c_2,c_3])/dt
        end
    end

    # Flatten data
    if isnothing(node_id)
        state = S_0[1,:,:]
        local_f = f[1,:,:]

        for J in 2:size(S_0)[1]
            state = vcat(state,S_0[J,:,:])
            local_f = vcat(local_f,f[J,:,:])
        end
    else
        state = S_0[node_id[1],:,:]
        local_f = f[node_id[1],:,:]

        for J in 2:length(node_id)
            state = vcat(state,S_0[node_id[J],:,:])
            local_f = vcat(local_f,f[node_id[J],:,:])
        end
    end

    return state, local_f
end


#
# # %%
# N = 16
# T = 30000
# wash = 2000
# p = 0.6
# dt = 0.02
# S,A,C = chaotic_network(N;dt = dt, T=T+wash, c=nothing, p=p, A=nothing, init=nothing, noise = 10^(-3))
#
# # %% Corrupt weights with noise
# ξ = 0.1
# C_est = C.*A + ξ*rand(Float64,size(C)).*(ones(size(C))-I)
#
# # %% Smooth out VF to get rid of noise
# S_0, V = uncouple_VF(S,C_est,dt)
# points, ave_V = smooth_VF(S_0,V, n_neighbours = 10)
#
# # %%
#
# # Make Model and data set
# input_train = points
# output_train = ave_V
# model = define_model(3,3,128)
#
# # Run training algorithm
# MLP(input_train, output_train, model, batchsize = 1024, max_epochs = 20)
#
# # %%
# sigma = 10
# beta = 8/3
# rho = 28
# lorenz((x,y,z)) = [sigma*(y-x), x*(rho-z)-y,x*y-beta*z]
#
# # %%
# S_test,A_test,C_test = chaotic_network(1;dt = dt, T=T+wash, c=nothing, p=p, A=nothing, init=nothing, noise = 10^(-3))
#
# # %%
# idx = rand(wash:size(S_test)[2])
#
# point = S_test[1,idx,:]
# display((lorenz(point)-model(point))/norm(lorenz(point)))
#
#
# # %%
# idx = rand(wash:size(S_test)[2])
# point = S_test[1,idx,:]
# real_partial((x,y,z)) = [[-sigma, rho-z,y] [sigma, -1, x] [0,-x,-beta]]
# display((real_partial(point)-partial_diff(model,point))./norm(real_partial(point)))
