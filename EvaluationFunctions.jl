include("BackpropRegressionGeneral2.jl")
include("ChaoticNetwork2.jl")
include("MLP2.jl")
include("MeanFieldVF2.jl")
include("UncoupleSystems2.jl")

using Distributed
using SharedArrays

# %% Run all required backprop steps for one epoch
function backprop(S, C_guess, T_IN, dt, id_start, model; RK = true, omit = 0, exact = false)
	S_guess = guessGeneral(S, C_guess, T_IN, dt, id_start=id_start, model=testmode!(model), RK = RK) # Simulate network with C_guess and same initial conditions
	err, gradient = grad(S_guess, S, C_guess, dt,  idx=id_start, omit = omit,
								exact = exact, model =　testmode!(model)) # Calculate gradient and errors
	return err, gradient
end

function eval_grad(S, C_guess, MINI_BATCH_SIZE , T_IN, dt, wash, model; RK = true, omit = 0, exact = false)

	rand_idx = SharedArray{Int}(MINI_BATCH_SIZE)
	rand_idx .= rand(wash:(size(S)[2]-T_IN), MINI_BATCH_SIZE) # Draw random starting points
	# S_shared = SharedArray{Float64}(size(S), init = deepcopy(S))
	# C_guess_shared = SharedArray{Float64}(size(C_guess), init = deepcopy(C_guess))

	# batch_errors = zeros(MINI_BATCH_SIZE)
	# batch_grad = zeros((MINI_BATCH_SIZE,N,N)) # variable to store gradient of weights w.r.t loss

	batch_errors = SharedArray{Float64}(MINI_BATCH_SIZE)
	batch_grad = SharedArray{Float64}((MINI_BATCH_SIZE,N,N))

	@sync @distributed for k in 1:MINI_BATCH_SIZE
		batch_errors[k], batch_grad[k,:,:] = backprop(S, C_guess, T_IN, dt, rand_idx[k], model, RK = true, omit = omit, exact = false)
	end

	mini_grad = mean(batch_grad, dims = 1)[1,:,:] # Divide by total contributions from each run in batch

    return mean(batch_errors), mini_grad
end

# %%

# """
# Obsolete function based on in-place numerical differentiation
# """
# function refitMLP(S, C_guess, wash, dt, REFIT_NEIGHBOURS, model, BATCH_SIZE, EPOCHS, η_REFIT)

# 	# Smooth out VF to get rid of noise
# 	S_0, V = uncouple_VF(S[:,wash+1:end,:],C_guess,dt)
# 	points, ave_V = smooth_VF(S_0,V, n_neighbours = REFIT_NEIGHBOURS)

# 	# Make Model and data set
# 	input_train = points
# 	output_train = ave_V

# 	trainedModel = deepcopy(model)

# 	trainedModel = trainModelGPU(input_train, output_train, trainedModel, BATCH_SIZE, EPOCHS, η_REFIT)

# 	return deepcopy(trainedModel)
# end

function refitMLP(S, V, C_guess, REFIT_NEIGHBOURS, model, BATCH_SIZE, EPOCHS, η_REFIT)

	# Smooth out VF to get rid of noise
	S_0, V_0 = pointify_uncouple_VF(S,V,C_guess)
	points, ave_V = smooth_VF(S_0,V_0, n_neighbours = REFIT_NEIGHBOURS)

	# Make Model and data set
	input_train = points
	output_train = ave_V

	trainedModel = deepcopy(model)

	trainedModel = trainModelGPU(input_train, output_train, trainedModel, BATCH_SIZE, EPOCHS, η_REFIT)

	return deepcopy(trainedModel)
end
