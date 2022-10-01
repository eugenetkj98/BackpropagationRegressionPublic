
# %%

function SingleRun(;DICT_SMOOTH=DICT_SMOOTH, DICT_MLP=DICT_MLP, DICT_DATAGEN=DICT_DATAGEN, DICT_BACKPROP=DICT_BACKPROP)

	# %% Smoothing Settings
	INIT_NEIGHBOURS = DICT_SMOOTH["INIT_NEIGHBOURS"] # Number of neighbours taken to smooth data for VFL
	REFIT_NEIGHBOURS = DICT_SMOOTH["REFIT_NEIGHBOURS"]

	# %% MLP Settings
	N_HIDDEN = DICT_MLP["N_HIDDEN"]
	BATCH_SIZE = DICT_MLP["BATCH_SIZE"]
	EPOCHS = DICT_MLP["EPOCHS"]
	ENSEMBLE_SIZE = DICT_MLP["ENSEMBLE_SIZE"]
	η = DICT_MLP["η"]
	η_REFIT = DICT_MLP["η_REFIT"]

	# %% Data Hyperparameters
	N = DICT_DATAGEN["N"]
	T = DICT_DATAGEN["T"]
	wash = DICT_DATAGEN["wash"]
	p = DICT_DATAGEN["p"]
	dt = DICT_DATAGEN["dt"]
	K = DICT_DATAGEN["K"]
	μ = DICT_DATAGEN["μ"]
	σ = DICT_DATAGEN["σ"]
	dims = DICT_DATAGEN["dims"]
	real_model = DICT_DATAGEN["real_model"]
	noise = DICT_DATAGEN["noise"]
	window = DICT_DATAGEN["window"]

	# %% Backprop Hyperparameters
	NODE_LEARNING_RATE = DICT_BACKPROP["NODE_LEARNING_RATE"]
	START_LEARNING_RATE = DICT_BACKPROP["START_LEARNING_RATE"]
	LR_ADJ_RATIO=DICT_BACKPROP["LR_ADJ_RATIO"]
	MAX_EPOCHS= DICT_BACKPROP["MAX_EPOCHS"] # First time to re-fit
	REFIT_EPOCHS = DICT_BACKPROP["REFIT_EPOCHS"]
	N_REFITS = DICT_BACKPROP["N_REFITS"]
	MINI_BATCH_SIZE = DICT_BACKPROP["MINI_BATCH_SIZE"]
	TERMINATION_ERROR = DICT_BACKPROP["TERMINATION_ERROR"]
	TERMINATION_WINDOW = DICT_BACKPROP["TERMINATION_WINDOW"]
	PATIENCE = DICT_BACKPROP["PATIENCE"]
	T_IN = DICT_BACKPROP["T_IN"]
	DECAY_PROB = DICT_BACKPROP["DECAY_PROB"]
	RESET_PROB = DICT_BACKPROP["RESET_PROB"]
	RESET_RATE = DICT_BACKPROP["RESET_RATE"]
	M = DICT_BACKPROP["M"] # Momentum
	DECAY_EFF = DICT_BACKPROP["DECAY_EFF"] # Effective Decay rate
	DECAY_WINDOW = DICT_BACKPROP["DECAY_WINDOW"] # Number of decays before reset
	DECAY = DICT_BACKPROP["DECAY"]
	OMISSIONS = DICT_BACKPROP["OMISSIONS"] # Number ofHidden nodes

	# %%
	S_original,A,C = chaotic_network(N;dt = dt, T=T+wash,
	                        K=K, p=p, A=nothing, reg_mode = false, dims = dims,
	                        init=nothing, model = real_model, RK = true, μ = μ, σ = σ,
	                        sym = true, neg_p = 0)

	S_original, scale, loc = normalise(S_original[:,wash+1:end,:])
	
	# Apply Noise
	S_original = S_original .+ noise*randn(size(S_original))
	
	if noise > 0
		# Apply smoothing to remove noise
		println("Applying Smoothing to Data...")
		flush(stdout)
		S = zeros(size(S_original))
		for node in 1:size(S_original)[1]
			for dim in 1:size(S_original)[3]
				S[node,:,dim] = spline_smoothing(S_original[node,:,dim], λ = 10.0)
				# S[node,:,dim] = MA_smoothing(S_original[node,:,dim], window = window)
			end
		end
		# # If using MA_smoothing trim data lengths
		# S_original = S_original[:,window+1:(end-window),:]
		# S = S[:,window+1:(end-window),:]
	else
		S = deepcopy(S_original)
	end

	####### Results Storage
	E = zeros(MAX_EPOCHS+N_REFITS*REFIT_EPOCHS) # Observation Error
	LR = zeros(MAX_EPOCHS+N_REFITS*REFIT_EPOCHS) # Learning Rate
	C_E = zeros(MAX_EPOCHS+N_REFITS*REFIT_EPOCHS) # Coupling Weight Error of coupling compared to ground truth
	MODELS = Array{Any}(undef, N_REFITS+2) #Store Models
	WEIGHTS = Array{Float64}(undef, (N_REFITS+2, N, N)) #Store Regressed Weights

	####### Guess a random initial weight
	MAG = 0.0001
	C_guess = (MAG .*rand(Float64,size(A))).*(ones(N,N)-I)
	
	####### Numerical Calculation of Vector Field
	
	# Numerical Differentiation of Time Series
	S_0 = S[:,1:(end-4),:]
	S_1 = S[:,2:(end-3),:]
	S_2 = S[:,3:(end-2),:]
	S_3 = S[:,4:(end-1),:]
	S_4 = S[:,5:end,:]
	
	
	# Correct datalengths to adjust for numerical differentiation
	S_original = S_original[:,3:(end-2),:]
	S = S[:,3:(end-2),:]
	V = (S_0-8*S_1+8*S_3-S_4)/(12*dt)
	
	# if noise > 0
	# 	# Apply smoothing to remove noise
	# 	print("Applying Smoothing to Vector Field Data...")
	# 	flush(stdout)
	# 	V_temp = zeros(size(V))
	# 	for node in 1:size(V)[1]
	# 		for dim in 1:size(V)[3]
	# 			V_temp[node,:,dim] = MA_smoothing(V[node,:,dim], window = window)
	# 		end
	# 	end
	# 	# If using MA_smoothing
	# 	V = V_temp[:,window+1:(end-window),:]
	# 	S = S[:,window+1:(end-window),:]
	# end
	
	# Numerically calculate vector field with mean field approach
	S_0, V_0, indexes = pointify_VF(S, V)
	points, ave_V = smooth_VF(S_0,V_0, n_neighbours = INIT_NEIGHBOURS)
	
	####### Initial MLP Model Construction

	dims = size(S)[3]

	# Make Model and data set
	input_train = points
	output_train = ave_V

	model = define_model(dims,dims,N_HIDDEN)
	model = trainModelGPU(input_train, output_train, deepcopy(model), BATCH_SIZE, EPOCHS, η)

	####### Save Initial Model and Weights
	MODELS[1] = deepcopy(model)
	WEIGHTS[1,:,:] = deepcopy(C_guess)

	####### Begin First Regression

	LEARNING_RATE = deepcopy(START_LEARNING_RATE)
	PREV_GRAD = 0

	j=0 # initialise adaptive learning rate|
	k=0 # Reset counter for adaptive learning rate
	# COUNTER=0 # Reset counter for large reset

	for i in 1:MAX_EPOCHS
		@everywhere GC.gc()
		LR[i] = deepcopy(LEARNING_RATE)

		batch_error, mini_grad = eval_grad(S, C_guess, MINI_BATCH_SIZE, T_IN, dt, wash, model; RK = true, omit = OMISSIONS, exact = false)

		if i == 1
			CURRENT_GRAD = -(mini_grad./(norm(mini_grad))).*LEARNING_RATE
			grad_step = CURRENT_GRAD
		else
			CURRENT_GRAD = -(mini_grad./(norm(mini_grad))).*LEARNING_RATE
			grad_step = M*PREV_GRAD + (1-M)*CURRENT_GRAD
		end

		PREV_GRAD = deepcopy(grad_step)

		# Update weight matrix
		C_guess = C_guess .+ deepcopy(grad_step)

		# Calculate normalised coupling error and save
		C_error = norm(C.*A.-C_guess)/norm(C.*A)
		C_E[i] = deepcopy(C_error)
		E[i] = deepcopy(batch_error)

		str = string("Epochs: [$i/$MAX_EPOCHS]", ", ")
		str = string(str, "Prediction Error: $(round(E[i], digits = 8))", ", ")
		str = string(str, "C_Error Magnitude: $(round(C_E[i], digits = 8))", ", ")
		str = string(str, "Learn Rate: $(round(LEARNING_RATE, digits = 8))", ", ")
		str = string(str, "Time to Reset: $(DECAY_WINDOW-k)", "\n")

		print(str)
		flush(stdout)

		# Early termination if fully learned
		if i>TERMINATION_WINDOW
			if sum(E[(i-TERMINATION_WINDOW+1):i])<TERMINATION_ERROR*TERMINATION_WINDOW
				print("Early Termination")
				break
			end
		end
		j = j+1
	end

	###### Save Model and Weights
	MODELS[2] = deepcopy(model)
	WEIGHTS[2,:,:] = deepcopy(C_guess)

	###### Start Refitting
	# k = 0
	TERMINATED = false

	###### Loop of Refits
	for REFIT_COUNT in 1:N_REFITS
		@everywhere GC.gc()# Clean Memory
		println("REFITTING... [$REFIT_COUNT/$N_REFITS]")
		flush(stdout)

		# Reset Learning rates for Re-fit
		if REFIT_COUNT == 1
			LEARNING_RATE= LEARNING_RATE/(DECAY^k)
			k=0
		end

		DECAY_PROB = 0.8
		RESET_PROB = 0.95
		RESET_RATE = 3
		DECAY_EFF = 0.98 #0.97 # Effective Decay rate
		DECAY_WINDOW = 15 #12 # Number of decays before reset
		DECAY = exp(log(DECAY_EFF/RESET_RATE)/DECAY_WINDOW)


		# Re-Make Improved MLP Model
		model = refitMLP(S, V, C_guess, REFIT_NEIGHBOURS, deepcopy(model), BATCH_SIZE, EPOCHS, η_REFIT)

		j=0 # initialise adaptive learning rate


		for i in 1:REFIT_EPOCHS
			current_index = MAX_EPOCHS + (REFIT_COUNT-1)*REFIT_EPOCHS + i # counter for tracking storage location in array
			LR[current_index] = deepcopy(LEARNING_RATE)

			batch_error, mini_grad = eval_grad(S, C_guess, MINI_BATCH_SIZE, T_IN, dt, wash, model; RK = true, omit = OMISSIONS, exact = false)

			if i == 1
				CURRENT_GRAD = -(mini_grad./(norm(mini_grad))).*LEARNING_RATE
				grad_step = CURRENT_GRAD
			else
				CURRENT_GRAD = -(mini_grad./(norm(mini_grad))).*LEARNING_RATE
				grad_step = M*PREV_GRAD + (1-M)*CURRENT_GRAD
			end

			PREV_GRAD = deepcopy(grad_step)

			# Update weight matrix
			C_guess = C_guess .+ deepcopy(grad_step)

			# Calculate normalised coupling error and save
			C_error = norm(C.*A.-C_guess)/norm(C.*A)
			C_E[current_index] = deepcopy(C_error)
			E[current_index] = deepcopy(batch_error)

			str = string("Epochs: [$i/$REFIT_EPOCHS]", ", ")
			str = string(str, "Prediction Error: $(round(E[current_index], digits = 8))", ", ")
			str = string(str, "C_Error Magnitude: $(round(C_E[current_index], digits = 8))", ", ")
			str = string(str, "Learn Rate: $(round(LEARNING_RATE, digits = 8))", ", ")
			str = string(str, "Time to Reset: $(DECAY_WINDOW-k)", "\n")

			print(str)
			flush(stdout)

			# Early termination if fully learned
			if current_index>TERMINATION_WINDOW
				if sum(E[(current_index-TERMINATION_WINDOW+1):current_index])<TERMINATION_ERROR*TERMINATION_WINDOW
					print("Early Termination")
					break
				end
			end
			j = j+1
		end

		MODELS[2+REFIT_COUNT] = deepcopy(model)
		WEIGHTS[2+REFIT_COUNT,:,:] = deepcopy(C_guess)
	end

	results = Dict("A" => deepcopy(A),
        "C" => deepcopy(C),
        "E" => deepcopy(E),
        "LR" => deepcopy(LR),
        "C_E" => deepcopy(C_E),
        "SCALE" => deepcopy(scale),
        "LOC" => deepcopy(loc),
        "MODELS" => deepcopy(MODELS),
        "WEIGHTS" => deepcopy(WEIGHTS),
        "dt" => deepcopy(dt),
        "μ" => deepcopy(μ),
        "σ" => deepcopy(σ),
        "T_IN" => deepcopy(T_IN),
        "MAX_EPOCHS" => deepcopy(MAX_EPOCHS),
        "REFIT_EPOCHS" => deepcopy(REFIT_EPOCHS),
        "REAL_MODEL" => deepcopy(real_model),
		"DATA" => deepcopy(S_original),
		"SMOOTHED_DATA" => deepcopy(S))
		

	return results
end

function MultiRun(filename;DICT_SMOOTH=DICT_SMOOTH, DICT_MLP=DICT_MLP, DICT_DATAGEN=DICT_DATAGEN, DICT_BACKPROP=DICT_BACKPROP, N_ITERATES = N_ITERATES)
	# %% Track Learning Data
	A_Arr = [] # Adjacency Matrix
	C_Arr = [] # Real Weights
	E_Arr = [] # Observation Error
	LR_Arr = [] # Learning Rate
	C_E_Arr = [] # Error of coupling compared to ground truth after filtering
	MODELS_Arr = [] #Store Models
	WEIGHTS_Arr = [] #Store Regressed Weights
	SCALE_Arr = []
	LOC_Arr = []
	S_DATA = []

	dt = deepcopy(DICT_DATAGEN["dt"])
	μ = deepcopy(DICT_DATAGEN["μ"])
	σ = deepcopy(DICT_DATAGEN["σ"])
	T_IN = deepcopy(DICT_BACKPROP["T_IN"])
	MAX_EPOCHS = deepcopy(DICT_BACKPROP["MAX_EPOCHS"])
	REFIT_EPOCHS = deepcopy(DICT_BACKPROP["REFIT_EPOCHS"])
	REAL_MODEL = deepcopy(DICT_DATAGEN["real_model"])
	NOISE = deepcopy(DICT_DATAGEN["noise"])

	for ITER in 1:N_ITERATES
		print("Iteration [$ITER / $N_ITERATES]")
		flush(stdout)
		output = SingleRun(DICT_SMOOTH=DICT_SMOOTH, DICT_MLP=DICT_MLP, DICT_DATAGEN=DICT_DATAGEN, DICT_BACKPROP=DICT_BACKPROP)

		A = output["A"]
		C = output["C"]
		E = output["E"]
		LR = output["LR"]
		C_E = output["C_E"]
		SCALE = output["SCALE"]
		LOC = output["LOC"]
		MODELS = output["MODELS"]
		WEIGHTS = output["WEIGHTS"]
		DATA = output["DATA"]

		A_Arr = push!(A_Arr, deepcopy(A)) # Adjacency Matrix
		C_Arr = push!(C_Arr, deepcopy(C)) # Real Weights
		E_Arr = push!(E_Arr, deepcopy(E)) # Observation Error
	    LR_Arr = push!(LR_Arr, deepcopy(LR)) # Learning Rate
	    C_E_Arr = push!(C_E_Arr, deepcopy(C_E)) # Error of coupling compared to ground truth after filtering
	    MODELS_Arr = push!(MODELS_Arr, deepcopy(MODELS)) #Store Models
	    WEIGHTS_Arr = push!(WEIGHTS_Arr, deepcopy(WEIGHTS)) #Store Regressed Weights
		SCALE_Arr = push!(SCALE_Arr, deepcopy(SCALE))
		LOC_Arr = push!(LOC_Arr, deepcopy(LOC))
		S_DATA = push!(S_DATA, deepcopy(DATA))
	end

	save(filename, Dict("A_Arr" => A_Arr,
	                                    "C_Arr" => C_Arr,
	                                    "E_Arr" => E_Arr,
	                                    "LR_Arr" => LR_Arr,
	                                    "C_E_Arr" => C_E_Arr,
	                                    "SCALE_Arr" => SCALE_Arr,
	                                    "LOC_Arr" => LOC_Arr,
	                                    "MODELS_Arr" => MODELS_Arr,
	                                    "WEIGHTS_Arr" => WEIGHTS_Arr,
	                                    "dt" => dt,
	                                    "μ" => μ,
	                                    "σ" => σ,
	                                    "T_IN" => T_IN,
	                                    "MAX_EPOCHS" => MAX_EPOCHS,
	                                    "REFIT_EPOCHS" => REFIT_EPOCHS,
	                                    "REAL_MODEL" => REAL_MODEL,
										"DATA" => S_DATA,
										"NOISE" => NOISE))
end
