"""
This script runs a series of backpropagation regression experiments with the written settings.
List of allowable chaotic oscillators are found in ChaoticNetwork2
"""

using ProgressBars
using LinearAlgebra
using FileIO
using CUDA
using SharedArrays
using Distributed
# %% Initiate system for multiprocessing
MINI_BATCH_SIZE = 8 #Number of parallel processes for regression.
addprocs(MINI_BATCH_SIZE)
@everywhere MINI_BATCH_SIZE = 8
println("System initialised with $(nworkers()) concurrent workers, each with $(Threads.nthreads()) threads.")
flush(stdout)

# Include relevant packages
@everywhere begin
	include("BackpropRegressionGeneral2.jl")
	include("ChaoticNetwork2.jl")
	include("MLP2.jl")
	include("MeanFieldVF2.jl")
	include("UncoupleSystems2.jl")
	include("EvaluationFunctions.jl")
	include("RunFunctions.jl")
	include("NoiseFilter.jl")
end

# %% Smoothing Settings for Mean Field Approximation
@everywhere begin
	DICT_SMOOTH = Dict(
	"INIT_NEIGHBOURS" => 8, # Number of neighbours taken to smooth data for VFL
	"REFIT_NEIGHBOURS" => 1
	)
end

# %% Feedforward Neural Network MLP Settings
@everywhere begin
	η = 0.001 #Initial learning rate
	DICT_MLP = Dict(
	"N_HIDDEN" => 128, # Number of nodes in each hidden layer
	"BATCH_SIZE" => 512,
	"EPOCHS" => 30,
	"ENSEMBLE_SIZE" => 1, # Number of candidate neural networks to use (Not functional)
	"η" => η,
	"η_REFIT" => 0.2*η # Learning rate of neural network in refitting stage.
	)
end

# %% Data Hyperparameters
@everywhere begin
	N = 16	# Number of nodes in dynamical network
	p = log(N)/N # Connection probability
	DICT_DATAGEN = Dict(
	"N" => N,
	"T" => 25000, # Number of observed timesteps
	"wash" => 2000, # Number of timesteps in washout period to remove transients
	"p" => p,
	"dt" => 0.02, ######
	"K" => 0.1, # Old variable for coupling strength (Redundant)
	"μ" => 0.15, # Mean coupling strength
	"σ" => 0.02, # Standard deviation of coupling strength
	"dims" => 3, # Dimension of chaotic oscillator (should be set to 3 or 4 (for FitzHugh-Nagumo))
	"real_model" => lorenz!, # Chaotic system selected from ChaoticNetwork2
	"noise" => 0.00, # Amount of noise
	"window" => 2 # Old variable (Redundant)
	)
end

# %% Initial Backprop Regression Hyper parameters
@everywhere begin
	RESET_RATE = 2
	DECAY_EFF = 0.98 # Effective Decay rate
	DECAY_WINDOW = 5 # Number of decays before reset
	NODE_LEARNING_RATE = 0.0005 # Average learning rate for each node in each regression step
	DICT_BACKPROP = Dict(
	"NODE_LEARNING_RATE" => NODE_LEARNING_RATE,
	"START_LEARNING_RATE" => sqrt(p*N*(N-1)*NODE_LEARNING_RATE^2),
	"LR_ADJ_RATIO" => 0.85, # Amount to decay learning rate in scheduler
	"MAX_EPOCHS" => 500, # First time to re-fit
	"REFIT_EPOCHS" => 300, # Number of regression steps in each backprop iteration
	"N_REFITS" => 40, # Number refit iterations
	"MINI_BATCH_SIZE" => MINI_BATCH_SIZE,
	"TERMINATION_ERROR" => 1e-11, # Prediction error threshold for automatic termination
	"TERMINATION_WINDOW" => 5,
	"PATIENCE" => 20,
	"T_IN" => 10, # Number of freerun prediction steps for each regression step
	"DECAY_PROB" => 0.8, # Learning rate scheduler parameter
	"RESET_PROB" => 0.95, # Learning rate scheduler parameter
	"RESET_RATE" => RESET_RATE, 
	"DECAY_EFF" => DECAY_EFF, # Effective Decay rate
	"DECAY_WINDOW" => DECAY_WINDOW, # Number of decays before reset
	"DECAY" => exp(log(DECAY_EFF/RESET_RATE)/DECAY_WINDOW),
	"M" => 0.9, # Momentum
	"OMISSIONS" => 0, # Number of Hidden nodes
	)
end

# Number of iterates to run system, save filename
N_ITERATES = 3 # Number of consecutive experiments to run
filename = "MultiTest_Lorenz16.jld2" # Name of saved file

# %% Run Experiment
MultiRun(filename;DICT_SMOOTH=DICT_SMOOTH, DICT_MLP=DICT_MLP, DICT_DATAGEN=DICT_DATAGEN, DICT_BACKPROP=DICT_BACKPROP, N_ITERATES = N_ITERATES)
