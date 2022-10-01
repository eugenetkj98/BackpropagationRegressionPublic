# These functions implement a simple MLP that learns a mapping form state space to vector field
using Flux
using ProgressBars

# %%
function define_model(n_in, n_out, n_hidden)
    model = Chain(
                # Dropout(0.2),
                Dense(n_in, n_hidden, σ),
                # Dropout(0.1),
                Dense(n_hidden, n_hidden, σ),
                # Dropout(0.1),
                # Dense(n_hidden, n_hidden, σ),
				# Dense(n_hidden, n_hidden, σ),
                # Dropout(0.2),
                Dense(n_hidden, n_out))
    return model
end


# %% Custom Training Function with error reporting
function custom_train!(loss, ps, data, opt)
  # training_loss is declared local so it will be available for logging outside the gradient calculation.
  local training_loss
  TL = []
  for d in data
    gs = gradient(ps) do
      training_loss = loss(d...)
      return training_loss
    end
    push!(TL,deepcopy(training_loss))
    Flux.update!(opt, ps, gs)
  end

  return mean(TL)
end
# Train system
function MLP(input_train, output_train, model; batchsize = 64, max_epochs = 20, η = 0.001)
    # Load Data
    train_DL = Flux.DataLoader((input_train, output_train), batchsize =batchsize, shuffle = true)

    # Define loss, params and optimiser
    ps = params(model)
    opt = Flux.Optimise.ADAM(η)
    loss(x,y) = Flux.Losses.mse(model(x),y)

    for epoch in 1:max_epochs
        batch_loss = custom_train!(loss, ps, train_DL, opt)
        if epoch%2==0
            print("Epoch [$epoch/$max_epochs]: Training Loss = $batch_loss\n")
			flush(stdout)
        end
    end
end

function trainModelGPU(input, output, model, BATCH_SIZE, EPOCHS, η)
    input_train = input |> gpu
    output_train = output |> gpu
    model_GPU = trainmode!(model) |> gpu
    MLP(input_train, output_train, model_GPU, batchsize = BATCH_SIZE, max_epochs = EPOCHS, η=η)
    return deepcopy(testmode!(model_GPU |> cpu))
end

# %%

function partial_diff(model, point; δ=0.001)
    dims = length(point)
    ∂ = zeros(dims,dims)
    for J in 1:dims
        p1 = deepcopy(point)
        p2 = deepcopy(point)
        p1[J] = p1[J]+δ
        p2[J] = p2[J]-δ
        ∂[:,J] = (model(p1)-model(p2))/(2*δ)
    end
    return ∂
end
