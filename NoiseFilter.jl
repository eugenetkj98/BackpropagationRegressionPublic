using SmoothingSplines

# Spline smoothing according to Reinsch and Green
# Reinsch, Christian H. "Smoothing by spline functions." Numerische mathematik 10.3 (1967): 177-183.
# Green, Peter J., and Bernard W. Silverman. Nonparametric regression and generalized linear models: a roughness penalty approach. CRC Press, 1993.
# Timeseries and λ = regularisation parameter
function spline_smoothing(timeseries; λ = 1.0)
    X = (1:1:size(timeseries)[1])*1.0
    Y = timeseries
    spl = fit(SmoothingSpline, X, Y, λ)
    Ypred = predict(spl)
    return Ypred
end

# Smoothing using a moving average with total moving window size of (2*window+1)
function MA_smoothing(timeseries; window = 2)
    smoothed_timeseries = zeros(length(timeseries))
    idx = -window:1:window
    weights = exp.(-(idx./window).^2)
    for i in 1:length(timeseries)
        if (i>window) & (i<length(timeseries)-window)
            # smoothed_timeseries[i] = dot(weights,timeseries[i-window:i+window])/sum(weights)
            smoothed_timeseries[i] = mean(timeseries[i-window:i+window])
        end
    end
    return smoothed_timeseries
end