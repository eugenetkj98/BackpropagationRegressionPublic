#These functions are used to estimate the vector field of local dynamics
# of a dynamical network via a mean field approach

using NearestNeighbors


function noisy_VF(S, dt)    # Extract Velocity Vectors where S is (N,T,Dim)
    # S_0 = zeros()
    # S_1 = zeros()
    # S_2 = zeros()
    points = zeros()
    velocities = zeros()
    N = size(S)[1]
    for i in 1:N
        if i == 1
            # # Forward Difference
            # S_0 = S[i,1:(end-1),:]
            # S_1 = S[i,2:end,:]
            # V = (S_1-S_0)/dt
            # points = deepcopy(S_0)

            # # Central difference
            # S_0 = S[i,1:(end-2),:]
            # S_1 = S[i,2:(end-1),:]
            # S_2 = S[i,3:end,:]
            # V = (S_2-S_0)/(2*dt)
            # points = deepcopy(S_1)

            # 5 Point Stencil Differentiation
            S_0 = S[i,1:(end-4),:]
            S_1 = S[i,2:(end-3),:]
            S_2 = S[i,3:(end-2),:]
            S_3 = S[i,4:(end-1),:]
            S_4 = S[i,5:end,:]
            V = (S_0-8*S_1+8*S_3-S_4)/(12*dt)
            points = deepcopy(S_2)


            velocities = deepcopy(V)

        else
            # Forward Difference
            # S_0 = S[i,1:(end-1),:]
            # S_1 = S[i,2:end,:]
            # V = (S_1-S_0)/dt
            # points = vcat(points, deepcopy(S_0))

            # # Central difference
            # S_0 = S[i,1:(end-2),:]
            # S_1 = S[i,2:(end-1),:]
            # S_2 = S[i,3:end,:]
            # V = (S_2-S_0)/(2*dt)
            # points = vcat(points, deepcopy(S_1))

            # 5 Point Stencil Differentiation
            S_0 = S[i,1:(end-4),:]
            S_1 = S[i,2:(end-3),:]
            S_2 = S[i,3:(end-2),:]
            S_3 = S[i,4:(end-1),:]
            S_4 = S[i,5:end,:]
            V = (S_0-8*S_1+8*S_3-S_4)/(12*dt)
            points = vcat(points, deepcopy(S_2))

            velocities = vcat(velocities, deepcopy(V))
        end
    end

    # V = (S_2-S_0)/(2*dt)
    return points, velocities#S_1,V
end

"""
Modified convert states S, and velocities V into a collection of points and corresponding velocities
To be used as training data for neural network
"""

function pointify_VF(S, V)    # Extract Velocity Vectors where S is (N,T,Dim)
    points = zeros()
    velocities = zeros()
    indexes = zeros(Int,size(S)[1], size(S)[2])
    N = size(S)[1]
    for i in 1:N
        if i == 1
            points = deepcopy(S[i,:,:])
            velocities = deepcopy(V[i,:,:])
            indexes[i,:] = 1:size(S[i,:,:])[1]
        else
            points = vcat(points, deepcopy(S[i,:,:]))
            velocities = vcat(velocities, deepcopy(V[i,:,:]))
            indexes[i,:] = (1:size(S[i,:,:])[1]) .+ indexes[i-1,end] 
        end
        
    end
    return points, velocities, indexes
end

"""
This function smooths out the numerical estimation of the vector field.
S_0 and V are dimensions (n_points, 3)
"""

function smooth_VF(S_0,V; n_neighbours = 20)
    

    #Find IDs of the k nearest neighbours
    kdtree = KDTree(S_0')
    idxs, dists = knn(kdtree, S_0', n_neighbours, true)

    #Declare variable for storing output
    n_points = length(idxs)
    dims = size(S_0)[2]
    ave_V = zeros((dims, n_points))
    points = zeros((dims, n_points))

    # Average out the points
    for i in 1:n_points
        points[:,i] = S_0[i,:]
        average = zeros(dims)
        # max_dist = maximum(dists[i])
        for idx in idxs[i]
            #Tally with scaling factor
            average = average + V[idx,:]#*(1-dists[i][j]/max_dist)
        end
        ave_V[:,i] = average./n_neighbours
    end

    return points,ave_V
end
