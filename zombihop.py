import utils
from acquisitions import *
import numpy as np
import pandas as pd
import random


class ZombiHop:
    def __init__(self, seed, X_init, Y_init, Y_experimental, Gammas, alphas, n_draws_per_activation, acquisition_type,
                 tolerance, penalty_width, m, k, lower_bound, upper_bound, resolution, sampler):
        '''
        Runs the ZoMBI-Hop optimization procedure over
        #Gammas hops, #alphas ZoMBI zoom-ins per hop, and #n_draws_per_activation experiments drawn per zoom-in.

        :param seed                     A random seed for model reproducibility
                                        IMPORTANT: *change the seed value every time a new independent trial is run*
        :param X_init:                  An (n, d) array of normalized [0,1] initialization coordinates,
                                        where n is the number of initial points and d is the number of dimensions
        :param Y_init:                  An (n, 1) array of values, corresponding to the X_init coordinates,
                                        where n is the number of initial points
        :param Y_experimental           A model of the form Y_experimental() that can take an (n, d) dimensional array
                                        of coordinates (X) to predict an output (Y) => supplements for experimental data
        :param Gammas:                  Number of hops to other needles
        :param alphas:                  Number of ZoMBI zoom-in for each hop
        :param n_draws_per_activation:  Number of samples drawn for each zoom-in
        :param acquisition_type:        Acquisition function model to conduct sampling of new points
        :param tolerance:               Error tolerance of GP prediction, used to end a ZoMBI zoom-in
                                        and move to the next needle
        :param penalty_width:           Width of penalty region about needle => inhibits BO searching from areas
                                        surrounding previously found needles
        :param m:                       Top m-number of data points used to zoom in bounds
        :param k:                       Top k-number of data points to keep
        :param lower_bound:             An (d,) array of lower bound coordinates for each dimension (d) to begin the search
        :param upper_bound:             An (d,) array of upper bound coordinates for each dimension (d) to begin the search
        :param resolution:              Number for the resolution of the mesh search space, e.g., resolution=10
        :param sampler:                 Either None or a sampler that takes in arguments (X_ask, dimension_meshes, acquisitions, Y_experimental) and
                                        outputs X_tell, Y_tell with X_tell (shape=(n,d)); Y_tell (shape=(n,)) for n samples and d dimensions.
                                        dimension_meshes (shape=(n,d)) are the X-values to the acquisitions (shape=(n,)) values. 
        '''

        if not (acquisition_type == LCB_ada or acquisition_type == EI):
            raise ValueError("Variable \"acquisition_type\" must be set to one of the following objects: EI or LCB_ada")

        # Input parameters
        self.seed = seed
        self.X_init = X_init
        self.Y_init = Y_init
        self.Y_experimental = Y_experimental
        self.Gammas = Gammas
        self.alphas = alphas
        self.n_draws_per_activation = n_draws_per_activation
        self.acquisition_type = acquisition_type
        self.tolerance = tolerance
        self.penalty_width = penalty_width
        self.m = m
        self.k = k
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.resolution = resolution
        self.sampler = sampler

        self.ftype = np.float16 # set formatting data type to float16 for memory conservation, float64 is too large

        # acquisition function parameters
        self.ratio = 3.
        self.decay = 0.95
        self.xi = 0.1


    def run_virtual(self, verbose = False, plot = True):
        # === INITIALIZATION ===#
        random.seed(self.seed)
        GP = utils.reset_GP(self.X_init, self.Y_init)  # initialize the first GP model
        lower_bound = self.lower_bound # initialize bounds based on user-selection
        upper_bound = self.upper_bound # initialize bounds based on user-selection
        full_mesh = utils.bounded_mesh(self.X_init.shape[1], lower_bound, upper_bound, self.ftype, self.resolution)  # full, un-zoomed mesh. This will not be updated
        dimension_meshes = utils.bounded_mesh(self.X_init.shape[1], lower_bound, upper_bound, self.ftype, self.resolution)  # mesh that will dynamically zoom in each iteration
        penalty_mask = np.ones((dimension_meshes.shape[0], 1)).astype(self.ftype)  # init penalty mask as all ones => has no multiplicative effect
        X_intermediate, Y_intermediate, X_all, Y_all, X_GPmemory, Y_GPmemory, X_BOUNDmemory, Y_BOUNDmemory, X_final, Y_final, needles, needle_locs = utils.initialize_arrays(self.X_init, self.Y_init)
        m_bias = 0 # bias to selecting m-best points for creating the bounds. Increasing the bias slides the window towards worse performing points to generate different boundary when no higher-perfoming points exist
        # ======================#

        # Begin Optimization Procedure #
        for Gamma in range(self.Gammas):  # number of zoom-outs to find new needles
            print(f'\nInitiating hop . . . ({Gamma + 1}/{self.Gammas})')
            error_list = []  # create error list for each zoom-in to determine deviation of model prediction versus "experimental"
            prune = True  # set prune to true again, attempt another zoom-in

            for alpha in range(self.alphas):  # number of ZoMBI activations
                print(f'\n    Activate ZoMBI . . . ({alpha + 1}/{self.alphas})')

                # === CREATE PENALTY MASK ===#
                # for every round of ZoMBI, create a new penalty mask using the previously stored needle values
                if Gamma == 0 or len(needle_locs) == 0:  # if it is the first round or if there have been no found needles, then pass.
                    pass  # use no penalty for the first series zoom-ins at Gamma = 0
                else:  # only create penalty mask for subsequent needle hops
                    penalty_mask = utils.create_penalty_mask(needle_locs, dimension_meshes, self.ftype, self.penalty_width)
                    if plot:
                        utils.plot_penalty_dist(penalty_mask, lower_bound, upper_bound)
                if np.sum(~penalty_mask.astype(bool)) / len(penalty_mask) >= 0.9:  # if the bounded area is penalized,by more than 90%, then zoom out and do not prune memory
                    print('Bounded region is over 90% penalized. Zooming out.')
                    prune = False  # do not prune if region is fully penalized and cannot zoom in => must keep sampling
                    m_bias += 1  # update sliding m-best point selector bias. Will shift window for m-best points when creating zooming bounds
                    break
                    # ==========================#

                inc = 1 / self.n_draws_per_activation
                for n in range(self.n_draws_per_activation):  # number of draws per ZoMBI activation
                    inc = utils.progress_bar(n, self.n_draws_per_activation, inc)
                    # acquisition function for mesh
                    acquisition = self.acquisition_type(X=dimension_meshes, GP_model=GP,  n=n, fX_best=Y_BOUNDmemory.min(), ratio=self.ratio, decay=self.decay, xi=self.xi, ftype=self.ftype)
                    acquisition = acquisition * penalty_mask
                    
                    # gather the best X-value to ask the virtual "experiment" to create
                    X_ask = dimension_meshes[np.argmax(acquisition),:].reshape(1,-1)

                    # tell the model what experiments were created
                    if self.sampler: # if using a sampler, required outputs: X_tell shape = (n,d); Y_tell shape = (n,) for n samples and d dimensions
                        X_tell, Y_tell = self.sampler(X_ask, dimension_meshes, acquisition, self.Y_experimental) 
                    else:
                        Y_tell = self.Y_experimental(X_ask)
                        X_tell = X_ask

                    # check posterior of surrogate with Y_tell, only check best performer of Y_tell
                    Y_tell_min = np.min(Y_tell) # get minimum Y_tell
                    X_tell_min = X_tell[np.argmin(Y_tell), :].reshape(1,-1) # get corresponding minimum X_tell
                    mu, std = utils.GP_pred(X_tell_min, GP, self.ftype) # check surrogate posterior at corresponding minimum X_tell
                    ymodel = mu[0][0]

                    # compute error between surrogate posterior and "experimental"
                    error = np.abs(Y_tell_min - ymodel)/np.abs(Y_tell_min)
                    error_list.append(error)

                    # get "experimental" measurements from the BO prediction for X and Y
                    newframeX = pd.DataFrame(X_tell, columns = X_intermediate.columns.values)
                    newframeY = pd.DataFrame(Y_tell.reshape(len(Y_tell),1), columns = Y_intermediate.columns.values)

                    if verbose:
                        print(f'\nactual: {Y_tell_min}')
                        print(f'model: {ymodel}')
                        print(f'error: {error}')
                        print(newframeX)

                    # append experimental measurements to intermediate set
                    X_intermediate = pd.concat([X_intermediate, newframeX], ignore_index=True)
                    Y_intermediate = pd.concat([Y_intermediate, newframeY], ignore_index=True)
                    # append experimental measurements to memory sets
                    X_GPmemory = pd.concat([X_GPmemory, newframeX], ignore_index=True)
                    Y_GPmemory = pd.concat([Y_GPmemory, newframeY], ignore_index=True)
                    X_BOUNDmemory = pd.concat([X_BOUNDmemory, newframeX], ignore_index=True)
                    Y_BOUNDmemory = pd.concat([Y_BOUNDmemory, newframeY], ignore_index=True)

                    # update GP
                    GP.fit(X_GPmemory, Y_GPmemory);  # contains init dataset, intermediate points [from current iteration ONLY], and top-k memory pruned points [from prior iterations]

                    # === CHECK FOR CONVERGENCE ON NEEDLE ===#
                    # check to see if we have found a needle that converges to within the set tolerance between: experimental minus GP prediction divided by experimental
                    if np.sum(np.array(error_list[-3:]) <= self.tolerance) >= 3:
                        break
                # break inner loop if we have found a needle.
                if np.sum(np.array(error_list[-3:]) <= self.tolerance) >= 3:
                    print('\nFound a needle!')
                    needles = pd.concat([needles, Y_intermediate[-1:]])
                    needle_locs = pd.concat([needle_locs, X_intermediate[-1:]])
                    break  # end zoom-in if the needle is found
                    # ========================================#

                # === COMPUTE BOUNDS ===#
                # find the m-best high-performing measured points to compute the zooming bounds: use init dataset + intermediate points
                top_m = np.argsort(np.array(Y_BOUNDmemory), axis=0)[m_bias:(self.m + m_bias)].reshape(-1)  # m-bias is used to slide the bound window towards lower-performing points to avoid zooming into the same region consecutively
                lower_bound = np.min(X_BOUNDmemory.iloc[top_m, :], axis=0)  # get all dim lower bounds from M points.
                upper_bound = np.max(X_BOUNDmemory.iloc[top_m, :], axis=0)  # get all dim upper bounds from M points
                # compute new bounds for ZoMBI
                dimension_meshes = utils.bounded_mesh(X_BOUNDmemory.shape[1], lower_bound, upper_bound, self.ftype, self.resolution)  # zoom in and update search mesh
                # ======================#

            # keep track of all data points, including all pruned points => append intermediate points before pruning. X,Y_all are not used in ZoMBI, only for record keeping and plotting
            X_all = pd.concat([X_all, X_intermediate], ignore_index=True)
            Y_all = pd.concat([Y_all, Y_intermediate], ignore_index=True)

            if prune:  # only prune if zoom-in is successful
                print(f'\nPruning memory and zooming out. Keep top k={self.k} points\n')
                # prune memory, keep only the top k-number of data points
                #         prior_n = len(Y_final) # number of data points present in data set from prior iteration
                X_pruned, Y_pruned = utils.memory_prune(self.k, self.tolerance, X_intermediate, Y_intermediate)
                # update final df
                X_final = pd.concat([X_final, X_pruned], ignore_index=True)
                Y_final = pd.concat([Y_final, Y_pruned], ignore_index=True)
                # update to intermediate array to wipe memory. Bounding occurs using only points accumulated during that ZoMBI activation => reset after each hop (Gamma)
                X_intermediate = pd.DataFrame([], columns=X_all.columns.values)  # contains only intermediate points
                Y_intermediate = pd.DataFrame([], columns=Y_all.columns.values)
                # remove memory of intermediate points after successful alpha-number of ZoMBI activation => only keep init + top-k points
                X_GPmemory = X_final.copy()  # contains the memory of what ZoMBI is able to see
                Y_GPmemory = Y_final.copy()
                # for bounds, keep only init dataset + current iteration's intermediate points
                X_BOUNDmemory = self.X_init.copy()  # contains the memory of what ZoMBI is able to see
                Y_BOUNDmemory = self.Y_init.copy()
                # reset the bias to selecting m-best points for creating the bounds
                m_bias = 0

            # Zoom back out to find other needles by resetting bounds
            lower_bound = np.zeros(self.X_init.shape[1])
            upper_bound = np.ones(self.X_init.shape[1])
            dimension_meshes = full_mesh  # reset search bounds

        return X_all, Y_all, needle_locs, needles
