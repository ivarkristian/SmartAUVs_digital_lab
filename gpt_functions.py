import gpytorch
import torch
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from copy import deepcopy
import random
from shapely.geometry import Point, Polygon, MultiPoint
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.path import Path
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D

def compute_exact_mll(y, K, u=None):
    """
    Computes the Exact Marginal Log Likelihood (MLL) for Gaussian Process models.

    Parameters:
    - K: numpy array, the covariance matrix derived from the GP kernel.
    - y: numpy array, the vector of observations.

    Returns:
    - The exact marginal log likelihood.
    """
    n = K.size()[0]  # Number of observations
    K_numpy = K.numpy()
    y_numpy = y.numpy()
    if u is not None:
        u_numpy = u.numpy()
        diff = y_numpy - u_numpy
    
    K_inv = np.linalg.inv(K_numpy)  # Inverse of the covariance matrix
    log_det_K = np.log(np.linalg.det(K_numpy))  # Log determinant of the covariance matrix
    
    # Compute the three terms of the MLL formula
    term1 = -0.5 * np.dot(diff.T, np.dot(K_inv, diff))
    term2 = -0.5 * log_det_K
    term3 = -0.5 * n * np.log(2 * np.pi)
    
    # Sum the terms to get the exact MLL
    mll = term1 + term2 + term3
    return mll

import numpy as np

def compute_exact_mll_individually(y, K, u=None, jitter=1e-6):
    """
    Computes the Exact Marginal Log Likelihood (MLL) for Gaussian Process models
    for each observation.

    Parameters:
    - K: numpy array, the covariance matrix derived from the GP kernel.
    - y: numpy array, the vector of observations.
    - u: numpy array, the mean vector (optional).

    Returns:
    - An array with the log likelihood for each observation.
    """
    # Convert tensors to numpy arrays if necessary
    if not isinstance(K, np.ndarray):
        K_numpy = K.numpy()
    else:
        K_numpy = K

    if not isinstance(y, np.ndarray):
        y_numpy = y.numpy()
    else:
        y_numpy = y

    if u is not None:
        if not isinstance(u, np.ndarray):
            u_numpy = u.numpy()
        else:
            u_numpy = u
        diff = y_numpy - u_numpy
    else:
        diff = y_numpy

    # Add jitter to the diagonal elements of K to ensure numerical stability
    K_numpy += jitter * np.eye(K_numpy.shape[0])

    # Calculate the inverse and the log determinant of the covariance matrix
    K_inv = np.linalg.inv(K_numpy)
    #print(f'np.linalg.det(K_numpy): {np.linalg.det(K_numpy)}')
    #log_det_K = np.log(np.linalg.det(K_numpy))
    #print(f'np.linalg.slogdet(K_numpy): {np.linalg.slogdet(K_numpy)}')
    log_det_K = np.linalg.slogdet(K_numpy)[1]
    # Number of observations
    n = K_numpy.shape[0]

    # Initialize the array to hold individual log likelihoods
    log_likelihoods = np.zeros(n)
    for i in range(n):
        # Calculate the log likelihood for each observation
        term1 = -0.5 * diff[i] * np.dot(K_inv[i], diff)
        term2 = -0.5 * log_det_K / n  # Distribute the log determinant term equally
        term3 = -0.5 * np.log(2 * np.pi)
        
        log_likelihoods[i] = term1 + term2 + term3

    return log_likelihoods

# Example usage:
# K = np.array([[...], [...], ...])  # covariance matrix
# y = np.array([...])  # observations
# log_likelihoods = compute_exact_mll(y, K)


def compute_weighted_mll(y, K, weights=None, u=None, jitter=1e-6):
    """
    Computes the Weighted Marginal Log Likelihood (MLL) for Gaussian Process models.

    Parameters:
    - y: numpy array, the vector of observations.
    - K: numpy array, the covariance matrix derived from the GP kernel.
    - weights: numpy array, the vector of weights for each observation.
    - u: numpy array, the vector of means (optional).
    - jitter: float, a small value added to the diagonal for numerical stability.

    Returns:
    - The weighted marginal log likelihood.
    """
    n = K.shape[0]  # Number of observations
    K_numpy = K if isinstance(K, np.ndarray) else K.numpy()
    y_numpy = y if isinstance(y, np.ndarray) else y.numpy()
    
    if weights is None:
        weights = np.ones(n)
    else:
        weights = weights if isinstance(weights, np.ndarray) else weights.numpy()
    
    if u is not None:
        u_numpy = u if isinstance(u, np.ndarray) else u.numpy()
        diff = y_numpy - u_numpy
    else:
        diff = y_numpy
    
    # Filter out elements with zero weights
    non_zero_indices = np.where(weights != 0)[0]
    K_filtered = K_numpy[np.ix_(non_zero_indices, non_zero_indices)]
    y_filtered = y_numpy[non_zero_indices]
    diff_filtered = diff[non_zero_indices]
    weights_filtered = weights[non_zero_indices]
    
    # Adjust the covariance matrix K by the filtered weights
    W_filtered = np.diag(weights_filtered)
    K_weighted = np.dot(W_filtered, np.dot(K_filtered, W_filtered))
    
    # Add jitter to the diagonal to ensure invertibility
    K_weighted += np.eye(K_weighted.shape[0]) * jitter
    
    K_inv = np.linalg.inv(K_weighted)  # Inverse of the weighted covariance matrix
    #log_det_K = np.log(np.linalg.det(K_weighted))  # Log determinant of the weighted covariance matrix
    log_det_K = np.linalg.slogdet(K_weighted)[1]

    # Compute the three terms of the weighted MLL formula
    term1 = -0.5 * np.dot(diff_filtered.T, np.dot(K_inv, diff_filtered))
    term2 = -0.5 * log_det_K
    term3 = -0.5 * len(y_filtered) * np.log(2 * np.pi)
    #print(f'term1: {term1}, term2: {term2}, term3: {term3}')
    
    # Sum the terms to get the weighted MLL
    weighted_mll = term1 + term2 + term3
    #print(f'{term1}, {term2}, {term3}')
    return weighted_mll

# Example usage:
# y = np.array([...])  # Your observations
# K = np.array([...])  # Your covariance matrix
# weights = np.array([...])  # Your weights for each observation
# u = np.array([...])  # Optional, your mean values

# weighted_mll = compute_weighted_mll(y, K, weights, u)


def validate_model(model, val_coords, val_values, mode='log_prob', debug=False):
    val_model = deepcopy(model)
    val_model.set_train_data(val_coords, val_values, strict=False)
    val_model.eval()
    val_model.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        validation_output = val_model(val_coords)
        
        if mode == 'log_prob':
            pred = val_model.likelihood(validation_output)
            log_prob = -pred.log_prob(val_values)
            if debug:
                print(f'Validation nlog_prob: {log_prob.item()}')
            return log_prob
        
        if mode == 'mll':
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(val_model.likelihood, val_model)
            validation_loss = -mll(validation_output, val_values)
            if debug:
                print(f'Validation mll loss: {validation_loss.item()}')
            return validation_loss
    
    print(f'mode {mode} not recognized')
    return

def train_model(coords, values, model, iter=100, lr=0.1, early_delta=(False, 'mll', 0, 0, 0), debug=False, resps=None):
    
    #model.eval()
    #model.likelihood.eval()
    #with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #    observed_pred = model.likelihood(model(coords)) # Generate test_coords along the longitude coords
    
    model.train()
    model.likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    loocv = gpytorch.mlls.LeaveOneOutPseudoLikelihood(model.likelihood, model)
    lowest_val_loss = torch.inf
    e_delta = early_delta[0] # early stopping trigger
    e_mode = early_delta[1]
    val_coords = early_delta[2]
    val_values = early_delta[3]
    delay_validation = early_delta[4]
    e_stop = False

    percentage = 10
    p = round(iter/percentage)
    print_number = 0
    c = 0

    print(f'Training for {iter} iterations, lr = {lr}, early stopping = {e_delta}')
    #if resps is not None:
        #with torch.no_grad():
            #noises = model.likelihood.noise/resps
            #model.add_noises = (1-resps)*5
            #llh = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noises, learn_additional_noise=False)

    for i in range(iter):
        c += 1
        if c > p:
            c = 0
            print_number += percentage
            if not debug:
                print(f'...{print_number}%', end='')
        
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        if resps is not None:
            #output = model(coords, add_training_noises=(1/resps - 1)**2) # Barter Island scaling
            #output = model(coords, add_training_noises=(1/resps) - 1) # Barter Island scaling 2
            output = model(coords, add_training_noises=(model.likelihood.noise/resps)) # Tjotta scaling
        else:
            output = model(coords)

        # Calc loss and backprop gradients
        loss = -mll(output, values)
        #loss = -loocv(output, values)
        loss.backward()
        model.save_loss((i, loss.item()))
        if debug:
            if (i > 0) and (i%percentage == 0):
                print(f'Iter {i}/{iter} - Loss: {loss.item()}')
                model.print_named_parameters()
        optimizer.step()

        # Periodically validate the model
        if i % 2 == 0 and e_delta and i > delay_validation:  # Validate every n iterations
            validation_loss = validate_model(model, val_coords, val_values, mode=e_mode, debug=debug)
            model.save_val_loss((i, validation_loss.item()))
            
            if validation_loss < lowest_val_loss:
                lowest_val_loss = validation_loss
            elif validation_loss >= lowest_val_loss + e_delta:
                e_stop = True
        
        if e_stop:
            print(f'')
            print(f'Early stopping at iteration {i} of {iter}: {validation_loss.item()} > {lowest_val_loss} + e_delta')
            print(f'Iter {i}/{iter} - Loss: {loss.item()}')
            model.print_named_parameters()
            print(f'')
            break
    
    if not debug:
        print(f'..100%')
    return

def compute_preds(models, coords):
    preds = []
    for model in models:
        model.eval()
        model.likelihood.eval()
        with torch.no_grad():
            # Update responsibilities
            pred = model.likelihood(model(coords))
            preds.append(pred)
    return preds

def compute_log_probs(models, coords, values, resps=None, jitter=1e-6):
    
    num_models = len(models)
    num_coords = coords.size(0)
    n = values.size(0)
    log_probs = torch.full((n, num_models), 1.0/num_models)
    pred_output = []

    for i, model in enumerate(models):
        #model.eval()
        #model.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if resps is not None:
                model.train()
                model.likelihood.train()

                #noises = (1/resps[:, i] - 1)**2 # Barter Island weighting
                #noises = (1/resps[:, i]) - 1 # Barter Island weighting 2
                noises = model.likelihood.noise/resps[:, i] # Tjøtta weighting
                preds = model.likelihood(model(coords, add_training_noises=noises))
            else:
                preds = model.likelihood(model(coords))
                #preds = model(coords)

            #preds = model(coords)
            #log_prob = torch.stack([preds.log_prob(y) for y in values])
            #log_prob = preds.log_prob(values.unsqueeze(1))
            log_prob = compute_exact_mll_individually(values, preds.covariance_matrix, preds.mean, jitter)
            log_probs[:, i] = torch.tensor(log_prob)
            pred_output.append(preds)
    
    return log_probs, pred_output

def compute_log_probs_denorm(models, coords, values):
    
    num_models = len(models)
    num_coords = coords.size(0)
    n = values.size(0)
    log_probs = torch.full((n, num_models), 1.0/num_models)
    logtwopi = torch.log(torch.tensor([2.0])*torch.pi)
    denorm = 0.5*num_coords*logtwopi
    pred_output = []

    for i, model in enumerate(models):
        model.eval()
        model.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = model.likelihood(model(coords))
            #preds = model(coords)
            #log_prob = torch.stack([preds.log_prob(y) for y in values])
            log_prob = preds.log_prob(values.unsqueeze(1)) + denorm
            log_probs[:, i] = log_prob
            pred_output.append(preds)
    
    return log_probs, pred_output

def compute_log_probs_total(models, coords, values):

    num_models = len(models)
    log_probs = torch.full((1, num_models), 1.0/num_models)
    for i, model in enumerate(models):
        model.eval()
        model.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Update responsibilities
            preds = model.likelihood(model(coords))
            #preds = model(coords)
            #log_prob = torch.stack([preds.log_prob(y) for y in values])
            log_prob = preds.log_prob(values)
            #print(f'log_prob:{log_prob}')
            log_probs[0, i] = log_prob.item()
    
    return log_probs

def compute_log_probs_total_denorm(models, coords, values):

    num_models = len(models)
    num_coords = coords.size(0)
    log_probs = torch.full((1, num_models), 1.0/num_models)
    logtwopi = torch.log(torch.tensor([2.0])*torch.pi)
    denorm = 0.5*num_coords*logtwopi
    for i, model in enumerate(models):
        model.eval()
        model.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Update responsibilities
            preds = model.likelihood(model(coords))
            #preds = model(coords)
            #log_prob = torch.stack([preds.log_prob(y) for y in values])
            log_prob = preds.log_prob(values) + denorm
            #print(f'log_prob:{log_prob}')
            log_probs[0, i] = log_prob.item()
    
    return log_probs

def compute_responsibilities(log_probs, normalize=False):

    if normalize:
        max_log_probs = torch.max(log_probs, dim=1)[0].unsqueeze(1)
        log_probs_n = log_probs - max_log_probs
    else:
        log_probs_n = log_probs

    log_probs_sum = torch.logsumexp(log_probs_n, dim=1, keepdim=True)
    log_responsibilities = log_probs_n - log_probs_sum

    return log_responsibilities

def compute_resps(X, gp_models, mixing_coeffs):
    """
    Compute the responsibilities for each data point under each GP model in the mixture.
    
    Parameters:
    - X: Tensor of data points.
    - gp_models: List of GP models.
    - mixing_coeffs: Tensor of mixing coefficients (pi_i's).
    
    Returns:
    - responsibilities: Tensor of responsibilities for each data point under each model.
    """
    num_data_points = X.size(0)
    num_models = len(gp_models)
    log_responsibilities = torch.zeros(num_data_points, num_models)
    
    for i, gp_model in enumerate(gp_models):
        # Set model and likelihood to eval mode
        gp_model.eval()
        likelihood = gp_model.likelihood
        likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = likelihood(gp_model(X))  # Get predictive distribution for X
            log_likelihood = preds.log_prob(X)  # Compute log likelihood for each data point
            log_responsibilities[:, i] = torch.log(mixing_coeffs[i]) + log_likelihood  # Log of weighted likelihood
    
    # Normalize responsibilities using log-sum-exp to avoid numerical underflow/overflow
    log_sum_exp = torch.logsumexp(log_responsibilities, dim=1, keepdim=True)
    responsibilities = torch.exp(log_responsibilities - log_sum_exp)
    
    return responsibilities


# EM_algorithm
def EM_algorithm(models, sample_coords, sample_values, iter, inner_iter, init='topk', visualization_coords=None, early_delta=(False, 'mll', 0, 0, 0), debug=False):

    e_delta = early_delta[0] # early stopping trigger
    e_mode = early_delta[1]
    val_coords = early_delta[2]
    val_values = early_delta[3]
    lowest_val_loss_tot = 9999
    e_stop = False
    if e_delta:
        val_responsibilities = torch.full((val_coords.size(0), len(models)), 1.0/len(models))
    else:
        val_responsibilities = None

    percentage = 10
    p = round(iter/percentage)
    print_number = 0
    c = 0
    
    responsibilities = torch.full((sample_coords.size(0), len(models)), 1.0/len(models))
    
    figs = []
    val_figs = []

    for it in range(iter):
        # E-step - estimate responsibilities
        print(f'iter: {it}')

        #log_probs, _ = compute_log_probs_denorm(models, sample_coords, sample_values)        
        
        for model in models:
            model.set_train_data(sample_coords, sample_values, strict=False)

        if it == 0 and init == 'topk':
            print(f'Using topk init...')
            mask = k_percent_to_ones(sample_values, 10, largest=True)
            responsibilities[:, 0] = mask
            responsibilities[:, 1] = 1 - mask
            #for i, model in enumerate(models):
                #mean_init = sample_values[responsibilities[:, i]>0.9].sum()/responsibilities[:, i].sum()
                #model.mean_module.constant.data.fill_(mean_init)
        elif it == 0:
            log_probs, _ = compute_log_probs(models, sample_coords, sample_values, jitter=1e-9)
            responsibilities = torch.exp(compute_responsibilities(log_probs, normalize=True))
        else:
            log_probs, _ = compute_log_probs(models, sample_coords, sample_values, resps=None, jitter=1e-9)
            #log_probs, _ = compute_log_probs_denorm(models, sample_coords, sample_values)
            responsibilities = torch.exp(compute_responsibilities(log_probs, normalize=True))
            if responsibilities.isnan().sum() > 0:
                print(f'found {responsibilities.isnan().sum()} nans in responsibilities')
        
        responsibilities = responsibilities.clamp(1e-6, 1.0)

        # Then make a fig if this is the first iteration
        if visualization_coords is not None and it == 0:
            predictions = predict_during_training(visualization_coords, models)
            fig = training_fig_matplotlib(sample_coords, sample_values, visualization_coords, predictions, responsibilities)
            fig_text = f'MoE training iter: {it}/{iter} (inner_iters: {inner_iter})'
            #fig.update_layout(title={'text': fig_text,\
            #'y': 0.90, 'x': 0.46, 'xanchor': 'center', 'yanchor': 'top'})
            # Add a title to the matplotlib figure, adjusting the position
            fig.suptitle(fig_text, x=0.50, y=1.02, ha='center', va='top')
            #params0 = f'noise: {models[0].likelihood.noise.item():.4f}, cov_outscale: {models[0].covar_module.outputscale.item():.4f}, cov_lenscale: {models[0].covar_module.base_kernel.lengthscale.item():.4f}'
            #params1 = f'noise: {models[1].likelihood.noise.item():.4f}, cov_outscale: {models[1].covar_module.outputscale.item():.4f}, cov_lenscale: {models[1].covar_module.base_kernel.lengthscale.item():.4f}'
            #fig.layout.annotations[0].update(text=params0)
            #fig.layout.annotations[1].update(text=params1)
            figs.append(fig)

        # M-step - maximize likelihood given responsibilities
        mix_losses = []
        for i, model in enumerate(models):
            #print(f'M - model #{i}')
            model.train()
            model.likelihood.train()
            #new_sample_coords = sample_coords[responsibilities[:, i]>=0.5]
            #new_sample_values = sample_values[responsibilities[:, i]>=0.5]
            #model.set_train_data(new_sample_coords, new_sample_values, strict=False)
            #train_model(new_sample_coords, new_sample_values, model, inner_iter, lr=0.1, early_delta=(False, 'mll', 0 , 0, 0), debug=debug)
            #mix_loss = validate_model(model, new_sample_coords, new_sample_values, e_mode, debug=debug)

            train_model(sample_coords, sample_values, model, inner_iter, lr=0.1, early_delta=(False, 'mll', 0 , 0, 0), debug=debug, resps=responsibilities[:, i])            
            #mix_loss = validate_model(model, sample_coords, sample_values, e_mode, debug=debug)

            #model.save_mix_loss((it, mix_loss.item()))
            #mix_losses.append(mix_loss.item())
            
        # Validation, early stopping
        if e_delta:
            val_models = []
            for model in models:
                val_model = deepcopy(model)
                #val_model.set_train_data(val_coords, val_values, strict=False)
                val_models.append(val_model)
            
            val_log_probs, _ = compute_log_probs_denorm(models, val_coords, val_values)
            val_responsibilities = torch.exp(compute_responsibilities(val_log_probs))

            val_losses = []
            for i, val_model in enumerate(val_models):
                new_val_coords = val_coords[val_responsibilities[:, i]>=0.5]
                new_val_values = val_values[val_responsibilities[:, i]>=0.5]
                val_model.set_train_data(new_val_coords, new_val_values, strict=False)
                val_loss = validate_model(val_model, new_val_coords, new_val_values, e_mode, debug=debug)
                models[i].save_val_loss((it, val_loss.item()))
                val_losses.append(val_loss.item())
            
            # Decide if early stop
            if it>=10:
                val_loss_tot = val_losses[0] + val_losses[1]    
                if val_loss_tot < lowest_val_loss_tot:
                    lowest_val_loss_tot = val_loss_tot
                elif val_loss_tot >= lowest_val_loss_tot + e_delta:
                    e_stop = True

        # print status, make training and validation figures
        #c += 1
        c = p
        if c >= p:
            c = 0
            print_number += percentage
            print(f'...{print_number}%', end='')
            if visualization_coords is not None:
                for i, model in enumerate(models):
                    print(f'model {i}:')
                    model.print_named_parameters()
                    print('')

                predictions = predict_during_training(visualization_coords, models)
                fig = training_fig_matplotlib(sample_coords, sample_values, visualization_coords, predictions, responsibilities)
                fig_text = f'MoE training iteration: {it}/{iter}'
                #fig.update_layout(title={'text': fig_text,\
                #'y': 0.90, 'x': 0.46, 'xanchor': 'center', 'yanchor': 'top'})
                # Add a title to the matplotlib figure, adjusting the position
                fig.suptitle(fig_text, x=0.50, y=1.02, ha='center', va='top')
                #params0 = f'noise: {models[0].likelihood.noise.item():.4f}, cov_outscale: {models[0].covar_module.outputscale.item():.4f} cov_lenscale: {models[0].covar_module.base_kernel.lengthscale.item():.4f}'
                #params1 = f'noise: {models[1].likelihood.noise.item():.4f}, cov_outscale: {models[1].covar_module.outputscale.item():.4f} cov_lenscale: {models[1].covar_module.base_kernel.lengthscale.item():.4f}'
                #fig.layout.annotations[0].update(text=params0)
                #fig.layout.annotations[1].update(text=params1)
                figs.append(fig)
            
            if e_delta:
                # Make validation figure
                val_predictions = predict_during_training(val_coords, val_models)
                val_fig = training_fig(val_coords, val_values, val_coords, val_predictions, val_responsibilities)
                val_fig_text = f'Validation: Train_iter: {it}/{iter} - inner_iters: {inner_iter}'
                val_fig.update_layout(title={'text': val_fig_text,\
                'y': 1.00, 'x': 0.50, 'xanchor': 'center', 'yanchor': 'top'})
                val_figs.append(val_fig)
        
        if e_stop:
            print(f'')
            print(f'Early stopping at iteration {it} of {iter}:')
            print(f'Validation loss {val_loss_tot} is larger than lowest validation loss {lowest_val_loss_tot}')
            print(f'')
            break

    # Final computation of log_probs
#    for i, model in enumerate(models):
#        model.eval()
#        model.likelihood.eval()
#        with torch.no_grad():
#            preds = model.likelihood(model(sample_coords))
#            log_prob = torch.stack([preds.log_prob(y) for y in sample_values])
#            log_probs[:, i] = log_prob
    for model in models:
       model.set_train_data(sample_coords, sample_values, strict=False)
    #log_probs, _ = compute_log_probs_denorm(models, sample_coords, sample_values)   
    log_probs, _ = compute_log_probs(models, sample_coords, sample_values)
    responsibilities = torch.exp(compute_responsibilities(log_probs, normalize=True))

    return models, responsibilities, figs, val_figs

def predict_during_training(test_coords, models, gating=None, resps=None):
    # Predict sequentially with one model at a time
    means = []; lowers = []; uppers = []
    for i, model in enumerate(models):
        model.eval()
        likelihood = model.likelihood
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if gating is not None and resps is not None:
                # Barter Island weightings
                #resp_noises = (1/resps[:, i] - 1)**2
                #gating_noises = (1/gating[:, i] - 1)**2

                # Barter Island weightings 2
                #resp_noises = (1/resps[:, i]) - 1
                #gating_noises = (1/gating[:, i]) - 1
                
                # Tjøtta weightings
                resp_noises = model.likelihood.noise/resps[:, i]
                gating_noises = model.likelihood.noise/gating[:, i]
                if resp_noises.isnan().sum() > 0:
                    print(f'found {resp_noises.isnan().sum()} nans in resp_noises')
                if gating_noises.isnan().sum() > 0:
                    print(f'found {gating_noises.isnan().sum()} nans in gate_noises')
                observed_pred = likelihood(model(test_coords, add_training_noises=resp_noises, add_gating_noises=gating_noises))
            else:
                observed_pred = likelihood(model(test_coords)) # Generate test_coords along the longitude coords
                model.print_named_parameters()
                print(f'observed_pred.mean.min(): {observed_pred.mean.min()}')

            means.append(observed_pred.mean)
            lower, upper = observed_pred.confidence_region()
            lowers.append(lower); uppers.append(upper)
    
    return (means, lowers, uppers)

def predict_mixture(test_coords, models, gating, responsibilities=None):

    gating_stack = torch.clamp(torch.stack(gating, dim=1), 1e-6, 1.0) # limit to torch.inf for barter and 1.0 for Tjotta?
    if responsibilities is not None:
        responsibilities = torch.clamp(responsibilities, 1e-6, 1.0) # comment out for barter and keep for Tjotta
        means, lowers, uppers = predict_during_training(test_coords, models, gating_stack, responsibilities)
    else:
        means, lowers, uppers = predict_during_training(test_coords, models)

    means_stack = torch.stack(means, dim=1)
    
    # SHOULD FIND a better way to clamp p_of_z to [0, 1]
    
    #p_of_z = torch.clamp(gating_stack/torch.sum(gating_stack, dim=1, keepdim=True), 0.0, 1.0)
    p_of_z = gating_stack/torch.sum(gating_stack, dim=1, keepdim=True)

    # mixture_sigma sometimes becomes NAN because of sqrt(neg), fix this!
    #p_of_z = torch.exp(gating_stack - torch.logsumexp(gating_stack, dim=1, keepdim=True)) # softmax
    mixture_mean = torch.sum(p_of_z*means_stack, dim=1)
    
    variances = torch.square((torch.stack(uppers, dim=1) - means_stack).div(2.0))
    
    var_norm = torch.square(means_stack - mixture_mean.unsqueeze(1))
    
    mixture_variances = torch.sum((variances + var_norm)*p_of_z, dim=1)
    
    mixture_variances[mixture_variances<=0] = 0.0
    
    mixture_sigma = torch.sqrt(mixture_variances)
    
    lower = mixture_mean - mixture_sigma*2
    upper = mixture_mean + mixture_sigma*2

    return (mixture_mean, lower, upper, p_of_z)

def training_fig(train_x, train_y, test_x, predictions, responsibilities):
    if train_x.ndim == 2:
        fig = training_fig_3d(train_x, train_y, test_x, predictions, responsibilities)
    elif train_x.ndim == 1:
        means, lowers, uppers = predictions
        fig = make_subplots(rows=1, cols=len(means), shared_yaxes=True, subplot_titles=("Anomalies", "Background"))
        showlegend=True
        for i, (mean, lower, upper) in enumerate(zip(means, lowers, uppers)):
            fig.add_trace(go.Scatter(x=train_x, y=train_y, name='CH4 observations', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=i+1)
            fig.add_trace(go.Scatter(x=train_x[responsibilities[:, i]>=0.5], y=train_y[responsibilities[:, i]>=0.5], name='Active observations', marker_color='red', mode='markers', marker_symbol='circle', marker_size=3, showlegend=showlegend), row=1, col=i+1)
            fig.add_trace(go.Scatter(x=test_x, y=mean, name='CH4 predicted', marker_color='dodgerblue', showlegend=showlegend), row=1, col=i+1)
            fig.add_trace(go.Scatter(x=train_x, y=responsibilities[:, i]*5, name='Responsibility', marker_color='green', showlegend=showlegend), row=1, col=i+1)
            fig.add_trace(go.Scatter(x=test_x, y=upper, fill='none', marker_color='lightgrey', name='2 sigma', showlegend=showlegend), row=1, col=i+1)
            showlegend=False
            fig.add_trace(go.Scatter(x=test_x, y=lower, fill='tonexty', fillcolor='rgba(0, 0, 255, 0.2)', marker_color='lightgrey', showlegend=showlegend), row=1, col=i+1) # fill to previous trace
            fig.update_layout(yaxis_range=[0, 1.1*max(fig.data[0].y)])
    
    return fig

def training_fig_matplotlib(train_x, train_y, test_x, predictions, responsibilities):
    if train_x.ndim == 2:
        fig = training_fig_3d_matplotlib(train_x, train_y, test_x, predictions, responsibilities)
    elif train_x.ndim == 1:
        means, lowers, uppers = predictions

        # Create figure and axes for the subplots
        fig, axs = plt.subplots(1, len(means), figsize=(10, 5), sharey=True)

        # Plot settings
        subplot_titles = ["Anomalies", "Background"]
        showlegend = False

        # Loop through the means, lowers, and uppers to create each subplot
        for i, (mean, lower, upper) in enumerate(zip(means, lowers, uppers)):
            
            # Plot CH4 predicted
            axs[i].plot(test_x, mean, color='dodgerblue', label='CH$_4$ predicted')
            
            # Plot responsibility
            #axs[i].plot(train_x, responsibilities[:, i] * 5, color='green', label='Responsibility' if showlegend else "")
            
            # Plot 2 sigma upper limit
            axs[i].plot(test_x, upper, color='lightgrey', label='2 std. dev.')
            
            # Fill between the lower and upper limits (2 sigma)
            axs[i].fill_between(test_x, lower, upper, color='lightgrey')
            
            # Plot CH4 observations
            axs[i].scatter(train_x, train_y, color='black', s=2, label='CH$_4$ observations')
            
            # Plot active observations (where responsibilities >= 0.5)
            axs[i].scatter(train_x[responsibilities[:, i] >= 0.5], train_y[responsibilities[:, i] >= 0.5],
                        color='red', s=3, label='Active observations')
            
            # Add subplot title
            axs[i].set_title(subplot_titles[i])

            # Remove the legend after the first plot
            if showlegend:
                axs[i].legend(loc='upper right', bbox_to_anchor=(0.5, 0.8))
                showlegend = False

        # Collect all handles and labels for the common legend
        handles, labels = axs[0].get_legend_handles_labels()

        # Set a common legend for the entire figure
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=1, frameon=True)

        # Set y-axis limits to match the largest y-value
        max_y = max(train_y) * 1.05
        for ax in axs:
            ax.set_ylim(0, max_y)

        # Set common x and y labels
        fig.text(0.5, -0.01, 'Distance from west to east [m]', ha='center')
        fig.text(-0.01, 0.5, 'CH$_4$ [ppm]', va='center', rotation='vertical')

        # Adjust the layout
        fig.tight_layout()
    
    return fig

def training_fig_3d(train_x, train_y, test_x, predictions, responsibilities):
    means, lowers, uppers = predictions
    fig = make_subplots(rows=1, cols=len(means), shared_yaxes=True, subplot_titles=("Anomalies", "Background"), specs=[[{'type': 'scene'}, {'type': 'scene'}]])
    showlegend=True
    
    for i, (mean, lower, upper) in enumerate(zip(means, lowers, uppers)):
        if len(responsibilities):
            tr_x = train_x[responsibilities[:, i]>=0.5]
            tr_y = train_y[responsibilities[:, i]>=0.5]
        else:
            tr_x = train_x
            tr_y = train_y
        
        fig.add_trace(go.Scatter3d(x=train_x[:, 0], y=train_x[:, 1], z=train_y, name='CH4 observations', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=i+1)
        fig.add_trace(go.Scatter3d(x=tr_x[:, 0], y=tr_x[:, 1], z=tr_y, name='Active observations', marker_color='red', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=i+1)
        fig.add_trace(go.Scatter3d(x=test_x[:, 0], y=test_x[:, 1], z=mean, name='CH4 predicted', marker_color='dodgerblue', mode='markers', marker_size=1, showlegend=showlegend), row=1, col=i+1)
        showlegend=False

    return fig


def training_fig_3d_matplotlib(train_x, train_y, test_x, predictions, responsibilities):
    means, lowers, uppers = predictions
    
    # Create a figure with 1 row and len(means) columns for 3D subplots
    fig = plt.figure(figsize=(12, 6))
    showlegend = True
    
    # Loop through means, lowers, and uppers to create subplots
    for i, (mean, lower, upper) in enumerate(zip(means, lowers, uppers)):
        # Create a 3D subplot
        ax = fig.add_subplot(1, len(means), i+1, projection='3d')
        
        # Filter active observations based on responsibilities
        if len(responsibilities):
            tr_x = train_x[responsibilities[:, i] >= 0.5]
            tr_y = train_y[responsibilities[:, i] >= 0.5]
        else:
            tr_x = train_x
            tr_y = train_y
        
        # Plot CH4 predicted
        ax.scatter(test_x[:, 0], test_x[:, 1], mean, color='dodgerblue', s=10, label='CH4 predicted' if showlegend else "")
        
        # Plot CH4 observations
        ax.scatter(train_x[:, 0], train_x[:, 1], train_y, color='black', s=20, label='CH4 observations' if showlegend else "")
        
        # Plot active observations
        ax.scatter(tr_x[:, 0], tr_x[:, 1], tr_y, color='red', s=20, label='Active observations' if showlegend else "")
        
        # Set axis labels and title
        ax.set_xlabel('East [m]')
        ax.set_ylabel('North [m]')
        ax.set_zlabel('CH4 [ppm]')
        ax.set_title("Anomalies" if i == 0 else "Background")
        
        # Display legend only on the first subplot
        if showlegend:
            ax.legend(loc='upper left')
            showlegend = False

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    
    # Show the plot
    #plt.show()
    return fig

def val_training_fig(train_x, train_y, test_x, predictions, responsibilities):
    mean, lower, upper = predictions
    fig = make_subplots(rows=1, cols=1, subplot_titles=("Plot 1"))
    showlegend=True

    fig.add_trace(go.Scatter(x=train_x, y=train_y, name='CH4 measurements', marker_color='black', mode='markers', marker_size=2, showlegend=showlegend), row=1, col=1)
    fig.add_trace(go.Scatter(x=train_x[responsibilities[:, 1]>=0.5], y=train_y[responsibilities[:, 1]>=0.5], name='High var measurements', marker_color='red', mode='markers', marker_symbol='circle', marker_size=3, showlegend=showlegend), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_x, y=mean, name='CH4 predicted', marker_color='dodgerblue', showlegend=showlegend), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_x, y=upper, fill='none', marker_color='lightgrey', name='2 sigma', showlegend=showlegend), row=1, col=1)
    showlegend=False
    fig.add_trace(go.Scatter(x=test_x, y=lower, fill='tonexty', fillcolor='rgba(0, 0, 255, 0.2)', marker_color='lightgrey', showlegend=showlegend), row=1, col=1) # fill to previous trace
    fig.update_layout(yaxis_range=[0, 1.1*max(fig.data[0].y)])
    
    return fig

def k_percent_to_ones(tensor, k, largest=True):
    """
    Find a mask where the largest or smallest k% of values in the tensor is 1, and the rest is 0.

    Parameters:
    - tensor (torch.Tensor): A 1D tensor of length n.
    - k (float): Percentage of values to set to 1 (between 0 and 100).

    Returns:
    - torch.Tensor: A tensor of the same shape as the input, where the top k% of values are 1 and the rest are 0.
    """
    # Ensure k is between 0 and 100
    if k < 0 or k > 100:
        raise ValueError("k must be between 0 and 100.")
    
    # Calculate the number of values to keep
    n = tensor.numel()
    cut_k = int(torch.ceil(torch.tensor(n * k / 100.)))

    # Find the k% threshold value
    _, indices = torch.topk(tensor, cut_k, largest=largest)
    result_tensor = torch.zeros_like(tensor)
    result_tensor[indices] = torch.tensor(1.0)
    return result_tensor
'''
    threshold_value = torch.min(tensor[indices])

    # Create a new tensor where values above the threshold are 1, and the rest are 0
    if largest:
        result_tensor = torch.where(tensor >= threshold_value, torch.tensor(1.0), torch.tensor(0.0))
    else:
        result_tensor = torch.where(tensor <= threshold_value, torch.tensor(1.0), torch.tensor(0.0))

    return result_tensor
'''
def generate_density_dataframe(n, a, f, m, i):
    # Generate distances
    distances = np.linspace(0, 1, n)
    
    # Initialize density array
    density = np.zeros(n)
    
    # Calculate density
    for index, distance in enumerate(distances):
        # Always present density component
        random_offset = (random.random() - 0.5) * 2.0 * 5.0
        density_component = a[0] * np.sin(2 * np.pi * (f[0] + random_offset) * distance) + m[0]
        
        # Conditionally add second component based on intervals
        for start, end in i:
            if start <= distance <= end:
                random_offset = (random.random() - 0.5) * 2.0 * 5.0
                density_component += a[1] * np.sin(2 * np.pi * (f[1] + random_offset) * distance) + m[1]
                break
        
        density[index] = density_component
    
    # Create DataFrame
    df = pd.DataFrame({'Longitude_m': distances, 'CH4 [ppmv]': density})
    
    return df

def detect_clusters(coords, values, threshold=0, min_cluster_distance=0, debug=False):

    # Calculate the 95th percentile as the threshold
    if threshold == 0:
        threshold = torch.tensor(np.percentile(values, 95))

    # Identify regions of the array that are above the threshold
    is_above_threshold = values > threshold
    diff = torch.diff(is_above_threshold.int())
    starts = torch.where(diff > 0)[0] + 1  # +1 to point to start of region, not end of previous region
    ends = torch.where(diff < 0)[0] + 1  # +1 because diff shifts everything to the left

    # If array starts or ends with a peak, diff won't catch it
    if is_above_threshold[0]:
        starts = torch.cat((torch.tensor([0]), starts))
    if is_above_threshold[-1]:
        ends = torch.cat((ends, torch.tensor([len(values) - 1])))

    # Find highest peak in each region
    #peak_regions = [array[start:end] for start, end in zip(starts, ends)]
    peak_regions_indices = [list(range(start, end)) for start, end in zip(starts, ends)]
    peak_regions_indices = [sublist for sublist in peak_regions_indices if len(sublist) > 0] #remove empty sublists
    #print(f'detect_peak peak_regions: {peak_region_indices}')

    # Sort the peaks in descending order by their value
    #peaks = np.array(peak_indices)[np.argsort(-array[peak_indices])]
    if debug:
        print(f'Found {len(peak_regions_indices)} peak_regions')
    
    # This merging of regions is meant to merge regions separated by only a 
    # few samples/meters. Merging over distances should be handled later.
    if min_cluster_distance:
        found = True
        while found == True:
            found = False
            temp_regions_indices = []
            i = 0
            while i < len(peak_regions_indices[:-1]):
                if debug:
                    print(f'len(peak_regions_indices[:-1]) = {len(peak_regions_indices[:-1])}')
                start_cur_idx = peak_regions_indices[i][0]
                end_cur_idx = peak_regions_indices[i][-1] # not used later
                end_cur = coords[peak_regions_indices[i][-1]]
                start_next_idx = peak_regions_indices[i+1][0] # not used later
                start_next = coords[peak_regions_indices[i+1][0]]
                end_next_idx = peak_regions_indices[i+1][-1]
                distance = torch.linalg.vector_norm(end_cur - start_next)
                
                if debug:
                    print(f'i: {i}, start_cur_idx {start_cur_idx} end cur_idx {end_cur_idx} start_next_idx {start_next_idx} end_next_idx {end_next_idx}')
                    print(f'end cur {end_cur} start_next {start_next} distance: {distance}')

                
                merge = False
                if distance < min_cluster_distance:
                    merge = True
                    
                    for wp in coords[peak_regions_indices[i][-1]:peak_regions_indices[i+1][0]]:
                        dist_cur_wp = torch.linalg.vector_norm(wp-end_cur)
                        dist_next_wp = torch.linalg.vector_norm(wp-start_next)
                        if max(dist_cur_wp, dist_next_wp) >= min_cluster_distance:
                            merge = False
                            break
                    
                    if merge:
                        if debug:
                            print(f'Merging regions...')
                        temp_regions_indices.append(list(range(start_cur_idx, end_next_idx+1)))
                        found = True
                        if i == len(peak_regions_indices) - 3:
                            temp_regions_indices.append(peak_regions_indices[-1])
                            if debug:
                                print(f'found, appending peak_regions_indices[-1]')
                        i += 2
                
                if not merge:
                    temp_regions_indices.append(peak_regions_indices[i])
                    if i == len(peak_regions_indices) - 2:
                        temp_regions_indices.append(peak_regions_indices[-1])
                        if debug:
                            print(f'not found, appending peak_regions_indices[-1]')
                    i += 1

            if len(peak_regions_indices) > 1:
                peak_regions_indices = temp_regions_indices

    if debug:
        print(f'len(peak_regions_indices): {len(peak_regions_indices)}')
    
    anomaly_indices = []
    for region in peak_regions_indices:
        anomaly_indices += region
    
    anomaly_indices = torch.tensor(anomaly_indices)
    no_anomaly_indices = negative_subset(values, anomaly_indices)
    
    return (peak_regions_indices, anomaly_indices, no_anomaly_indices)


def negative_subset(original, subset):
    # Then add all indices without plume at the end
    original_sample_indices = torch.arange(len(original))
    
    combined = torch.cat((original_sample_indices, subset))
    uniques, counts = combined.unique(dim=0, return_counts=True)
    not_plume_indices = uniques[counts == 1]
    
    return not_plume_indices.tolist()


def oneD_to_twoD_coords(coords):
    if len(coords.shape) == 1:
        coords_ = coords.unsqueeze(1)
        coords_2d = torch.cat((coords_, torch.zeros_like(coords_)), dim=1)
        return coords_2d
    else:
        print(f'coords not 1D: {coords.shape}')
        return

def get_region_polygons(peak_regions_indices, coords, buffer=0.01, dbscan_eps=2.0):

    if len(coords.shape) > 2 or len(coords.shape) < 1:
        print(f'coords are not 1D or 2D: {coords.shape}')
        return
    
    if len(coords.shape) == 1:
        coords_2d = oneD_to_twoD_coords(coords)
    else:
        coords_2d = coords

    n_regions = len(peak_regions_indices)
    # Create a matrix to contain distances between regions
    reg_dist = np.zeros((n_regions, n_regions))
    # Then we convert regions into multipoints
    geom = []
    for region in peak_regions_indices:
        geom.append(MultiPoint(coords_2d[region]))
    
    #print(f'len(geom): {len(geom)} geom: {geom}')
    
    # And compute distances between all the regions
    for i, region1 in enumerate(geom):
        for j, region2 in enumerate(geom):
            reg_dist[i][j] = region1.distance(region2)

    #print(f'reg_dist: {reg_dist}')
    # Then we do the clustering
    db = DBSCAN(eps=dbscan_eps, min_samples=1, metric='precomputed').fit(reg_dist)
    labels = db.labels_
    #print(f'labels:{labels}')
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    #print(f'len(clusters): {len(clusters)} clusters: {clusters}')
    # It is time to make Multipoints of clusters of regions
    polygons = []
    for cluster in clusters:
        merge_regions = []
        for region_num in clusters[cluster]:
            merge_regions += peak_regions_indices[region_num]
        #self.polygons.append(MultiPoint(self.sample_coords[merge_regions]).convex_hull.buffer(2))
        polygon = MultiPoint(coords_2d[merge_regions]).convex_hull.buffer(buffer)
        x = polygon.exterior.xy[0]
        y = polygon.exterior.xy[1]
        points = np.vstack((x, y)).T
        polygons.append(Path(points))

    return polygons

def polygons_to_1d_tensors(polygons):
    polygon_x_coords = []
    for polygon in polygons:
        x_list = []

        p_ = polygon.to_polygons()[0]
        for x in p_:
            x_list.append(x[0])
        
        # Convert the list to a tensor and reshape it to be a column vector
        column_tensor = torch.tensor(x_list).unsqueeze(1)
        
        # Append the column tensor to the list
        polygon_x_coords.append(torch.unique(column_tensor).squeeze())
    
    return polygon_x_coords

def polygons_to_2d_tensors(polygons):
    polygon_coords = []
    for polygon in polygons:
        
        p_ = polygon.to_polygons()[0]
        # Convert the list to a tensor
        # Append the column tensor to the list
        polygon_coords.append(torch.from_numpy(p_))
    
    return polygon_coords

def rolling_weight(models, coords, window_length=1, method='BMA'):
    df = pd.DataFrame({'coords': coords})
    value_columns_base = 'value'

    preds = []
    probs = torch.full((len(coords), len(models)), 1.0/len(models))
    for i, model in enumerate(models):
        print(f'Computing rolling weights for model {i}')

        with torch.no_grad():
            preds.append(model.likelihood(model(coords)))

        value_column = value_columns_base + '_%d' % (i)
        df[value_column] = preds[-1].mean
        p = 0
        for j, row in df.iterrows():
            if (j+1) % int(len(df)/10) == 0:
                p += 10
                print(f'{p}%.. ', end='')

            lower_bound = row['coords'] - window_length / 2.0
            upper_bound = row['coords'] + window_length / 2.0
            filtered_df = df[(df['coords'] >= lower_bound) & (df['coords'] <= upper_bound)].reset_index()
            
            # Using a data window of length n:
            #lon_m = torch.tensor(filtered_df['coords'])
            ch4 = torch.tensor(filtered_df[value_column])

            if method == 'BMA':
                probs[j, i] = preds[-1].log_prob(ch4.unsqueeze(1)).mean()
            else:
                probs[j, i] = compute_exact_mll(preds[-1].covariance_matrix, ch4)
        
        print('') # \n
    resps = torch.exp(compute_responsibilities(probs, normalize=False))
    
    return (resps, probs, preds)

# How does log_prob vary if train data is set to values from within window only?

def variable_window_rolling_mll(df, method, column_to_roll, column_for_window, window_value, model0, model1):
    p_y_gvn_m0 = []
    p_y_gvn_m1 = []
    log_prob0 = []
    log_prob1 = []
    resps = []
    mmK0_mml = gpytorch.mlls.ExactMarginalLogLikelihood(model0.likelihood, model0)
    mmK1_mml = gpytorch.mlls.ExactMarginalLogLikelihood(model1.likelihood, model1)
    for i, row in df.iterrows():
        lower_bound = row[column_for_window] - window_value / 2.0
        upper_bound = row[column_for_window] + window_value / 2.0
        filtered_df = df[(df[column_for_window] >= lower_bound) & (df[column_for_window] <= upper_bound)]
        if method == 'BMA':        
            # Using a data window of length n:
            lon_m = torch.Tensor(filtered_df[column_for_window].values)
            ch4 = torch.Tensor(filtered_df[column_to_roll].values)
            # compute mll for both
            p_y_gvn_m0.append(mmK0_mml(model0(lon_m), ch4).exp().item())
            p_y_gvn_m1.append(mmK1_mml(model1(lon_m), ch4).exp().item())

            with torch.no_grad():
                preds0 = model0.likelihood(model0(lon_m))
                preds1 = model1.likelihood(model1(lon_m))
            
            log_prob0.append(preds0.log_prob(ch4))
            log_prob1.append(preds1.log_prob(ch4))
            
    return (pd.Series(p_y_gvn_m0), pd.Series(p_y_gvn_m1), pd.Series(log_prob0), pd.Series(log_prob1))

def plot_2d(smooth_coords, smooth_values, sample_coords, title='Untitled'):
    df_sns = pd.DataFrame({
        'x': smooth_coords.numpy()[:, 0],
        'y': smooth_coords.numpy()[:, 1],
        'value': smooth_values
    })

    df_path = pd.DataFrame({
        'x': sample_coords.numpy()[:, 0],
        'y': sample_coords.numpy()[:, 1]
    })

    ax = plt.subplots()
    ax = sns.scatterplot(data=df_sns, x='x', y='y', hue='value', palette='coolwarm', s=4, linewidth=0, hue_norm=(smooth_values.min(), smooth_values.max()))
    ax = sns.scatterplot(data=df_path, x='x', y='y', s=4, color='black')
    plt.title(title)
    plt.xlabel('East [m]')
    plt.ylabel('North [m]')

    return plt

def tensors_to_df(coords, values=None):
    
    if len(coords.size()) == 1:
        df = pd.DataFrame({
            'x': coords.numpy()
        })
    elif len(coords.size()) == 2:
        df = pd.DataFrame({
            'x': coords.numpy()[:, 0],
            'y': coords.numpy()[:, 1],
        })
    
    if values is not None:
        df['value'] = values.numpy()
    
    return df
    

def plot_2d_NLE(ax=None, cluster_tensors=None, region_tensors=None, sample_coords=None, sample_values=None, smooth_coords=None, smooth_values=None, hue_min=None, hue_max=None, title='Untitled', cmap='coolwarm'):
    
    if ax is None:
        ax = plt.subplots(1, 1)

    if hue_min is None or hue_max is None:
        if sample_values is not None:
            hue_min = sample_values.min()
            hue_max = sample_values.max()
        else:
            hue_min = smooth_values.min()
            hue_max = smooth_values.max()

    if smooth_coords is not None and smooth_values is not None:
        df_smooth = pd.DataFrame({
        'x': smooth_coords.numpy()[:, 0],
        'y': smooth_coords.numpy()[:, 1],
        'value': smooth_values
        })
        sns.scatterplot(data=df_smooth, x='x', y='y', hue='value', palette=cmap, marker='s', s=6, linewidth=0, hue_norm=(hue_min, hue_max), ax=ax)
        x = df_smooth['x'].values
        y = df_smooth['y'].values
        z = df_smooth['value'].values
        # Create a grid for the contour plot
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        Xi, Yi = np.meshgrid(xi, yi)
        #zi = plt.tricontourf(x, y, z, levels=100, cmap='viridis')

        # Create the contour plot
        #contour = ax.contourf(Xi, Yi, zi, levels=100)

    if cluster_tensors is not None:
        for i, cluster in enumerate(cluster_tensors):
            df_cluster = tensors_to_df(cluster)
            sns.lineplot(data=df_cluster, x='x', y='y', color='green', sort=False, estimator=None, ax=ax)
        
        #df_cluster = pd.DataFrame({
        #    'x': cluster_coords.numpy()[:, 0],
        #    'y': cluster_coords.numpy()[:, 1]
        #})
        #ax = sns.scatterplot(data=df_cluster, x='x', y='y', s=4, color='green')

    if region_tensors is not None:
        for i, region in enumerate(region_tensors):
            df_region = tensors_to_df(region)
            sns.lineplot(data=df_region, x='x', y='y', color='red', sort=False, estimator=None, ax=ax)

        #df_region = pd.DataFrame({
        #    'x': region_coords.numpy()[:, 0],
        #    'y': region_coords.numpy()[:, 1]
        #})
        #ax = sns.scatterplot(data=df_region, x='x', y='y', s=4, color='red')

    if sample_coords is not None:
        df_path = pd.DataFrame({
            'x': sample_coords.numpy()[:, 0],
            'y': sample_coords.numpy()[:, 1],
            'value': sample_values
        })
        if sample_values is not None:
            sns.scatterplot(data=df_path, x='x', y='y', hue='value', palette=cmap, s=6, linewidth=0, hue_norm=(hue_min, hue_max), ax=ax)
        else:
            sns.scatterplot(data=df_path, x='x', y='y', color='black', s=4, linewidth=0, ax=ax)
    
    ax.set_title(title)
    #ax.set_xlabel('East [m]')
    #ax.set_ylabel('North [m]')

    return plt, ax

def generate_smooth_coords(df, padding):
    # Generate prediction coordinates
    #padding = 15
    min_lon = df['Lon [m]'].min() - padding
    min_lat = df['Lat [m]'].min() - padding
    max_lon = df['Lon [m]'].max() + padding
    max_lat = df['Lat [m]'].max() + padding

    resolution = 2.0 # resolution in meters
    lons = torch.linspace(min_lon, max_lon, int((max_lon-min_lon)/resolution))
    lats = torch.linspace(min_lat, max_lat, int((max_lat-min_lat)/resolution))

    # Generate the meshgrid for longitude and latitude
    lon_grid, lat_grid = torch.meshgrid(lons, lats)

    # Flatten the grids and combine them into a single (n^2, 2) tensor
    smooth_coords = torch.stack([lon_grid.flatten(), lat_grid.flatten()], dim=1)
    
    return smooth_coords

def detect_clusters_wrapper(sample_coords, sample_values, threshold, min_cluster_dist, min_region_dist, debug=False):
    peak_regions_indices, anomaly_indices, no_anomaly_indices = detect_clusters(sample_coords, sample_values, threshold=threshold, min_cluster_distance=min_cluster_dist, debug=debug)
    clusters = get_region_polygons(peak_regions_indices, sample_coords, buffer = 5.0, dbscan_eps=2.0)
    regions = get_region_polygons(peak_regions_indices, sample_coords, buffer = 8.0, dbscan_eps=min_region_dist)
    sample_coords_subsets = [sample_coords[anomaly_indices], sample_coords[no_anomaly_indices]]
    sample_values_subsets = [sample_values[anomaly_indices], sample_values[no_anomaly_indices]]
    clusters_tensors = polygons_to_2d_tensors(clusters)
    regions_tensors = polygons_to_2d_tensors(regions)

    return anomaly_indices, no_anomaly_indices, sample_coords_subsets, sample_values_subsets, clusters, clusters_tensors, regions, regions_tensors