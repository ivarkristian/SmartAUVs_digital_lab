import torch
import gpytorch
import matplotlib.pyplot as plt
import os
from copy import deepcopy

def plot_prediction(x, y, pred, path=None, vmin=None, vmax=None, spacing=0, rot=0, RMSE=0):
    fig, ax = plt.subplots(figsize=(8, 6))
    if vmin == None:
        vmin = pred.min()
    if vmax == None:
        vmax = pred.max()

    scatter = ax.scatter(x, y, c=pred, cmap='coolwarm', s=1, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Value')

    if path is not None:
        ax.scatter(path[:, 0], path[:, 1], color='grey', alpha=0.5, s=1, label='Path')
    
    # Add labels and title
    ax.set_xlabel('Easting [m]')
    ax.set_ylabel('Northing [m]')
    ax.set_title(f'Prediction from spacing {spacing}, rotation {rot}. RMSE: {RMSE:.3}')
    
    #plt.close(fig)
    return fig

def normalize_tensor(tensor, range_min=0, range_max=1):
    min_val = tensor.min()
    max_val = tensor.max()
    
    if min_val == max_val:
        # Avoid division by zero; set to range_min or another constant value
        return torch.full_like(tensor, fill_value=range_min)
    else:
        # Normalize to [range_min, range_max]
        normalized_tensor = range_min + ((tensor - min_val) * (range_max - range_min)) / (max_val - min_val)
        return normalized_tensor


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
