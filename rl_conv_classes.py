import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gpytorch
import numpy as np

from gpt_class_exactgpmodel import ExactGPModel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# %%
class ScalarFieldEnv:
    def __init__(self, gp_mean, gp_variance, action_range=(-1, 1), patch_size=5):
        self.gp_mean = gp_mean
        self.gp_variance = gp_variance
        self.grid_size = gp_mean.shape  # (n, n)
        self.action_range = action_range
        self.patch_size = patch_size
        self.position = np.random.randint(0, self.grid_size[0], size=2)  # Random initial position
        
        # Initialize GP model
        kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        self.gp_model = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, normalize_y=True)
        self.observations = []  # Stores observed points
        self.values = []  # Stores observed scalar field values
    
    def get_local_patch(self, pos):
        x, y = pos
        half_patch = self.patch_size // 2
        x_min, x_max = max(0, x - half_patch), min(self.grid_size[0], x + half_patch + 1)
        y_min, y_max = max(0, y - half_patch), min(self.grid_size[1], y + half_patch + 1)
        local_mean = self.gp_mean[x_min:x_max, y_min:y_max]
        local_variance = self.gp_variance[x_min:x_max, y_min:y_max]
        return local_mean, local_variance
    
    def step(self, action):
        # Apply the action (dx, dy)
        dx, dy = action
        self.position[0] = np.clip(self.position[0] + dx, 0, self.grid_size[0] - 1)
        self.position[1] = np.clip(self.position[1] + dy, 0, self.grid_size[1] - 1)
        
        # Sample the scalar field at the new position
        x, y = int(self.position[0]), int(self.position[1])
        sampled_value = self.gp_mean[x, y]  # True value at the position (can add noise if desired)
        self.observations.append(self.position.copy())
        self.values.append(sampled_value)
        
        # Update the GP posterior with new data
        self.update_gp_posterior()
        
        # Compute reward
        reward = 0.5 * self.gp_variance[x, y] + 0.5 * sampled_value
        
        # Get next state (local patch + position)
        local_mean, local_variance = self.get_local_patch(self.position)
        next_state = {
            "position": self.position.copy(),
            "local_mean": local_mean,
            "local_variance": local_variance
        }
        
        done = False  # Define termination conditions if any
        return next_state, reward, done
    
    def update_gp_posterior(self):
        if len(self.observations) > 1:  # Update only if there is enough data
            obs_array = np.array(self.observations)
            val_array = np.array(self.values)
            self.gp_model.fit(obs_array, val_array)
            
            # Predict mean and variance over the entire grid
            grid_points = np.array([[i, j] for i in range(self.grid_size[0]) for j in range(self.grid_size[1])])
            gp_pred_mean, gp_pred_var = self.gp_model.predict(grid_points, return_std=True)
            self.gp_mean = gp_pred_mean.reshape(self.grid_size)
            self.gp_variance = gp_pred_var.reshape(self.grid_size)
    
    def reset(self):
        self.position = np.random.randint(0, self.grid_size[0], size=2)
        local_mean, local_variance = self.get_local_patch(self.position)
        return {
            "position": self.position.copy(),
            "local_mean": local_mean,
            "local_variance": local_variance
        }

# %%
class CNNPolicyNetwork(nn.Module):
    def __init__(self, grid_size, action_dim):
        super(CNNPolicyNetwork, self).__init__()
        self.grid_size = grid_size
        self.action_dim = action_dim

        # Convolutional layers for feature extraction
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2
        )

        # Fully connected layers for action prediction
        conv_output_size = (grid_size // 4) * (grid_size // 4) * 128  # After two MaxPool layers
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size + 2, 256),  # Include (x, y) position
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Separate heads for mean and standard deviation
        self.mean_layer = nn.Linear(128, action_dim)
        self.std_layer = nn.Linear(128, action_dim)

    def forward(self, gp_mean, gp_variance, position):
        """
        Args:
            gp_mean: Tensor of shape (batch_size, grid_size, grid_size) - GP posterior mean
            gp_variance: Tensor of shape (batch_size, grid_size, grid_size) - GP posterior variance
            position: Tensor of shape (batch_size, 2) - Agent's position (x, y)
        Returns:
            mean: Mean of the Gaussian policy for actions
            std: Standard deviation of the Gaussian policy for actions
        """
        # Combine GP mean and variance into a single tensor
        gp_input = torch.stack([gp_mean, gp_variance], dim=1)  # Shape: (batch_size, 2, grid_size, grid_size)

        # Extract features using the CNN
        conv_features = self.conv(gp_input)  # Shape: (batch_size, 128, grid_size//4, grid_size//4)
        conv_features = conv_features.view(conv_features.size(0), -1)  # Flatten

        # Concatenate agent's position to the flattened features
        features = torch.cat([conv_features, position], dim=1)  # Shape: (batch_size, conv_output_size + 2)

        # Fully connected layers
        x = self.fc(features)

        # Output mean and std
        mean = self.mean_layer(x)
        std = torch.clamp(self.std_layer(x), min=1e-3, max=1.0)  # Ensure std > 0
        return mean, std
    
class PPOAgent:
    def __init__(self, grid_size, action_dim, env_xy, speed, sampling_freq, learning_rate=1e-3, gamma=0.99, gp_lengthscale_constraint=gpytorch.constraints.Positive(), nn_filename=None):
        self.env_xy = env_xy
        self.speed = speed
        self.sampling_freq = sampling_freq
        self.length_constraint = gp_lengthscale_constraint
        self.kernel_name = 'scale_rbf'
        self.llh = gpytorch.likelihoods.GaussianLikelihood()
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.nn_filename = nn_filename
        
        if self.nn_filename:
            self.load_model(self.nn_filename)

        self.policy_net = CNNPolicyNetwork(grid_size, action_dim)
        self.value_net = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(grid_size * grid_size * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=1e-3)

        self.reset_samples()

    def reset_samples(self):
        self.mdl = ExactGPModel(torch.tensor([]), torch.tensor([]), self.llh, self.kernel_name, lengthscale_constraint=self.length_constraint)

    def get_action(self, gp_mean, gp_variance, position):
        mean, std = self.policy_net(gp_mean, gp_variance, position)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def estimate_env(self, env_xy, sampled_coords, sampled_vals):
        if self.mdl is None:
            self.mdl = ExactGPModel(sampled_coords, sampled_vals, self.llh, self.kernel_name, lengthscale_constraint=self.length_constraint)
        
        self.mdl.set_train_data(sampled_coords, sampled_vals, strict=False)
        
        # Then predict
        self.mdl.eval()
        self.mdl.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            current_pred = self.mdl.likelihood(self.mdl(env_xy))
    
        return current_pred

    def compute_loss(self, gp_mean, gp_variance, position, actions, rewards, old_log_probs, advantages):
        # Policy loss
        mean, std = self.policy_net(gp_mean, gp_variance, position)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        ratio = torch.exp(log_probs - old_log_probs)
        policy_loss = -torch.min(
            ratio * advantages,
            torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
        ).mean()

        # Value loss
        values = self.value_net(torch.stack([gp_mean, gp_variance], dim=1)).squeeze(-1)
        value_loss = nn.MSELoss()(values, rewards)

        return policy_loss + 0.5 * value_loss
    
    def save_model(self, filename=None):
        """Save the model weights and optimizer state to a file."""
        if filename is None:
            filename = 'GPConvAgent.nn'
        
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon
        }, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename=None):
        """Load model weights and optimizer state from a file."""
        try:
            checkpoint = torch.load(filename, weights_only=True)
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.learning_rate = checkpoint.get('learning_rate', self.learning_rate)
            self.gamma = checkpoint.get('gamma', self.gamma)
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            print(f"Model loaded from {filename}")
        except FileNotFoundError:
            print(f"No saved model found at {filename}, starting fresh.")
        except Exception as e:
            print(f'Error loading model: {e}')

