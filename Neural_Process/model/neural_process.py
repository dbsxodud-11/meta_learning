import torch
from torch import nn
from torch.distributions import Normal

from .models import Encoder, MuSigmaEncoder, Decoder

class NeuralProcess(nn.Module):
    """
    Implements Neural Process for functions of arbitrary dimensions.
    Parameters
    ----------
    x_dim : int
        Dimension of x values.
    y_dim : int
        Dimension of y values.
    r_dim : int
        Dimension of output representation r.
    z_dim : int
        Dimension of latent variable z.
    h_dim : int
        Dimension of hidden layer in encoder and decoder.
    """
    def __init__(self, x_dim, y_dim, r_dim, z_dim, h_dim):
        super(NeuralProcess, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        # Initialize networks
        self.xy_to_r = Encoder(x_dim, y_dim, h_dim, r_dim)
        self.r_to_mu_sigma = MuSigmaEncoder(r_dim, z_dim)
        self.xz_to_y = Decoder(x_dim, z_dim, h_dim, y_dim)

    def aggregate(self, r_i):
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation.
        Parameters
        ----------
        r_i : torch.Tensor
            Shape (batch_size, num_points, r_dim)
        """
        return torch.mean(r_i, dim=1)

    def xy_to_mu_sigma(self, x, y):
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)
        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        """
        batch_size, num_points, _ = x.size()
        # Flatten tensors, as encoder expects one dimensional inputs
        x_flat = x.view(batch_size * num_points, self.x_dim)
        y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        # Encode each point into a representation r_i
        r_i_flat = self.xy_to_r(x_flat, y_flat)
        # Reshape tensors into batches
        r_i = r_i_flat.view(batch_size, num_points, self.r_dim)
        # Aggregate representations r_i into a single representation r
        r = self.aggregate(r_i)
        # Return parameters of distribution
        return self.r_to_mu_sigma(r)

    def forward(self, x_context, y_context, x_target, y_target=None, return_std=False):
        """
        Given context pairs (x_context, y_context) and target points x_target,
        returns a distribution over target points y_target.
        Parameters
        ----------
        x_context : torch.Tensor
            Shape (batch_size, num_context, x_dim). Note that x_context is a
            subset of x_target.
        y_context : torch.Tensor
            Shape (batch_size, num_context, y_dim)
        x_target : torch.Tensor
            Shape (batch_size, num_target, x_dim)
        y_target : torch.Tensor or None
            Shape (batch_size, num_target, y_dim). Only used during training.
        Note
        ----
        We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """
        # Infer quantities from tensor dimensions
        batch_size, num_context, x_dim = x_context.size()
        _, num_target, _ = x_target.size()
        _, _, y_dim = y_context.size()

        if y_target is not None:
            # Encode target and context (context needs to be encoded to
            # calculate kl term)
            mu_target, sigma_target = self.xy_to_mu_sigma(x_target, y_target)
            mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context)
            # Sample from encoded distribution using reparameterization trick
            q_target = Normal(mu_target, sigma_target)
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_target.rsample()
            # Get parameters of output distribution
            y_pred_mu, y_pred_sigma = self.xz_to_y(x_target, z_sample)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            return p_y_pred, q_target, q_context
        else:
            # At testing time, encode only context
            mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context)
            # Sample from distribution based on context
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_context.rsample()
            # Predict target points based on context
            y_pred_mu, y_pred_sigma = self.xz_to_y(x_target, z_sample)
            # p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            # return p_y_pred
            return y_pred_mu, y_pred_sigma
    
    def __str__(self):
        return "Independent NP"