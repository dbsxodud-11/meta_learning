import torch
from torch import nn
from torch.distributions.kl import kl_divergence


def calculate_loss(p_y_pred, y_target, q_target, q_context):
        """
        Computes Neural Process loss.
        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.
        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)
        q_target : one of torch.distributions.Distribution
            Latent distribution for target points.
        q_context : one of torch.distributions.Distribution
            Latent distribution for context points.
        """
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
        # KL has shape (batch_size, r_dim). Take mean over batch and sum over
        # r_dim (since r_dim is dimension of normal distribution)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return -log_likelihood + kl