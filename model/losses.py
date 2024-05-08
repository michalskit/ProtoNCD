import torch
from torch import nn
import math
import torch.nn.functional as F
import os


class SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.05):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.iter = 0

    @torch.no_grad()
    def forward(self, logits):
        Q = torch.exp(logits / self.epsilon).t()
        B = Q.shape[1]
        K = Q.shape[0]  # how many prototypes
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.num_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()


class SinkhornKnopp_2(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.05):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.iter = 0

    @torch.no_grad()
    def forward(self, logits):

        # Subtract the maximum logit value for numerical stability
        logits = logits - torch.max(logits)

        Q = torch.exp(logits / self.epsilon).t()
        B = Q.shape[1]
        K = Q.shape[0]  # how many prototypes
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.num_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

class StableSinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.05, stability_eps=1e-9):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.stability_eps = stability_eps  # Small constant to prevent division by zero

    def check_for_nans_and_infs(self, tensor, message):
        if torch.isnan(tensor).any():
            print(f"NaN detected at: {message}")
        if torch.isinf(tensor).any():
            print(f"Inf detected at: {message}")

    @torch.no_grad()
    def forward(self, logits):
        # Check for NaNs or infs in input logits
        self.check_for_nans_and_infs(logits, "input logits")

        # Subtract the maximum logit value for numerical stability
        logits = logits - torch.max(logits)
        
        Q = torch.exp(logits / self.epsilon).t()
        self.check_for_nans_and_infs(Q, "after initial exponentiation")

        B = Q.shape[1]
        K = Q.shape[0]  # how many prototypes

        # Add a small constant to prevent division by zero or very small numbers
        sum_Q = torch.sum(Q) + self.stability_eps
        Q /= sum_Q
        self.check_for_nans_and_infs(Q, "after normalizing Q with sum_Q")

        for it in range(self.num_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True) + self.stability_eps
            Q /= sum_of_rows
            Q /= K
            self.check_for_nans_and_infs(Q, f"after normalizing rows, iteration {it}")

            # normalize each column: total weight per sample must be 1/B
            sum_of_columns = torch.sum(Q, dim=0, keepdim=True) + self.stability_eps
            Q /= sum_of_columns
            Q /= B
            self.check_for_nans_and_infs(Q, f"after normalizing columns, iteration {it}")

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        self.check_for_nans_and_infs(Q, "final result")

        return Q.t()



def KD(args, origin_logits, new_logits, mask_lab, T=0.15):
    nlc = args.num_labeled_classes
    origin_logits, new_logits = origin_logits / args.temperature, new_logits / args.temperature
    origin_logits = origin_logits.detach()

    # the predictions of pretrained model on unlabeled data
    # we want to destilate from this model
    preds = F.softmax(origin_logits[:, :, ~mask_lab] / T, dim=-1)

    # predictions of new model on unlabeled data
    pseudo_logits = new_logits[:, :, ~mask_lab]
    pseudo_preds = F.softmax(pseudo_logits[:, :, :, :nlc] / T, dim=-1)

    # gate function to control the weight of kd loss.
    pseudo_preds_all = F.softmax(pseudo_logits / T, dim=-1)
    weight = torch.sum(pseudo_preds_all[:, :, :, :nlc], dim=-1, keepdim=True)
    
    weight = weight / torch.mean(weight)

    if args.eta_equals_one:
        weight = torch.ones_like(weight)

    loss_unseen_kd = torch.mean(torch.sum(-torch.log(pseudo_preds) * preds * (T**2)* weight, dim=-1), dim=[0, 2])

    return loss_unseen_kd