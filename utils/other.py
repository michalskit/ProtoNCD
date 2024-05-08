import torch
import numpy as np


def validate_difference(original_tensor, noisy_tensor, expected_diff=0.1):
    """Validates if the noisy tensor differs by approximately `expected_diff`*100% from the original tensor,
    and provides additional measures of difference."""
    relative_differences = []
    mad_differences = []
    cosine_similarities = []
    euclidean_distances = []
    
    for original_row, noisy_row in zip(original_tensor.view(original_tensor.size(0), -1), noisy_tensor.view(noisy_tensor.size(0), -1)):
        # Calculate the norm of the difference
        diff_norm = (original_row - noisy_row).norm()
        # Calculate the norm of the original row
        original_norm = original_row.norm()
        # Calculate the relative difference
        relative_difference = diff_norm / original_norm if original_norm > 0 else torch.tensor(float('nan'))
        relative_differences.append(relative_difference.item())

        # Calculate Mean Absolute Difference (MAD)
        mad = torch.abs(original_row - noisy_row).mean().item()
        mad_differences.append(mad)

        # Calculate Cosine Similarity
        cos_sim = torch.nn.functional.cosine_similarity(original_row, noisy_row, dim=0).item()
        cosine_similarities.append(cos_sim)

        # Calculate Euclidean Distance
        euclidean_distance = torch.dist(original_row, noisy_row, p=2).item()
        euclidean_distances.append(euclidean_distance)
    
    # Compute averages and other statistics
    avg_relative_difference = np.mean(relative_differences)
    max_relative_difference = np.max(relative_differences)
    std_dev_relative_difference = np.std(relative_differences)
    avg_mad = np.mean(mad_differences)
    avg_cosine_similarity = np.mean(cosine_similarities)
    avg_euclidean_distance = np.mean(euclidean_distances)
    is_approx_10_percent_diff = abs(avg_relative_difference - expected_diff) <= expected_diff * 1.2

    print(f"Average relative difference: {avg_relative_difference*100:.2f}%")
    print(f"Maximum relative difference: {max_relative_difference*100:.2f}%")
    print(f"Standard deviation of relative differences: {std_dev_relative_difference*100:.2f}%")
    print(f"Average Mean Absolute Difference (MAD): {avg_mad}")
    print(f"Average cosine similarity: {avg_cosine_similarity}")
    print(f"Average Euclidean distance: {avg_euclidean_distance}")
    print("Is approximately 10% different:", is_approx_10_percent_diff)