import numpy as np


def grad_finite_diff(function, w, eps=1e-8):
    eye = np.eye(len(w))

    def compute_perturbed_grad(w_i):
        perturbed_w = w + eps * w_i
        return (function(perturbed_w) - function(w)) / eps

    result = np.apply_along_axis(compute_perturbed_grad, axis=1, arr=eye)
    return result
