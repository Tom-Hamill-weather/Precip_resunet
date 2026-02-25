"""
Gamma Mixture Model - EM Algorithm
Converted from R's mixtools::gammamixEM to Python

Reference:
- mixtools R package: https://cran.r-project.org/package=mixtools
- Benaglia et al. (2009) "mixtools: An R Package for Analyzing Finite Mixture Models"

This implementation follows the same algorithm as R's gammamixEM for consistency
with precipitation postprocessing literature.

Author: Tom Hamill with Claude Code assistance
Date: February 2026
"""

import numpy as np
from scipy.stats import gamma
from scipy.optimize import minimize
from scipy.special import digamma, polygamma
import warnings

class GammaMixtureEM:
    """
    Fit a mixture of Gamma distributions using the EM algorithm.

    Equivalent to R's mixtools::gammamixEM function.

    Parameters
    ----------
    n_components : int
        Number of mixture components (k in R version)
    max_iter : int
        Maximum number of EM iterations
    tol : float
        Convergence tolerance (epsilon in R version)
    init_method : str
        'moments' (default, like R's mom.start=TRUE) or 'random' or 'quantiles'
    random_state : int or None
        Random seed for reproducibility
    verbose : bool
        Print iteration progress

    Attributes
    ----------
    weights_ : array of shape (n_components,)
        Mixing proportions (lambda in R)
    shapes_ : array of shape (n_components,)
        Shape parameters (alpha in R)
    scales_ : array of shape (n_components,)
        Scale parameters (beta in R)
    posterior_ : array of shape (n_samples, n_components)
        Posterior probabilities (like R's posterior)
    loglik_ : float
        Final log-likelihood
    n_iter_ : int
        Number of iterations until convergence
    converged_ : bool
        Whether algorithm converged
    """

    def __init__(self, n_components=2, max_iter=1000, tol=1e-8,
                 init_method='moments', random_state=None, verbose=False):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.init_method = init_method
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, weights_init=None, shapes_init=None, scales_init=None):
        """
        Fit the Gamma mixture model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Training data (should be positive values)
        weights_init : array of shape (n_components,), optional
            Initial mixing proportions
        shapes_init : array of shape (n_components,), optional
            Initial shape parameters
        scales_init : array of shape (n_components,), optional
            Initial scale parameters

        Returns
        -------
        self : object
            Fitted model
        """
        X = np.asarray(X).ravel()

        # Remove non-positive values
        X = X[X > 0]
        if len(X) == 0:
            raise ValueError("No positive values in data")

        n_samples = len(X)
        k = self.n_components

        # Initialize parameters
        if weights_init is None or shapes_init is None or scales_init is None:
            weights, shapes, scales = self._initialize_parameters(X)
        else:
            weights = np.asarray(weights_init)
            shapes = np.asarray(shapes_init)
            scales = np.asarray(scales_init)

        # Normalize weights
        weights = weights / weights.sum()

        # EM iterations
        log_likelihood_old = -np.inf

        for iteration in range(self.max_iter):
            # E-step: Compute responsibilities (posterior probabilities)
            responsibilities = self._e_step(X, weights, shapes, scales)

            # M-step: Update parameters
            weights = self._m_step_weights(responsibilities)
            shapes, scales = self._m_step_gamma_params(X, responsibilities)

            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(X, weights, shapes, scales)

            if self.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: log-likelihood = {log_likelihood:.6f}")

            # Check convergence
            if log_likelihood - log_likelihood_old < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                self.converged_ = True
                break

            log_likelihood_old = log_likelihood
        else:
            warnings.warn(f"EM did not converge after {self.max_iter} iterations")
            self.converged_ = False

        # Store results
        self.weights_ = weights
        self.shapes_ = shapes
        self.scales_ = scales
        self.posterior_ = responsibilities
        self.loglik_ = log_likelihood
        self.n_iter_ = iteration + 1

        return self

    def _initialize_parameters(self, X):
        """
        Initialize parameters using method of moments on data partitions.

        Equivalent to R's mom.start=TRUE option.
        """
        k = self.n_components
        n = len(X)

        if self.init_method == 'moments':
            # Partition data into k quantile-based regions
            # Fit separate Gamma to each region (method of moments)
            quantiles = np.linspace(0, 100, k + 1)
            percentiles = np.percentile(X, quantiles)

            shapes = np.zeros(k)
            scales = np.zeros(k)

            for j in range(k):
                # Data in j-th partition
                if j == 0:
                    mask = X <= percentiles[j + 1]
                elif j == k - 1:
                    mask = X > percentiles[j]
                else:
                    mask = (X > percentiles[j]) & (X <= percentiles[j + 1])

                X_j = X[mask]
                if len(X_j) > 1:
                    # Method of moments: shape = mean^2 / var, scale = var / mean
                    mean_j = np.mean(X_j)
                    var_j = np.var(X_j)
                    if var_j > 0:
                        shapes[j] = mean_j ** 2 / var_j
                        scales[j] = var_j / mean_j
                    else:
                        shapes[j] = 1.0
                        scales[j] = mean_j
                else:
                    shapes[j] = 1.0
                    scales[j] = np.mean(X)

            # Ensure reasonable values
            shapes = np.clip(shapes, 0.1, 100)
            scales = np.clip(scales, 0.01, 100)

        elif self.init_method == 'quantiles':
            # Initialize with quantiles
            shapes = np.ones(k)
            scales = np.percentile(X, np.linspace(20, 80, k))

        elif self.init_method == 'random':
            # Random initialization
            rng = np.random.RandomState(self.random_state)
            shapes = rng.uniform(0.5, 5, k)
            scales = rng.uniform(0.1, np.mean(X) * 2, k)

        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")

        # Equal weights initially
        weights = np.ones(k) / k

        return weights, shapes, scales

    def _e_step(self, X, weights, shapes, scales):
        """
        E-step: Compute responsibilities (posterior probabilities).

        Returns array of shape (n_samples, n_components)
        """
        n_samples = len(X)
        k = self.n_components

        responsibilities = np.zeros((n_samples, k))

        for j in range(k):
            # Compute weighted PDF for component j
            responsibilities[:, j] = weights[j] * gamma.pdf(X, a=shapes[j], scale=scales[j])

        # Normalize to get posterior probabilities
        total = responsibilities.sum(axis=1, keepdims=True)
        total = np.where(total > 0, total, 1e-10)  # Avoid division by zero
        responsibilities /= total

        return responsibilities

    def _m_step_weights(self, responsibilities):
        """
        M-step: Update mixing proportions.

        Simple average of responsibilities.
        """
        return responsibilities.mean(axis=0)

    def _m_step_gamma_params(self, X, responsibilities):
        """
        M-step: Update shape and scale parameters for each component.

        Uses weighted MLE via numerical optimization (no closed form for Gamma).
        """
        k = self.n_components
        shapes = np.zeros(k)
        scales = np.zeros(k)

        for j in range(k):
            shapes[j], scales[j] = self._fit_gamma_weighted(X, responsibilities[:, j])

        return shapes, scales

    def _fit_gamma_weighted(self, X, weights):
        """
        Fit Gamma distribution with weighted MLE.

        Uses numerical optimization since no closed-form solution exists.
        """
        # Normalize weights
        weights = weights / weights.sum()

        # Weighted sufficient statistics
        log_x = np.log(X)
        mean_x = np.average(X, weights=weights)
        mean_log_x = np.average(log_x, weights=weights)

        # Initial guess using method of moments
        s = np.log(mean_x) - mean_log_x
        shape_init = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
        if not np.isfinite(shape_init) or shape_init <= 0:
            shape_init = 1.0
        scale_init = mean_x / shape_init

        # Negative weighted log-likelihood
        def neg_log_lik(params):
            shape, scale = params
            if shape <= 0 or scale <= 0:
                return 1e10
            try:
                log_pdf = gamma.logpdf(X, a=shape, scale=scale)
                return -np.sum(weights * log_pdf)
            except:
                return 1e10

        # Optimize
        result = minimize(
            neg_log_lik,
            x0=[shape_init, scale_init],
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8}
        )

        if result.success:
            shape, scale = result.x
        else:
            # Fall back to method of moments
            var_x = np.average((X - mean_x) ** 2, weights=weights)
            shape = mean_x ** 2 / var_x
            scale = var_x / mean_x

        # Ensure reasonable bounds
        shape = np.clip(shape, 0.1, 100)
        scale = np.clip(scale, 0.01, 100)

        return shape, scale

    def _compute_log_likelihood(self, X, weights, shapes, scales):
        """
        Compute the observed data log-likelihood.
        """
        n_samples = len(X)
        k = self.n_components

        log_prob = np.zeros(n_samples)

        for j in range(k):
            log_prob += weights[j] * gamma.pdf(X, a=shapes[j], scale=scales[j])

        return np.sum(np.log(log_prob + 1e-10))

    def predict_proba(self, X):
        """
        Predict posterior probabilities for new data.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Data to predict

        Returns
        -------
        proba : array of shape (n_samples, n_components)
            Posterior probabilities
        """
        X = np.asarray(X).ravel()
        return self._e_step(X, self.weights_, self.shapes_, self.scales_)

    def predict(self, X):
        """
        Predict component labels for new data.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Data to predict

        Returns
        -------
        labels : array of shape (n_samples,)
            Component labels (0 to n_components-1)
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X):
        """
        Compute log-likelihood of data.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Data to score

        Returns
        -------
        log_likelihood : float
        """
        X = np.asarray(X).ravel()
        X = X[X > 0]
        return self._compute_log_likelihood(X, self.weights_, self.shapes_, self.scales_)


def fit_gamma_mixture(data, n_components=2, **kwargs):
    """
    Convenience function to fit Gamma mixture model.

    Parameters
    ----------
    data : array-like
        Positive-valued data (e.g., precipitation amounts > 0)
    n_components : int
        Number of mixture components
    **kwargs : dict
        Additional arguments passed to GammaMixtureEM

    Returns
    -------
    weights : array
        Mixing proportions
    shapes : array
        Shape parameters (alpha)
    scales : array
        Scale parameters (theta)
    model : GammaMixtureEM
        Fitted model object

    Example
    -------
    >>> wet_mrms = mrms_data[mrms_data > 0]
    >>> weights, shapes, scales, model = fit_gamma_mixture(wet_mrms, n_components=2)
    >>> print(f"Component 1: shape={shapes[0]:.2f}, scale={scales[0]:.2f}, weight={weights[0]:.2f}")
    >>> print(f"Component 2: shape={shapes[1]:.2f}, scale={scales[1]:.2f}, weight={weights[1]:.2f}")
    """
    model = GammaMixtureEM(n_components=n_components, **kwargs)
    model.fit(data)

    return model.weights_, model.shapes_, model.scales_, model


if __name__ == '__main__':
    # Example usage
    print("Testing Gamma Mixture EM implementation")
    print("=" * 60)

    # Generate synthetic data: mixture of two Gammas
    np.random.seed(42)
    n1, n2 = 500, 300

    # Component 1: light rain (shape=2, scale=1, mean=2)
    X1 = np.random.gamma(shape=2, scale=1, size=n1)

    # Component 2: heavy rain (shape=3, scale=3, mean=9)
    X2 = np.random.gamma(shape=3, scale=3, size=n2)

    # Mix them
    X = np.concatenate([X1, X2])
    true_weights = np.array([n1 / (n1 + n2), n2 / (n1 + n2)])

    print(f"\nTrue parameters:")
    print(f"  Weights: {true_weights}")
    print(f"  Shapes: [2.0, 3.0]")
    print(f"  Scales: [1.0, 3.0]")
    print(f"  Means: [2.0, 9.0]")

    # Fit model
    print(f"\nFitting 2-component Gamma mixture...")
    weights, shapes, scales, model = fit_gamma_mixture(
        X, n_components=2, verbose=True, max_iter=1000
    )

    print(f"\nEstimated parameters:")
    print(f"  Weights: {weights}")
    print(f"  Shapes: {shapes}")
    print(f"  Scales: {scales}")
    print(f"  Means: {shapes * scales}")
    print(f"\nConverged: {model.converged_}")
    print(f"Iterations: {model.n_iter_}")
    print(f"Log-likelihood: {model.loglik_:.2f}")
