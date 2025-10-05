"""
Time Series Models Module for Risk Management System
FULLY CORRECTED based on Gemini AI feedback
- Dedicated CV-t implementation
- Global optimization for all t-models
- Proper parameter bounds
"""

import numpy as np
from typing import Dict
from scipy import stats
from scipy.optimize import minimize, differential_evolution
from scipy.special import gammaln

class TimeSeriesModels:
    """
    Implementation of CV, GARCH(1,1), and ARMA-GARCH(1,1) models
    CORRECTED for proper Student's t implementation and optimization
    """
    
    def __init__(self):
        self.data = None
        self.n = 0
        self.models = {}
        
    def set_data(self, returns: np.ndarray):
        """Set the returns data"""
        self.data = np.array(returns).flatten()
        self.n = len(self.data)
        
    def negative_log_likelihood_normal(self, params: np.ndarray, model_type: str) -> float:
        """Calculate negative log-likelihood under Normal distribution"""
        y = self.data
        T = self.n
        
        if model_type == 'CV':
            alpha_0, c = params
            if alpha_0 <= 0:
                return 1e10
            
            eta = y - c
            ll = -T/2 * np.log(2 * np.pi) - T/2 * np.log(alpha_0) - np.sum(eta**2) / (2 * alpha_0)
            
        elif model_type == 'GARCH':
            alpha_0, alpha_1, beta_1, c = params
            
            if alpha_0 <= 0 or alpha_1 < 0 or beta_1 < 0 or alpha_1 + beta_1 >= 1:
                return 1e10
            
            sigma2 = np.zeros(T)
            sigma2[0] = alpha_0 / (1 - alpha_1 - beta_1)
            
            eta = y - c
            ll = -0.5 * (np.log(2 * np.pi * sigma2[0]) + eta[0]**2 / sigma2[0])
            
            for t in range(1, T):
                sigma2[t] = alpha_0 + alpha_1 * eta[t-1]**2 + beta_1 * sigma2[t-1]
                sigma2[t] = max(sigma2[t], 1e-10)
                ll += -0.5 * (np.log(2 * np.pi * sigma2[t]) + eta[t]**2 / sigma2[t])
            
        elif model_type == 'ARMA_GARCH':
            a, b, c, alpha_0, alpha_1, beta_1 = params
            
            if alpha_0 <= 0 or alpha_1 < 0 or beta_1 < 0 or alpha_1 + beta_1 >= 1:
                return 1e10
            
            sigma2 = np.zeros(T)
            eta = np.zeros(T)
            sigma2[0] = alpha_0 / (1 - alpha_1 - beta_1)
            eta[0] = y[0] - c
            
            ll = -0.5 * (np.log(2 * np.pi * sigma2[0]) + eta[0]**2 / sigma2[0])
            
            for t in range(1, T):
                mu_t = c + a * y[t-1] + b * eta[t-1]
                eta[t] = y[t] - mu_t
                sigma2[t] = alpha_0 + alpha_1 * eta[t-1]**2 + beta_1 * sigma2[t-1]
                sigma2[t] = max(sigma2[t], 1e-10)
                ll += -0.5 * (np.log(2 * np.pi * sigma2[t]) + eta[t]**2 / sigma2[t])
        
        return -ll
    
    def negative_log_likelihood_studentt(self, params: np.ndarray, model_type: str) -> float:
        """
        Calculate negative log-likelihood under Student-t distribution
        CORRECTED: Dedicated CV implementation + vectorized GARCH/ARMA-GARCH
        """
        y = self.data
        T = self.n
        
        # ==================== CV MODEL (DEDICATED IMPLEMENTATION) ====================
        if model_type == 'CV':
            # CV parameters: [alpha_0, c, nu]
            if len(params) != 3:
                return 1e20
            
            alpha_0, c, nu = params
            
            # Parameter constraints
            if alpha_0 <= 1e-6 or nu <= 2.001:
                return 1e20
            
            # For CV, variance is constant: h_t = alpha_0
            epsilon = y - c
            h = np.full(T, alpha_0)  # Constant variance
            
            # Student's t log-likelihood with (ν-2) scaling
            log_const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
            log_h_term = -0.5 * np.sum(np.log(h))
            log_core_term = -((nu + 1) / 2) * np.sum(np.log(1 + (epsilon**2) / ((nu - 2) * h)))
            
            log_likelihood = T * log_const + log_h_term + log_core_term
            
            return -log_likelihood
        
        # ==================== GARCH MODEL ====================
        elif model_type == 'GARCH':
            # GARCH parameters: [alpha_0, alpha_1, beta_1, c, nu]
            if len(params) != 5:
                return 1e20
            
            alpha_0, alpha_1, beta_1, c, nu = params
            
            # STRICTER Parameter constraints (per Gemini's recommendation)
            # Check for GARCH stationarity and positivity
            if (alpha_1 + beta_1) >= 0.9999 or alpha_0 < 1e-10 or alpha_1 < 0 or beta_1 < 0:
                return 1e20  # Penalize non-stationary/non-positive parameters
            
            if nu <= 2.001:
                return 1e20
            
            # Compute conditional variance h_t
            h = np.zeros(T)
            h[0] = alpha_0 / (1 - alpha_1 - beta_1)
            if h[0] <= 0:
                return 1e20
            
            epsilon = y - c
            
            for t in range(1, T):
                h[t] = alpha_0 + alpha_1 * epsilon[t-1]**2 + beta_1 * h[t-1]
                if h[t] <= 1e-10:
                    h[t] = 1e-10
            
            # Vectorized log-likelihood calculation
            log_const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
            log_h_term = -0.5 * np.sum(np.log(h))
            log_core_term = -((nu + 1) / 2) * np.sum(np.log(1 + (epsilon**2) / ((nu - 2) * h)))
            
            log_likelihood = T * log_const + log_h_term + log_core_term
            
            return -log_likelihood
        
        # ==================== ARMA-GARCH MODEL ====================
        elif model_type == 'ARMA_GARCH':
            # ARMA-GARCH parameters: [a, b, c, alpha_0, alpha_1, beta_1, nu]
            if len(params) != 7:
                return 1e20
            
            a, b, c, alpha_0, alpha_1, beta_1, nu = params
            
            # STRICTER Parameter constraints (per Gemini's recommendation)
            # Check for GARCH stationarity and positivity
            if (alpha_1 + beta_1) >= 0.9999 or alpha_0 < 1e-10 or alpha_1 < 0 or beta_1 < 0:
                return 1e20  # Penalize non-stationary/non-positive parameters
            
            if nu <= 2.001:
                return 1e20
            
            # Compute conditional variance h_t and residuals epsilon_t
            h = np.zeros(T)
            epsilon = np.zeros(T)
            
            # Initialization
            h[0] = alpha_0 / (1 - alpha_1 - beta_1)
            if h[0] <= 0:
                return 1e20
            epsilon[0] = y[0] - c
            
            for t in range(1, T):
                y_pred = a * y[t-1] + b * epsilon[t-1] + c  # ARMA(1,1) mean equation
                epsilon[t] = y[t] - y_pred
                h[t] = alpha_0 + alpha_1 * epsilon[t-1]**2 + beta_1 * h[t-1]
                if h[t] <= 1e-10:
                    h[t] = 1e-10
            
            # Vectorized log-likelihood calculation
            log_const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
            log_h_term = -0.5 * np.sum(np.log(h))
            log_core_term = -((nu + 1) / 2) * np.sum(np.log(1 + (epsilon**2) / ((nu - 2) * h)))
            
            log_likelihood = T * log_const + log_h_term + log_core_term
            
            return -log_likelihood
        
        else:
            return 1e20
    
    def fit_cv_normal(self) -> Dict:
        """Fit CV model with Normal distribution - analytical solution"""
        c_hat = np.mean(self.data)
        alpha_0_hat = np.mean((self.data - c_hat)**2)
        
        ll = -self.negative_log_likelihood_normal([alpha_0_hat, c_hat], 'CV')
        
        return {
            'alpha_0': alpha_0_hat, 'alpha_1': 0, 'beta_1': 0,
            'a': 0, 'b': 0, 'c': c_hat, 'nu': None,
            'log_likelihood': ll
        }
    
    def fit_cv_t(self) -> Dict:
        """Fit CV model with Student-t distribution using global optimization"""
        print(f"\nFitting CV Student-t (dedicated implementation)...")
        
        c_init = np.mean(self.data)
        alpha_0_init = np.var(self.data)
        
        # Estimate nu from kurtosis
        excess_kurt = stats.kurtosis(self.data)
        if excess_kurt > 0:
            nu_init = 6 / excess_kurt + 4
        else:
            nu_init = 10.0
        nu_init = np.clip(nu_init, 3, 30)
        
        print(f"Initial: α₀={alpha_0_init:.6e}, c={c_init:.6e}, ν={nu_init:.2f}")
        
        # Global optimization with differential_evolution
        # Bounds: [alpha_0, c, nu]
        bounds = [
            (1e-6, 1e-2),      # alpha_0: variance
            (-0.5, 0.5),       # c: mean (FINITE bounds, not infinite)
            (2.1, 30)          # nu: degrees of freedom
        ]
        
        # NO constraints needed for CV model
        result = differential_evolution(
            lambda p: self.negative_log_likelihood_studentt(p, 'CV'),
            bounds,
            seed=42,
            maxiter=1000,
            atol=1e-12,
            tol=1e-12,
            workers=1
        )
        
        alpha_0, c, nu = result.x
        ll = -result.fun
        
        print(f"Final: α₀={alpha_0:.6e}, c={c:.6e}, ν={nu:.2f}")
        print(f"Log-likelihood: {ll:.4f}")
        
        return {
            'alpha_0': alpha_0, 'alpha_1': 0, 'beta_1': 0,
            'a': 0, 'b': 0, 'c': c, 'nu': nu,
            'log_likelihood': ll
        }
    
    def fit_garch_normal(self) -> Dict:
        """Fit GARCH model with Normal distribution"""
        print(f"\nFitting GARCH Normal...")
        
        # Try multiple starting points
        best_result = None
        best_ll = -np.inf
        
        starting_points = [
            [1e-6, 0.05, 0.90, 0.0],
            [5e-7, 0.06, 0.93, 0.0003],
            [1e-6, 0.08, 0.88, 0.0],
            [8e-7, 0.062, 0.933, 0.0003]
        ]
        
        for init_params in starting_points:
            try:
                result = minimize(
                    lambda p: self.negative_log_likelihood_normal(p, 'GARCH'),
                    init_params,
                    method='L-BFGS-B',
                    bounds=[(1e-9, 1e-4), (1e-6, 0.3), (0.5, 0.999), (-1e-3, 1e-3)],
                    options={'ftol': 1e-12, 'maxiter': 5000}
                )
                
                if -result.fun > best_ll:
                    best_ll = -result.fun
                    best_result = result
            except:
                continue
        
        if best_result is None:
            bounds = [(1e-9, 1e-4), (1e-6, 0.3), (0.5, 0.999), (-1e-3, 1e-3)]
            best_result = differential_evolution(
                lambda p: self.negative_log_likelihood_normal(p, 'GARCH'),
                bounds,
                seed=42,
                maxiter=1000
            )
            best_ll = -best_result.fun
        
        alpha_0, alpha_1, beta_1, c = best_result.x
        
        print(f"Final: α₀={alpha_0:.6e}, α₁={alpha_1:.4f}, β₁={beta_1:.4f}, c={c:.6e}")
        print(f"Log-likelihood: {best_ll:.4f}")
        
        return {
            'alpha_0': alpha_0, 'alpha_1': alpha_1, 'beta_1': beta_1,
            'a': 0, 'b': 0, 'c': c, 'nu': None,
            'log_likelihood': best_ll
        }
    
    def fit_garch_t(self) -> Dict:
        """Fit GARCH model with Student-t using global optimization with theory-driven bounds"""
        print(f"\nFitting GARCH Student-t (theory-driven initialization)...")
        
        # Get Normal estimates for initialization
        garch_normal = self.fit_garch_normal()
        
        print(f"Using Normal estimates as starting point:")
        print(f"  α₀={garch_normal['alpha_0']:.6e}, α₁={garch_normal['alpha_1']:.4f}, β₁={garch_normal['beta_1']:.4f}")
        
        # THEORY-DRIVEN BOUNDS (tighter, based on typical GARCH behavior)
        # α₀: very small (1e-6 typical)
        # α₁: 0.05 to 0.10 (typical shock response)
        # β₁: 0.85 to 0.90 (high persistence)
        # α₁ + β₁ ≈ 0.95 (high but < 1)
        # ν: 5.0 to 7.0 (close to paper's typical finding)
        
        best_result = None
        best_ll = -np.inf
        
        # Grid search over nu with tighter bounds
        nu_grid = [5.0, 6.0, 7.0, 8.0, 9.0, 9.5, 9.8, 10.0]
        
        for nu_init in nu_grid:
            # Tighter bounds around typical values
            bounds_theory = [
                (1e-7, 1e-5),        # alpha_0: very small baseline
                (0.03, 0.15),        # alpha_1: typical shock response
                (0.80, 0.96),        # beta_1: high persistence
                (-0.5, 0.5),         # c: mean (FINITE bounds for differential_evolution)
                (nu_init - 1.0, nu_init + 1.0)  # nu: narrow range
            ]
            
            # Use Normal estimates as initialization hint
            init_point = np.array([
                garch_normal['alpha_0'] * 0.6,  # Student's t typically has smaller α₀
                garch_normal['alpha_1'],
                garch_normal['beta_1'],
                garch_normal['c'],
                nu_init
            ])
            
            try:
                result = differential_evolution(
                    lambda p: self.negative_log_likelihood_studentt(p, 'GARCH'),
                    bounds_theory,
                    seed=42 + int(nu_init),
                    maxiter=800,
                    atol=1e-12,
                    tol=1e-12,
                    workers=1,
                    x0=init_point  # Use Normal estimates as hint
                )
                
                ll = -result.fun
                if ll > best_ll:
                    best_ll = ll
                    best_result = result
                    print(f"  ν≈{nu_init:.1f}: LL={ll:.4f}, α₁+β₁={result.x[1]+result.x[2]:.4f} (improved)")
            except Exception as e:
                print(f"  ν≈{nu_init:.1f}: optimization failed")
                continue
        
        if best_result is None:
            print("Warning: Global optimization failed, using Normal estimates")
            return {
                'alpha_0': garch_normal['alpha_0'],
                'alpha_1': garch_normal['alpha_1'],
                'beta_1': garch_normal['beta_1'],
                'a': 0, 'b': 0,
                'c': garch_normal['c'],
                'nu': 10.0,
                'log_likelihood': garch_normal['log_likelihood']
            }
        
        alpha_0, alpha_1, beta_1, c, nu = best_result.x
        
        print(f"Final: α₀={alpha_0:.6e}, α₁={alpha_1:.4f}, β₁={beta_1:.4f}, c={c:.6e}, ν={nu:.2f}")
        print(f"Persistence: α₁+β₁={alpha_1+beta_1:.4f}")
        print(f"Log-likelihood: {best_ll:.4f}")
        
        return {
            'alpha_0': alpha_0, 'alpha_1': alpha_1, 'beta_1': beta_1,
            'a': 0, 'b': 0, 'c': c, 'nu': nu,
            'log_likelihood': best_ll
        }
    
    def fit_arma_garch_normal(self) -> Dict:
        """Fit ARMA-GARCH model with Normal distribution"""
        print(f"\nFitting ARMA-GARCH Normal...")
        
        best_result = None
        best_ll = -np.inf
        
        # CORRECTED: Use wider bounds and starting points closer to Table 3
        starting_points = [
            [0.76, -0.81, 0.0, 8e-7, 0.061, 0.934],      # Close to Table 3
            [0.70, -0.80, 0.0001, 8e-7, 0.061, 0.934],   # Variation 1
            [0.80, -0.85, 0.0, 1e-6, 0.05, 0.93],        # Variation 2
            [0.75, -0.75, 0.0, 5e-7, 0.06, 0.93],        # Variation 3
            [0.5, -0.5, 0.0, 1e-6, 0.05, 0.90],          # Conservative
            [0.1, -0.1, 0.0, 1e-6, 0.08, 0.88]           # Very conservative
        ]
        
        # CORRECTED: Wider bounds to allow paper's estimates
        bounds = [
            (-0.95, 0.95),      # a: allow near unit-root AR
            (-0.95, 0.95),      # b: allow near unit-root MA
            (-1e-3, 1e-3),      # c: mean
            (1e-9, 1e-4),       # alpha_0
            (1e-6, 0.3),        # alpha_1
            (0.5, 0.999)        # beta_1
        ]
        
        for init_params in starting_points:
            try:
                result = minimize(
                    lambda p: self.negative_log_likelihood_normal(p, 'ARMA_GARCH'),
                    init_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'ftol': 1e-15, 'gtol': 1e-12, 'maxiter': 10000}
                )
                
                if -result.fun > best_ll:
                    best_ll = -result.fun
                    best_result = result
                    print(f"  Starting {init_params[:2]}: LL={best_ll:.4f}, a={result.x[0]:.4f}, b={result.x[1]:.4f}")
            except:
                continue
        
        # If local optimization didn't work well, try global
        if best_result is None or best_ll < 7900:  # Heuristic threshold
            print("  Local optimization suboptimal, trying global search...")
            try:
                result = differential_evolution(
                    lambda p: self.negative_log_likelihood_normal(p, 'ARMA_GARCH'),
                    bounds,
                    seed=42,
                    maxiter=1000,
                    atol=1e-12,
                    tol=1e-12,
                    workers=1
                )
                if -result.fun > best_ll:
                    best_ll = -result.fun
                    best_result = result
            except:
                pass
        
        if best_result is None:
            raise ValueError("ARMA-GARCH Normal optimization failed")
        
        a, b, c, alpha_0, alpha_1, beta_1 = best_result.x
        
        print(f"Final: a={a:.4f}, b={b:.4f}, c={c:.6e}, α₀={alpha_0:.6e}, α₁={alpha_1:.4f}, β₁={beta_1:.4f}")
        print(f"Log-likelihood: {best_ll:.4f}")
        
        return {
            'alpha_0': alpha_0, 'alpha_1': alpha_1, 'beta_1': beta_1,
            'a': a, 'b': b, 'c': c, 'nu': None,
            'log_likelihood': best_ll
        }

    def fit_arma_garch_t(self) -> Dict:
        """Fit ARMA-GARCH model with Student-t using theory-driven optimization"""
        print(f"\nFitting ARMA-GARCH Student-t (theory-driven initialization)...")
        
        # Get Normal estimates for initialization
        arma_normal = self.fit_arma_garch_normal()
        
        print(f"Using Normal estimates as starting point:")
        print(f"  a={arma_normal['a']:.4f}, b={arma_normal['b']:.4f}, c={arma_normal['c']:.6e}")
        print(f"  α₀={arma_normal['alpha_0']:.6e}, α₁={arma_normal['alpha_1']:.4f}, β₁={arma_normal['beta_1']:.4f}")
        
        best_result = None
        best_ll = -np.inf
        
        nu_grid = [5.0, 6.0, 7.0, 8.0, 9.0, 9.5, 9.8, 10.0]
        
        for nu_init in nu_grid:
            # CORRECTED: Wider bounds for ARMA parameters
            bounds_theory = [
                (-0.95, 0.95),       # a: AR coefficient
                (-0.95, 0.95),       # b: MA coefficient  
                (-5e-4, 5e-4),       # c: small mean
                (1e-7, 1e-5),        # alpha_0: very small baseline
                (0.03, 0.15),        # alpha_1: typical shock response
                (0.80, 0.96),        # beta_1: high persistence
                (nu_init - 1.0, nu_init + 1.0)  # nu: narrow range
            ]
            
            # Use Normal estimates as initialization
            init_point = np.array([
                arma_normal['a'],
                arma_normal['b'],
                arma_normal['c'],
                arma_normal['alpha_0'] * 0.6,  # Student's t typically smaller α₀
                arma_normal['alpha_1'],
                arma_normal['beta_1'],
                nu_init
            ])
            
            try:
                result = differential_evolution(
                    lambda p: self.negative_log_likelihood_studentt(p, 'ARMA_GARCH'),
                    bounds_theory,
                    seed=42 + int(nu_init * 10),
                    maxiter=800,
                    atol=1e-12,
                    tol=1e-12,
                    workers=1,
                    x0=init_point
                )
                
                ll = -result.fun
                if ll > best_ll:
                    best_ll = ll
                    best_result = result
                    print(f"  ν≈{nu_init:.1f}: LL={ll:.4f}, a={result.x[0]:.4f}, b={result.x[1]:.4f}, α₁+β₁={result.x[4]+result.x[5]:.4f}")
            except Exception as e:
                print(f"  ν≈{nu_init:.1f}: optimization failed")
                continue
        
        if best_result is None:
            print("Warning: Global optimization failed, using Normal estimates with ν=10")
            return {
                'alpha_0': arma_normal['alpha_0'],
                'alpha_1': arma_normal['alpha_1'],
                'beta_1': arma_normal['beta_1'],
                'a': arma_normal['a'],
                'b': arma_normal['b'],
                'c': arma_normal['c'],
                'nu': 10.0,
                'log_likelihood': arma_normal['log_likelihood']
            }
        
        a, b, c, alpha_0, alpha_1, beta_1, nu = best_result.x
        
        print(f"Final: a={a:.4f}, b={b:.4f}, c={c:.6e}")
        print(f"       α₀={alpha_0:.6e}, α₁={alpha_1:.4f}, β₁={beta_1:.4f}, ν={nu:.2f}")
        print(f"Persistence: α₁+β₁={alpha_1+beta_1:.4f}")
        print(f"Log-likelihood: {best_ll:.4f}")
        
        return {
            'alpha_0': alpha_0, 'alpha_1': alpha_1, 'beta_1': beta_1,
            'a': a, 'b': b, 'c': c, 'nu': nu,
            'log_likelihood': best_ll
        }
    
    def fit_all(self):
        """Fit all 6 models"""
        print("\n" + "="*70)
        print("FITTING ALL MODELS - GLOBAL OPTIMIZATION FOR STUDENT'S T")
        print("="*70)
        
        self.models = {
            'CV_normal': self.fit_cv_normal(),
            'CV_t': self.fit_cv_t(),
            'GARCH_normal': self.fit_garch_normal(),
            'GARCH_t': self.fit_garch_t(),
            'ARMA_GARCH_normal': self.fit_arma_garch_normal(),
            'ARMA_GARCH_t': self.fit_arma_garch_t()
        }
        
        return self.models
    
    def forecast_one_step(self, model_name: str) -> Dict:
        """
        Forecast μ_{T+1} and h_{T+1} for one-step-ahead prediction
        CRITICAL: Uses conditional variance h_{T+1}, not unconditional variance
        
        Formula: h_{T+1} = α_0 + α_1 * ε_T^2 + β_1 * h_T
        """
        params = self.models[model_name]
        y = self.data
        T = self.n
        
        if 'CV' in model_name:
            # CV: constant mean and variance
            mu_forecast = params['c']
            h_forecast = params['alpha_0']
            
        elif 'ARMA_GARCH' in model_name:
            # Compute all historical h_t and ε_t
            h = np.zeros(T)
            epsilon = np.zeros(T)
            h[0] = params['alpha_0'] / (1 - params['alpha_1'] - params['beta_1'])
            epsilon[0] = y[0] - params['c']
            
            for t in range(1, T):
                y_pred = params['a'] * y[t-1] + params['b'] * epsilon[t-1] + params['c']
                epsilon[t] = y[t] - y_pred
                h[t] = params['alpha_0'] + params['alpha_1'] * epsilon[t-1]**2 + params['beta_1'] * h[t-1]
            
            # One-step-ahead forecast using LAST observation (T)
            mu_forecast = params['a'] * y[T-1] + params['b'] * epsilon[T-1] + params['c']
            h_forecast = params['alpha_0'] + params['alpha_1'] * epsilon[T-1]**2 + params['beta_1'] * h[T-1]
            
        else:  # GARCH
            # Compute all historical h_t
            h = np.zeros(T)
            h[0] = params['alpha_0'] / (1 - params['alpha_1'] - params['beta_1'])
            
            epsilon = y - params['c']
            for t in range(1, T):
                h[t] = params['alpha_0'] + params['alpha_1'] * epsilon[t-1]**2 + params['beta_1'] * h[t-1]
            
            # One-step-ahead forecast using LAST observation (T)
            mu_forecast = params['c']
            h_forecast = params['alpha_0'] + params['alpha_1'] * epsilon[T-1]**2 + params['beta_1'] * h[T-1]
        
        return {
            'mu_forecast': mu_forecast,
            'h_forecast': h_forecast,
            'epsilon_T': epsilon[T-1] if 'ARMA_GARCH' in model_name or 'GARCH' in model_name else y[T-1] - params['c'],
            'h_T': h[T-1] if 'ARMA_GARCH' in model_name or 'GARCH' in model_name else params['alpha_0']
        }
    
    def calculate_efpt_monte_carlo(self, model_name: str, L_max: float = -0.09, 
                                   n_simulations: int = 10000, T_max_steps: int = 252*25) -> Dict:
        """
        Calculate Expected First Passage Time (EFPT) to crash barrier using Monte Carlo
        
        Simulates the fitted GARCH/ARMA-GARCH process until first crash (r_t < L_max)
        
        Parameters:
        model_name: Name of fitted model
        L_max: Crash barrier (default -0.09 for ~9% daily loss)
        n_simulations: Number of Monte Carlo paths
        T_max_steps: Maximum steps per simulation (default 25 years)
        
        Returns:
        Dictionary with EFPT in days and years, plus simulation statistics
        """
        params = self.models[model_name]
        
        # Determine distribution for innovations
        if 'normal' in model_name:
            dist_type = 'normal'
            nu = None
        else:
            dist_type = 't'
            nu = params['nu']
        
        # Get last observation for initialization
        forecast = self.forecast_one_step(model_name)
        h_initial = forecast['h_T']
        epsilon_initial = forecast['epsilon_T']
        y_initial = self.data[-1]
        
        first_passage_times = []
        
        for sim in range(n_simulations):
            # Initialize simulation
            if 'ARMA_GARCH' in model_name:
                h_t = h_initial
                epsilon_t = epsilon_initial
                y_t = y_initial
            elif 'GARCH' in model_name:
                h_t = h_initial
                epsilon_t = epsilon_initial
            else:  # CV
                h_t = params['alpha_0']
            
            # Simulate until crash or max time
            for t in range(1, T_max_steps + 1):
                # Generate innovation
                if dist_type == 'normal':
                    z_t = np.random.normal(0, 1)
                else:
                    # Student's t: generate raw t(ν) then apply correct variance scaling
                    # ε_t = sqrt(h_t) * sqrt((ν-2)/ν) * z_raw where z_raw ~ t(ν)
                    z_raw = np.random.standard_t(nu)
                    z_t = z_raw * np.sqrt((nu - 2) / nu)  # Scale to match NLL formula
                
                # Generate return based on model type
                if 'CV' in model_name:
                    r_t = params['c'] + np.sqrt(h_t) * z_t
                    
                elif 'ARMA_GARCH' in model_name:
                    # ARMA(1,1) mean equation
                    mu_t = params['c'] + params['a'] * y_t + params['b'] * epsilon_t
                    r_t = mu_t + np.sqrt(h_t) * z_t
                    
                    # Update for next step
                    epsilon_t = r_t - mu_t
                    h_t = params['alpha_0'] + params['alpha_1'] * epsilon_t**2 + params['beta_1'] * h_t
                    y_t = r_t
                    
                else:  # GARCH
                    r_t = params['c'] + np.sqrt(h_t) * z_t
                    
                    # Update variance for next step
                    epsilon_t = r_t - params['c']
                    h_t = params['alpha_0'] + params['alpha_1'] * epsilon_t**2 + params['beta_1'] * h_t
                
                # Check if crash occurred
                if r_t < L_max:
                    first_passage_times.append(t)
                    break
            else:
                # Reached max time without crash - censored observation
                first_passage_times.append(T_max_steps)
        
        # Calculate statistics
        first_passage_times = np.array(first_passage_times)
        n_crashes = np.sum(first_passage_times < T_max_steps)
        n_censored = np.sum(first_passage_times == T_max_steps)
        
        efpt_days = np.mean(first_passage_times)
        efpt_years = efpt_days / 252
        
        # Calculate confidence interval (excluding censored)
        crashed_times = first_passage_times[first_passage_times < T_max_steps]
        if len(crashed_times) > 0:
            ci_lower = np.percentile(crashed_times, 2.5) / 252
            ci_upper = np.percentile(crashed_times, 97.5) / 252
        else:
            ci_lower = ci_upper = efpt_years
        
        return {
            'efpt_days': efpt_days,
            'efpt_years': efpt_years,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_simulations': n_simulations,
            'n_crashes': n_crashes,
            'n_censored': n_censored,
            'crash_probability': n_crashes / n_simulations
        }
    
    def get_residuals(self, model_name: str) -> np.ndarray:
        """
        Calculate standardized residuals for a model
        CORRECTED: Proper standardization for all models
        """
        params = self.models[model_name]
        y = self.data
        n = self.n
        
        if 'CV' in model_name:
            # CV model: z_t = (r_t - μ) / σ = (r_t - c) / sqrt(α₀)
            sigma = np.sqrt(params['alpha_0'])
            residuals = (y - params['c']) / sigma
            
        elif 'ARMA_GARCH' in model_name:
            # ARMA-GARCH: compute conditional variance and innovations
            h = np.zeros(n)
            epsilon = np.zeros(n)
            h[0] = params['alpha_0'] / (1 - params['alpha_1'] - params['beta_1'])
            epsilon[0] = y[0] - params['c']
            
            for t in range(1, n):
                y_pred = params['a'] * y[t-1] + params['b'] * epsilon[t-1] + params['c']
                epsilon[t] = y[t] - y_pred
                h[t] = params['alpha_0'] + params['alpha_1'] * epsilon[t-1]**2 + params['beta_1'] * h[t-1]
            
            # Standardized residuals: z_t = ε_t / sqrt(h_t)
            residuals = epsilon / np.sqrt(h)
            
        else:  # GARCH
            # GARCH: compute conditional variance
            h = np.zeros(n)
            h[0] = params['alpha_0'] / (1 - params['alpha_1'] - params['beta_1'])
            
            epsilon = y - params['c']
            for t in range(1, n):
                h[t] = params['alpha_0'] + params['alpha_1'] * epsilon[t-1]**2 + params['beta_1'] * h[t-1]
            
            # Standardized residuals: z_t = (r_t - c) / sqrt(h_t)
            residuals = epsilon / np.sqrt(h)
        
        return residuals