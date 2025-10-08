"""
Performance Analysis Module for Risk Management System
CORRECTED: Uses actual crisis return from data, proper table formatting
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class PerformanceAnalysis:
    """
    Goodness-of-fit testing with corrected forecasting and table formatting
    """
    
    def __init__(self):
        self.models = None
        self.test_results = {}
        self.results_table = None
        self.crash_analysis = {}
        self.var_results = None
        self.var_data = None
        self.testing_period = None
        
    def set_models(self, ts_models):
        """Set the time series models object"""
        self.models = ts_models
        
    def set_testing_period(self, start_date: str = None, end_date: str = None):
        """Set the testing period for analysis"""
        self.testing_period = (start_date, end_date)
        
    def ks_test(self, residuals: np.ndarray, distribution: str, nu: float = None) -> Tuple[float, float, float]:
        """Kolmogorov-Smirnov test with critical value"""
        if distribution == 'normal':
            ks_stat, p_value = stats.kstest(residuals, 'norm')
        else:  # Student-t
            t_cdf = lambda x: stats.t.cdf(x, df=nu)
            ks_stat, p_value = stats.kstest(residuals, t_cdf)
        
        # Critical value at 5% significance for KS test
        n = len(residuals)
        ks_critical = 1.36 / np.sqrt(n)
        
        return ks_stat, p_value, ks_critical
    
    def ad_test(self, residuals: np.ndarray, distribution: str, nu: float = None) -> Tuple[float, float]:
        """Anderson-Darling test"""
        if distribution == 'normal':
            result = stats.anderson(residuals, dist='norm')
            ad_stat = result.statistic
            critical_value = result.critical_values[2]
        else:  # Student-t
            uniform_data = stats.t.cdf(residuals, df=nu)
            normal_data = stats.norm.ppf(uniform_data)
            normal_data = normal_data[np.isfinite(normal_data)]
            result = stats.anderson(normal_data, dist='norm')
            ad_stat = result.statistic
            critical_value = result.critical_values[2]
        return ad_stat, critical_value
    
    def evaluate_all(self, crisis_return: float = None) -> pd.DataFrame:
        """Evaluate all models and create results table with Pass/Fail columns"""
        results = []
        
        for model_name, params in self.models.models.items():
            # Get residuals
            residuals = self.models.get_residuals(model_name)
            
            # Determine distribution
            if 'normal' in model_name:
                dist = 'normal'
                nu = None
            else:
                dist = 't'
                nu = params['nu']
            
            # Perform tests
            ks_stat, ks_pval, ks_crit = self.ks_test(residuals, dist, nu)
            ad_stat, ad_crit = self.ad_test(residuals, dist, nu)
            
            # Pass/Fail determination
            ks_pass = "Pass" if ks_stat < ks_crit else "Fail"
            ad_pass = "Pass" if ad_stat < ad_crit else "Fail"
            
            # Calculate crisis residual if crisis return provided
            if crisis_return is not None:
                # Use one-step-ahead forecast for conditional variance
                forecast = self.models.forecast_one_step(model_name)
                mu_forecast = forecast['mu_forecast']
                h_forecast = forecast['h_forecast']
                
                # Standardized crisis residual
                crisis_residual = (crisis_return - mu_forecast) / np.sqrt(h_forecast)
            else:
                crisis_residual = None
            
            # Store results
            self.test_results[model_name] = {
                'ks_stat': ks_stat,
                'ks_pval': ks_pval,
                'ks_crit': ks_crit,
                'ks_pass': ks_pass,
                'ad_stat': ad_stat,
                'ad_crit': ad_crit,
                'ad_pass': ad_pass,
                'residuals': residuals,
                'crisis_residual': crisis_residual
            }
            
            # Create table row
            row = {
                'Model': model_name.replace('_', ' '),
                'α₀': f"{params['alpha_0']:.6e}",
                'α₁': f"{params['alpha_1']:.6f}",
                'β₁': f"{params['beta_1']:.6f}",
                'a': f"{params['a']:.6f}",
                'b': f"{params['b']:.6f}",
                'c': f"{params['c']:.6e}",
                'ν': f"{params['nu']:.2f}" if params['nu'] is not None else "-",
                'ε*': f"{crisis_residual:.4f}" if crisis_residual is not None else "-",
                'KS_stat': f"{ks_stat:.6f}",
                'KS_pval': f"{ks_pval:.6f}",
                'KS_Test': ks_pass,
                'AD_stat': f"{ad_stat:.6f}",
                'AD_Test': ad_pass
            }
            results.append(row)
        
        self.results_table = pd.DataFrame(results)
        
        # Print with better formatting
        print("\nRESULTS TABLE:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(self.results_table.to_string(index=False))
        
        return self.results_table

    def historical_crash_analysis(self, data_cleaner, crisis_date: pd.Timestamp = None):
        """
        Historical Crash Probability Analysis using conditional variance
        Uses ACTUAL crisis return from data, not hardcoded threshold
        """
        print("\n" + "="*70)
        print("HISTORICAL CRASH PROBABILITY ANALYSIS")
        print("="*70)
        
        # Get full data including the crisis date
        if data_cleaner.full_data is not None:
            full_data = data_cleaner.full_data.copy()
        else:
            full_data = data_cleaner.cleaned_data.copy()
        
        full_data = full_data.set_index('Date')
        
        if crisis_date is None:
            crisis_date = pd.Timestamp('2008-09-29')
        
        # Get ACTUAL crisis return from data
        if crisis_date not in full_data.index:
            print(f"Warning: Crisis date {crisis_date.date()} not found in data")
            print("Using worst return date instead")
            crisis_date = full_data['Return'].idxmin()
        
        # THIS IS THE KEY FIX: Use actual return from data
        crisis_return = full_data.loc[crisis_date, 'Return']
        
        print(f"\nCrisis Event:")
        print(f"  Date: {crisis_date.date()}")
        print(f"  Actual Return: {crisis_return:.6f} ({crisis_return*100:.4f}%)")
        
        print("\n" + "-"*70)
        print("EXPECTED TIME TO CRASH")
        print("-"*70)
        
        results_by_model = {}
        
        for model_name in self.models.models:
            params = self.models.models[model_name]
            
            # Get one-step-ahead forecasts
            forecast = self.models.forecast_one_step(model_name)
            mu_forecast = forecast['mu_forecast']
            h_forecast = forecast['h_forecast']
            
            # Use ACTUAL crisis_return here, not hardcoded L_max
            epsilon_star = (crisis_return - mu_forecast) / np.sqrt(h_forecast)
            
            # Calculate probability
            if 'normal' in model_name:
                prob_crash = stats.norm.cdf(epsilon_star)
            else:
                nu = params['nu']
                prob_crash = stats.t.cdf(epsilon_star, df=nu)
            
            # Calculate expected time
            if prob_crash > 0:
                avg_time_days = 1 / prob_crash           # Days
                avg_time_years = avg_time_days / 250     # Years
            else:
                avg_time_days = np.inf
                avg_time_years = np.inf
            
            print(f"\n{model_name}:")
            print(f"  μ_{{T+1}}: {mu_forecast:.6f}")
            print(f"  h_{{T+1}}: {h_forecast:.6e}")
            print(f"  ε*: {epsilon_star:.4f}")
            print(f"  P[ε ≤ ε*]: {prob_crash:.6e}")
            print(f"  Expected time: {avg_time_days:.1f} days ({avg_time_years:.2e} years)")
            
            results_by_model[model_name] = {
                'mu_forecast': mu_forecast,
                'h_forecast': h_forecast,
                'epsilon_star': epsilon_star,
                'probability': prob_crash,
                'avg_time_days': avg_time_days,
                'avg_time_years': avg_time_years
            }
        
        # Store results
        self.crash_analysis['crisis_date'] = crisis_date
        self.crash_analysis['crisis_return'] = crisis_return
        self.crash_analysis['model_results'] = results_by_model
        
        # Print summary table ONCE
        print("\n" + "-"*80)
        print("SUMMARY: CRASH PROBABILITY AND AVERAGE TIME")
        print("-"*80)
        print(f"{'Model':<20} {'ε*':<8} {'Probability':<14} {'Days':<15} {'Years':<15}")
        print("-" * 80)
        
        summary_data = []
        for model_name, results in results_by_model.items():
            days_str = f"{results['avg_time_days']:.1f}" if results['avg_time_days'] < 1e10 else f"{results['avg_time_days']:.2e}"
            years_str = f"{results['avg_time_years']:.2e}"
            
            print(f"{model_name.replace('_', ' '):<20} {results['epsilon_star']:<8.2f} {results['probability']:<14.2e} {days_str:<15} {years_str:<15}")
            
            summary_data.append({
                'Model': model_name.replace('_', ' '),
                'Residual (ε*)': results['epsilon_star'],
                'Probability': results['probability'],
                'Average Time (days)': results['avg_time_days'],
                'Average Time (years)': results['avg_time_years']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('crash_analysis_results.csv', index=False)
        print("\nCrash analysis results saved to: crash_analysis_results.csv")
        
        return summary_df
    
    def var_backtesting_custom(self, data_cleaner, 
                              rolling_window_years: int = 10,
                              test_start_date: str = None,
                              test_end_date: str = None,
                              confidence_level: float = 0.95):
        """VaR backtesting with custom testing period"""
        test_start = pd.to_datetime(test_start_date)
        test_end = pd.to_datetime(test_end_date)
        
        print("\n" + "="*70)
        print("VAR BACKTESTING ANALYSIS")
        print("="*70)
        print(f"Rolling window: {rolling_window_years} years")
        print(f"Testing period: {test_start.date()} to {test_end.date()}")
        print(f"Confidence level: {confidence_level:.1%}")
        
        if data_cleaner.full_data is not None:
            full_data = data_cleaner.full_data.copy()
        else:
            full_data = data_cleaner.cleaned_data.copy()
        
        full_data = full_data.set_index('Date')
        
        data_start = test_start - pd.DateOffset(years=rolling_window_years)
        analysis_data = full_data[data_start:test_end]
        
        if len(analysis_data) == 0:
            print("Error: No data available for specified period")
            return None, None
        
        returns = analysis_data['Return'].dropna().values
        dates = analysis_data['Return'].dropna().index
        
        rolling_window = rolling_window_years * 252
        
        test_start_idx = np.where(dates >= test_start)[0]
        if len(test_start_idx) == 0:
            print("Error: Testing start date not in data range")
            return None, None
        test_start_idx = test_start_idx[0]
        
        if test_start_idx < rolling_window:
            print(f"Warning: Insufficient data for {rolling_window_years}-year rolling window")
            rolling_window = test_start_idx - 1
        
        var_results = {}
        alpha = 1 - confidence_level
        
        for model_name in self.models.models:
            print(f"\nCalculating VaR for {model_name}...")
            params = self.models.models[model_name]
            
            var_forecast = []
            avar_forecast = []
            
            if 'GARCH' in model_name:
                sigma2_t = params['alpha_0'] / (1 - params['alpha_1'] - params['beta_1'])
            
            for t in range(test_start_idx, len(returns)):
                window_returns = returns[t-rolling_window:t]
                
                if 'CV' in model_name:
                    window_var = np.var(window_returns)
                    sigma = np.sqrt(window_var)
                    
                elif 'GARCH' in model_name and 'ARMA' not in model_name:
                    if t > test_start_idx:
                        prev_return = returns[t-1]
                        epsilon_sq = (prev_return - params['c']) ** 2
                        sigma2_t = params['alpha_0'] + params['alpha_1'] * epsilon_sq + params['beta_1'] * sigma2_t
                        sigma2_t = max(sigma2_t, 1e-10)
                    sigma = np.sqrt(sigma2_t)
                    
                else:  # ARMA-GARCH
                    if t > test_start_idx:
                        prev_return = returns[t-1]
                        epsilon_sq = (prev_return - params['c']) ** 2
                        sigma2_t = params['alpha_0'] + params['alpha_1'] * epsilon_sq + params['beta_1'] * sigma2_t
                        sigma2_t = max(sigma2_t, 1e-10)
                    sigma = np.sqrt(sigma2_t)
                
                if 'normal' in model_name:
                    var_value = params['c'] + sigma * stats.norm.ppf(alpha)
                    avar_value = params['c'] - sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
                else:
                    nu = params['nu']
                    t_adjustment = np.sqrt((nu - 2) / nu) if nu > 2 else 1
                    var_value = params['c'] + sigma * t_adjustment * stats.t.ppf(alpha, df=nu)
                    t_ppf = stats.t.ppf(alpha, df=nu)
                    t_pdf = stats.t.pdf(t_ppf, df=nu)
                    avar_value = params['c'] - sigma * t_adjustment * (t_pdf / alpha) * ((nu + t_ppf**2) / (nu - 1))
                
                var_forecast.append(var_value)
                avar_forecast.append(avar_value)
            
            actual_returns = returns[test_start_idx:]
            var_forecast = np.array(var_forecast)
            avar_forecast = np.array(avar_forecast)
            
            violations = (actual_returns < var_forecast).astype(int)
            
            var_results[model_name] = {
                'var': var_forecast,
                'avar': avar_forecast,
                'returns': actual_returns,
                'violations': violations,
                'dates': dates[test_start_idx:]
            }
        
        self.var_results = var_results
        self.var_data = {'rolling_window': rolling_window, 'alpha': alpha}
        
        results_df = self._calculate_var_test_statistics(var_results, alpha)
        self._plot_var_results(var_results, alpha)
        
        return results_df, var_results
    
    def _calculate_var_test_statistics(self, var_results, alpha):
        """Calculate CLR and BLR test statistics"""
        print("\n" + "-"*50)
        print("LIKELIHOOD RATIO TESTS")
        print("-"*50)
        
        test_results = {}
        
        for model_name in var_results:
            violations = var_results[model_name]['violations']
            returns = var_results[model_name]['returns']
            var_values = var_results[model_name]['var']
            
            T = len(violations)
            n = np.sum(violations)
            p_hat = n / T if T > 0 else 0
            
            if n > 0 and n < T:
                L0_uc = (1 - alpha)**(T - n) * alpha**n
                L1_uc = (1 - p_hat)**(T - n) * p_hat**n
                CLRuc = -2 * np.log(L0_uc / L1_uc) if L1_uc > 0 else np.inf
                CLRuc_pval = 1 - stats.chi2.cdf(CLRuc, 1)
            else:
                CLRuc = 0 if n == 0 else np.inf
                CLRuc_pval = 1.0 if n == 0 else 0.0
            
            n00, n01, n10, n11 = 0, 0, 0, 0
            for i in range(1, T):
                if violations[i-1] == 0 and violations[i] == 0:
                    n00 += 1
                elif violations[i-1] == 0 and violations[i] == 1:
                    n01 += 1
                elif violations[i-1] == 1 and violations[i] == 0:
                    n10 += 1
                else:
                    n11 += 1
            
            if n01 > 0 and n11 > 0 and n00 > 0 and n10 > 0:
                pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
                pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
                pi = (n01 + n11) / (n00 + n01 + n10 + n11)
                
                L0_ind = ((1 - pi)**(n00 + n10)) * (pi**(n01 + n11))
                L1_ind = ((1 - pi01)**n00) * (pi01**n01) * ((1 - pi11)**n10) * (pi11**n11)
                CLRind = -2 * np.log(L0_ind / L1_ind) if L1_ind > 0 else 0
                CLRind_pval = 1 - stats.chi2.cdf(CLRind, 1)
            else:
                CLRind = 0
                CLRind_pval = 1.0
            
            CLRcc = CLRuc + CLRind
            CLRcc_pval = 1 - stats.chi2.cdf(CLRcc, 2)
            
            if n > 1:
                violation_mean = np.mean(violations)
                violation_centered = violations - violation_mean
                if np.var(violation_centered) > 0:
                    autocorr = np.corrcoef(violation_centered[:-1], violation_centered[1:])[0, 1]
                    BLRind = T * autocorr**2
                    BLRind_pval = 1 - stats.chi2.cdf(BLRind, 1)
                else:
                    BLRind = 0
                    BLRind_pval = 1.0
            else:
                BLRind = 0
                BLRind_pval = 1.0
            
            exceedances = returns[violations == 1] - var_values[violations == 1]
            if len(exceedances) > 1:
                mean_excess = np.mean(np.abs(exceedances))
                std_excess = np.std(np.abs(exceedances))
                if std_excess > 0:
                    BLR_tail = len(exceedances) * ((mean_excess / std_excess - 1)**2) / 2
                    BLR_tail_pval = 1 - stats.chi2.cdf(BLR_tail, 1)
                else:
                    BLR_tail = 0
                    BLR_tail_pval = 1.0
            else:
                BLR_tail = 0
                BLR_tail_pval = 1.0
            
            test_results[model_name] = {
                'CLRuc': CLRuc,
                'CLRuc_pval': CLRuc_pval,
                'CLRind': CLRind,
                'CLRind_pval': CLRind_pval,
                'CLRcc': CLRcc,
                'CLRcc_pval': CLRcc_pval,
                'BLRind': BLRind,
                'BLRind_pval': BLRind_pval,
                'BLR_tail': BLR_tail,
                'BLR_tail_pval': BLR_tail_pval,
                'violations': n,
                'violation_rate': p_hat
            }
        
        grouped_results = self._group_similar_models(test_results)
        
        print("\n" + "="*70)
        print("BACKTESTING RESULTS TABLE (1% SIGNIFICANCE LEVEL)")
        print("="*70)
        
        crit_1df_01 = stats.chi2.ppf(0.99, 1)
        crit_2df_01 = stats.chi2.ppf(0.99, 2)
        
        table_data = []
        for model_group, results in grouped_results.items():
            clruc_pass = "PASS" if results['CLRuc_pval'] > 0.01 else "FAIL"
            clrind_pass = "PASS" if results['CLRind_pval'] > 0.01 else "FAIL"
            clrcc_pass = "PASS" if results['CLRcc_pval'] > 0.01 else "FAIL"
            blrind_pass = "PASS" if results['BLRind_pval'] > 0.01 else "FAIL"
            blrtail_pass = "PASS" if results['BLR_tail_pval'] > 0.01 else "FAIL"
            
            table_data.append({
                'Model': model_group,
                'Violations': f"{results['violations']} ({results['violation_rate']:.2%})",
                'CLRuc': f"{results['CLRuc']:.3f}",
                'CLRuc p-val': f"{results['CLRuc_pval']:.4f}",
                'CLRuc Test': clruc_pass,
                'CLRind': f"{results['CLRind']:.3f}",
                'CLRind p-val': f"{results['CLRind_pval']:.4f}",
                'CLRind Test': clrind_pass,
                'CLRcc': f"{results['CLRcc']:.3f}",
                'CLRcc p-val': f"{results['CLRcc_pval']:.4f}",
                'CLRcc Test': clrcc_pass,
                'BLRind': f"{results['BLRind']:.3f}",
                'BLRind p-val': f"{results['BLRind_pval']:.4f}",
                'BLRind Test': blrind_pass,
                'BLR_tail': f"{results['BLR_tail']:.3f}",
                'BLR_tail p-val': f"{results['BLR_tail_pval']:.4f}",
                'BLR_tail Test': blrtail_pass
            })
        
        results_df = pd.DataFrame(table_data)
        print(results_df.to_string(index=False))
        
        print(f"\nCritical values at 1% significance: χ²(1) = {crit_1df_01:.3f}, χ²(2) = {crit_2df_01:.3f}")
        
        print("\n" + "-"*50)
        print("SUMMARY")
        print("-"*50)
        for index, row in results_df.iterrows():
            model = row['Model']
            tests = ['CLRuc', 'CLRind', 'CLRcc', 'BLRind', 'BLR_tail']
            failed = [test for test in tests if row[f'{test} Test'] == 'FAIL']
            if failed:
                print(f"{model}: Failed tests - {', '.join(failed)}")
            else:
                print(f"{model}: All tests passed")
        
        results_df.to_csv('var_backtesting_results.csv', index=False)
        print("\nBacktesting results saved to: var_backtesting_results.csv")
        
        return results_df
    
    def _group_similar_models(self, test_results):
        """Group models with identical test statistics"""
        grouped = {}
        processed = set()
        
        for model1, results1 in test_results.items():
            if model1 in processed:
                continue
            
            similar_models = [model1]
            for model2, results2 in test_results.items():
                if model2 != model1 and model2 not in processed:
                    if (abs(results1['CLRuc'] - results2['CLRuc']) < 0.001 and
                        abs(results1['CLRind'] - results2['CLRind']) < 0.001 and
                        results1['violations'] == results2['violations']):
                        similar_models.append(model2)
                        processed.add(model2)
            
            if len(similar_models) > 1:
                base_names = []
                for model in similar_models:
                    if 'ARMA_GARCH' in model:
                        base = 'ARMA-GARCH'
                    elif 'GARCH' in model:
                        base = 'GARCH'
                    elif 'CV' in model:
                        base = 'CV'
                    else:
                        base = model
                    
                    dist = 'Normal' if 'normal' in model else 'Student-t'
                    base_names.append(f"{base} {dist}")
                
                if 'GARCH Normal' in base_names and 'ARMA-GARCH Normal' in base_names:
                    group_name = '(ARMA-)GARCH Normal'
                elif 'GARCH Student-t' in base_names and 'ARMA-GARCH Student-t' in base_names:
                    group_name = '(ARMA-)GARCH Student-t'
                else:
                    group_name = ' & '.join(sorted(set(base_names)))
            else:
                model = similar_models[0]
                if 'ARMA_GARCH' in model:
                    base = 'ARMA-GARCH'
                elif 'GARCH' in model:
                    base = 'GARCH'
                elif 'CV' in model:
                    base = 'CV'
                else:
                    base = model
                dist = 'Normal' if 'normal' in model else 'Student-t'
                group_name = f"{base} {dist}"
            
            grouped[group_name] = results1
            processed.add(model1)
        
        return grouped
    
    def economic_significance_analysis(self):
        """Economic significance analysis - ARD calculation"""
        if self.var_results is None:
            print("Error: VaR backtesting must be run first")
            return None
        
        print("\n" + "="*70)
        print("ECONOMIC SIGNIFICANCE ANALYSIS")
        print("="*70)
        
        first_model = list(self.var_results.keys())[0]
        dates = self.var_results[first_model]['dates']
        years = dates.year.unique()
        
        ard_results = {}
        avar_results = {}
        
        model_pairs = [
            ('CV', 'CV_normal', 'CV_t'),
            ('GARCH', 'GARCH_normal', 'GARCH_t'),
            ('ARMA-GARCH', 'ARMA_GARCH_normal', 'ARMA_GARCH_t')
        ]
        
        for model_type, normal_model, t_model in model_pairs:
            if normal_model not in self.var_results or t_model not in self.var_results:
                continue
                
            var_normal = self.var_results[normal_model]['var']
            var_t = self.var_results[t_model]['var']
            avar_normal = self.var_results[normal_model]['avar']
            avar_t = self.var_results[t_model]['avar']
            
            for model in [normal_model, t_model]:
                yearly_ard = {}
                yearly_avar = {}
                
                for year in years:
                    year_mask = dates.year == year
                    if np.sum(year_mask) > 0:
                        relative_diff = (var_t[year_mask] - var_normal[year_mask]) / np.abs(var_normal[year_mask])
                        ard = np.mean(relative_diff) * 100
                        yearly_ard[year] = ard
                        
                        if 'normal' in model:
                            yearly_avar[year] = np.mean(avar_normal[year_mask])
                        else:
                            yearly_avar[year] = np.mean(avar_t[year_mask])
                
                ard_results[model] = yearly_ard
                avar_results[model] = yearly_avar
        
        print("\nARD (%) - Average Relative Difference between Student-t and Normal VaR")
        print("-"*80)
        
        table_data = []
        for model_name in ['CV_normal', 'CV_t', 'GARCH_normal', 'GARCH_t', 'ARMA_GARCH_normal', 'ARMA_GARCH_t']:
            if model_name in ard_results:
                row = {'Model': model_name.replace('_', ' ')}
                for year in years:
                    if year in ard_results[model_name]:
                        row[str(year)] = f"{ard_results[model_name][year]:.2f}"
                    else:
                        row[str(year)] = "N/A"
                table_data.append(row)
        
        ard_df = pd.DataFrame(table_data)
        print(ard_df.to_string(index=False))
        
        print("\n" + "-"*80)
        print("AVaR (Average VaR / Expected Shortfall) by Year")
        print("-"*80)
        
        avar_table_data = []
        for model_name in ['CV_normal', 'CV_t', 'GARCH_normal', 'GARCH_t', 'ARMA_GARCH_normal', 'ARMA_GARCH_t']:
            if model_name in avar_results:
                row = {'Model': model_name.replace('_', ' ')}
                for year in years:
                    if year in avar_results[model_name]:
                        row[str(year)] = f"{avar_results[model_name][year]:.4f}"
                    else:
                        row[str(year)] = "N/A"
                avar_table_data.append(row)
        
        avar_df = pd.DataFrame(avar_table_data)
        print(avar_df.to_string(index=False))
        
        combined_results = pd.concat([
            pd.DataFrame({'': ['ARD Results (%)']}),
            ard_df,
            pd.DataFrame({'': ['']}),
            pd.DataFrame({'': ['AVaR Results']}),
            avar_df
        ], ignore_index=True)
        
        combined_results.to_csv('economic_significance_results.csv', index=False)
        print("\nEconomic significance results saved to: economic_significance_results.csv")
        
        return ard_df, avar_df
    
    def _plot_var_results(self, var_results: Dict, alpha: float):
        """Create VaR visualization plots"""
        normal_models = [m for m in var_results.keys() if 'normal' in m]
        t_models = [m for m in var_results.keys() if '_t' in m]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        first_model = normal_models[0] if normal_models else list(var_results.keys())[0]
        dates = var_results[first_model]['dates']
        returns = var_results[first_model]['returns']
        
        ax1.set_title(f'VaR Backtesting - Normal Distribution Models ({alpha:.1%} VaR)', fontsize=14)
        ax1.plot(dates, returns, color='gray', alpha=0.3, linewidth=0.5, label='Returns')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        colors = ['blue', 'green', 'orange']
        for idx, model_name in enumerate(normal_models[:3]):
            var_values = var_results[model_name]['var']
            avar_values = var_results[model_name]['avar']
            violations = var_results[model_name]['violations']
            
            ax1.plot(dates, var_values, color=colors[idx], alpha=0.7, 
                    linewidth=1, label=f'VaR {model_name.replace("_", " ")}')
            ax1.plot(dates, avar_values, color=colors[idx], alpha=0.5, 
                    linewidth=1, linestyle='--', label=f'AVaR {model_name.replace("_", " ")}')
            
            violation_dates = dates[violations == 1]
            violation_returns = returns[violations == 1]
            ax1.scatter(violation_dates, violation_returns, color=colors[idx], 
                       marker='x', s=30, alpha=0.8)
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Return')
        ax1.legend(loc='lower left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title(f'VaR Backtesting - Student-t Distribution Models ({alpha:.1%} VaR)', fontsize=14)
        ax2.plot(dates, returns, color='gray', alpha=0.3, linewidth=0.5, label='Returns')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        colors = ['red', 'purple', 'brown']
        for idx, model_name in enumerate(t_models[:3]):
            var_values = var_results[model_name]['var']
            avar_values = var_results[model_name]['avar']
            violations = var_results[model_name]['violations']
            
            ax2.plot(dates, var_values, color=colors[idx], alpha=0.7, 
                    linewidth=1, label=f'VaR {model_name.replace("_", " ")}')
            ax2.plot(dates, avar_values, color=colors[idx], alpha=0.5, 
                    linewidth=1, linestyle='--', label=f'AVaR {model_name.replace("_", " ")}')
            
            violation_dates = dates[violations == 1]
            violation_returns = returns[violations == 1]
            ax2.scatter(violation_dates, violation_returns, color=colors[idx], 
                       marker='x', s=30, alpha=0.8)
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Return')
        ax2.legend(loc='lower left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('var_backtesting_plots.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\nVaR plots saved to: var_backtesting_plots.png")