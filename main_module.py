"""
Main Analysis System for Time Series Risk Management
Consolidates all modules and executes the analysis pipeline
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from data_cleaner_module import DataCleaner
from time_series_models_module import TimeSeriesModels
from performance_analysis_module import PerformanceAnalysis

class TimeSeriesAnalysisSystem:
    """
    Main system that links all modules together with flexible testing periods
    """
    
    def __init__(self):
        self.data_cleaner = DataCleaner(target_years=20)
        self.ts_models = TimeSeriesModels()
        self.performance = PerformanceAnalysis()
        self.testing_period = None
        self.selected_crisis_date = None
        self.rolling_window_years = None
        
    def run_analysis(self, data_dir: str = "."):
        """Run complete analysis pipeline with new workflow"""
        
        # Step 1: Data Cleaning
        print("="*70)
        print("STEP 1: DATA CLEANING")
        print("="*70)
        
        self.data_cleaner.load_and_merge(data_dir)
        self.data_cleaner.clean()  # Clean all data first
        
        print(f"\nFull data shape: {self.data_cleaner.full_data.shape}")
        print(f"Full data period: {self.data_cleaner.full_data['Date'].min().date()} to {self.data_cleaner.full_data['Date'].max().date()}")
        
        # Step 2: Get rolling window size
        print("\n" + "="*70)
        print("STEP 2: ROLLING WINDOW CONFIGURATION")
        print("="*70)
        
        try:
            self.rolling_window_years = int(input("Enter rolling window size in years (default 10): ") or "10")
        except:
            self.rolling_window_years = 10
        print(f"Rolling window set to: {self.rolling_window_years} years")
        
        # Step 3: Find crisis dates after calibration period
        print("\n" + "="*70)
        print("STEP 3: CRISIS DATE IDENTIFICATION")
        print("="*70)
        
        # Assume calibration period is first 10 years of data
        calibration_end = self.data_cleaner.full_data['Date'].min() + pd.DateOffset(years=10)
        print(f"Searching for crisis events after calibration period (after {calibration_end.date()})")
        
        crisis_dates = self.data_cleaner.find_crisis_dates(
            calibration_end_date=calibration_end.strftime('%Y-%m-%d'),
            top_n=3
        )
        
        print("\nTop 3 Crisis Dates (Worst Returns):")
        print("-"*50)
        crisis_options = {}
        for idx, crisis in enumerate(crisis_dates, 1):
            print(f"{idx}. {crisis['date'].date()}: {crisis['return']:.4%}")
            crisis_options[idx] = crisis['date']
        
        # Step 4: Get testing period configuration
        print("\n" + "="*70)
        print("STEP 4: TESTING PERIOD CONFIGURATION")
        print("="*70)
        
        print("\nAnalysis Options:")
        print("1. Analyze all data before selected crisis")
        print("2. Analyze full dataset")
        print("3. Specify custom testing period")
        
        choice = input("Select option (1/2/3, default=2): ").strip() or "2"
        
        if choice == "1":
            # Option 1: All data before crisis
            print("\nSelect crisis date from the list above:")
            crisis_choice = input("Enter crisis number (1/2/3): ").strip()
            
            try:
                crisis_idx = int(crisis_choice)
                if crisis_idx in crisis_options:
                    self.selected_crisis_date = crisis_options[crisis_idx]
                    
                    # Set testing period from start to day before crisis
                    start_date = self.data_cleaner.full_data['Date'].min().strftime('%Y-%m-%d')
                    end_date = (self.selected_crisis_date - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
                    
                    print(f"\nAnalyzing data before crisis: {start_date} to {end_date}")
                    print(f"Crisis date for stress testing: {self.selected_crisis_date.date()}")
                else:
                    raise ValueError("Invalid crisis selection")
            except:
                print("Invalid selection, using full dataset")
                choice = "2"
                
        if choice == "3":
            # Option 3: Custom period
            print("\nEnter custom testing period:")
            print("Examples:")
            print("  - 1 year (2005): Start=2005-01-01, End=2005-12-31")
            print("  - 4 years (2005-2008): Start=2005-01-01, End=2008-12-31")
            
            start_date = input("Start date (YYYY-MM-DD): ").strip()
            end_date = input("End date (YYYY-MM-DD): ").strip()
            
            if not start_date or not end_date:
                print("Invalid dates, using full dataset")
                choice = "2"
            else:
                print(f"\nAnalyzing custom period: {start_date} to {end_date}")
                
                # Ask for crisis date for stress testing
                print("\nSelect crisis date for stress testing from the list above:")
                crisis_choice = input("Enter crisis number (1/2/3, default=1): ").strip() or "1"
                try:
                    crisis_idx = int(crisis_choice)
                    if crisis_idx in crisis_options:
                        self.selected_crisis_date = crisis_options[crisis_idx]
                        print(f"Crisis date for stress testing: {self.selected_crisis_date.date()}")
                except:
                    self.selected_crisis_date = crisis_options[1]
                    print(f"Using default crisis date: {self.selected_crisis_date.date()}")
                    
        if choice == "2":
            # Option 2: Full dataset
            print("\nAnalyzing full dataset")
            start_date = None
            end_date = None
            
            # Use worst crisis date for stress testing
            self.selected_crisis_date = crisis_options[1]
            print(f"Using worst crisis date for stress testing: {self.selected_crisis_date.date()}")
        
        # Apply the selected testing period
        if choice != "2":
            self.testing_period = (start_date, end_date)
            self.data_cleaner.clean(start_date=start_date, end_date=end_date)
        else:
            self.testing_period = None
        
        returns = self.data_cleaner.get_returns()
        
        print(f"\nTesting data shape: {self.data_cleaner.cleaned_data.shape}")
        print(f"Testing data period: {self.data_cleaner.cleaned_data['Date'].min().date()} to {self.data_cleaner.cleaned_data['Date'].max().date()}")
        print(f"Returns available: {len(returns)}")
        
        # Step 5: Model Fitting
        print("\n" + "="*70)
        print("STEP 5: MODEL FITTING")
        print("="*70)
        
        self.ts_models.set_data(returns)
        self.ts_models.fit_all()
        
        for model_name in self.ts_models.models:
            print(f"Fitted: {model_name}")
        
        # Step 6: Performance Analysis
        print("\n" + "="*70)
        print("STEP 6: PERFORMANCE ANALYSIS")
        print("="*70)
        
        self.performance.set_models(self.ts_models)
        if self.testing_period:
            self.performance.set_testing_period(*self.testing_period)
        
        # Get crisis return for the selected crisis date
        crisis_return = None
        if self.selected_crisis_date is not None:
            full_data = self.data_cleaner.full_data.set_index('Date')
            if self.selected_crisis_date in full_data.index:
                crisis_return = full_data.loc[self.selected_crisis_date, 'Return']
        
        results_table = self.performance.evaluate_all(crisis_return=crisis_return)
        
        print("\nRESULTS TABLE:")
        print(results_table.to_string(index=False))
        
        # Step 7: Historical Crash Probability Analysis
        crash_results = self.performance.historical_crash_analysis(
            self.data_cleaner, 
            crisis_date=self.selected_crisis_date
        )
        
        # Step 8: VaR Backtesting
        print("\n" + "="*70)
        print("STEP 8: VAR BACKTESTING CONFIGURATION")
        print("="*70)
        
        print(f"\nUsing {self.rolling_window_years}-year rolling window")
        earliest_start = pd.Timestamp('2000-01-01') + pd.DateOffset(years=self.rolling_window_years)
        print(f"Earliest possible start date: {earliest_start.date()}")
        
        test_start_date = input(f"Enter backtesting start date (YYYY-MM-DD, e.g., {earliest_start.date()}): ").strip()
        if not test_start_date:
            test_start_date = str(earliest_start.date())
        
        max_date = self.data_cleaner.cleaned_data['Date'].max()
        test_end_date = input(f"Enter backtesting end date (YYYY-MM-DD, max: {max_date.date()}): ").strip()
        if not test_end_date:
            test_end_date = str(max_date.date())
        
        var_backtesting, var_data = self.performance.var_backtesting_custom(
            self.data_cleaner,
            rolling_window_years=self.rolling_window_years,
            test_start_date=test_start_date,
            test_end_date=test_end_date
        )
        
        # Step 9: Economic Significance Analysis
        if var_backtesting is not None:
            economic_results = self.performance.economic_significance_analysis()
        else:
            economic_results = None
        
        # Export results with period suffix
        if self.testing_period:
            start, end = self.testing_period
            period_suffix = f"_{start.replace('-','')}_{end.replace('-','')}"
        else:
            period_suffix = "_full"
        
        results_table.to_csv(f'model_results{period_suffix}.csv', index=False)
        self.data_cleaner.cleaned_data.to_csv(f'cleaned_data{period_suffix}.csv', index=False)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"Files saved with suffix: {period_suffix}")
        print("  - model_results.csv")
        if crash_results is not None:
            print("  - crash_analysis_results.csv")
        if var_backtesting is not None:
            print("  - var_backtesting_results.csv")
            print("  - economic_significance_results.csv")
            print("  - var_backtesting_plots.png")
        print("  - cleaned_data.csv")
        
        return results_table, crash_results, var_backtesting


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    system = TimeSeriesAnalysisSystem()
    results, crash_analysis, var_backtesting = system.run_analysis()