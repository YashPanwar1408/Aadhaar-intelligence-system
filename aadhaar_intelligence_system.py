"""
Aadhaar Stability & Service Load Intelligence System
====================================================
UIDAI Government Analytics Project

Author: Senior Data Scientist
Date: January 14, 2026

This system analyzes Aadhaar enrolment, demographic, and biometric update data
to compute the Aadhaar Stability Index (ASI) and predict future service load.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
import os
from pathlib import Path
import time

warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class AadhaarIntelligenceSystem:
    """
    Complete pipeline for Aadhaar data analysis and prediction.
    """
    
    def __init__(self, base_path):
        """
        Initialize the system with base directory path.
        
        Args:
            base_path (str): Base directory containing the three data folders
        """
        self.base_path = Path(base_path)
        self.enrolment_df = None
        self.demographic_df = None
        self.biometric_df = None
        self.master_df = None
        self.predictions_df = None
        
        # Create output directory
        self.output_dir = self.base_path / 'outputs'
        self.output_dir.mkdir(exist_ok=True)
        
        print("=" * 80)
        print("AADHAAR STABILITY & SERVICE LOAD INTELLIGENCE SYSTEM")
        print("=" * 80)
        print(f"Base Path: {self.base_path}")
        print(f"Output Directory: {self.output_dir}")
        print()
    
    
    # =========================================================================
    # SECTION 1: DATA LOADING
    # =========================================================================
    
    def load_all_datasets(self):
        """
        Load all CSV files from the three directories and combine them.
        """
        print("SECTION 1: DATA LOADING")
        print("-" * 80)
        
        # Load enrolment data
        print("Loading Enrolment Data...")
        enrolment_files = list((self.base_path / 'api_data_aadhar_enrolment').glob('*.csv'))
        enrolment_dfs = []
        for file in enrolment_files:
            df = pd.read_csv(file)
            enrolment_dfs.append(df)
            print(f"  ✓ Loaded {file.name}: {len(df):,} records")
        self.enrolment_df = pd.concat(enrolment_dfs, ignore_index=True)
        print(f"  Total Enrolment Records: {len(self.enrolment_df):,}\n")
        
        # Load demographic data
        print("Loading Demographic Update Data...")
        demographic_files = list((self.base_path / 'api_data_aadhar_demographic').glob('*.csv'))
        demographic_dfs = []
        for file in demographic_files:
            df = pd.read_csv(file)
            demographic_dfs.append(df)
            print(f"  ✓ Loaded {file.name}: {len(df):,} records")
        self.demographic_df = pd.concat(demographic_dfs, ignore_index=True)
        print(f"  Total Demographic Records: {len(self.demographic_df):,}\n")
        
        # Load biometric data
        print("Loading Biometric Update Data...")
        biometric_files = list((self.base_path / 'api_data_aadhar_biometric').glob('*.csv'))
        biometric_dfs = []
        for file in biometric_files:
            df = pd.read_csv(file)
            biometric_dfs.append(df)
            print(f"  ✓ Loaded {file.name}: {len(df):,} records")
        self.biometric_df = pd.concat(biometric_dfs, ignore_index=True)
        print(f"  Total Biometric Records: {len(self.biometric_df):,}\n")
        
        print("✓ All datasets loaded successfully!\n")
    
    
    # =========================================================================
    # SECTION 2: DATA CLEANING
    # =========================================================================
    
    def clean_and_standardize(self):
        """
        Clean and standardize all three datasets.
        """
        print("\nSECTION 2: DATA CLEANING & STANDARDIZATION")
        print("-" * 80)
        
        # Clean Enrolment Data
        print("Cleaning Enrolment Data...")
        self.enrolment_df.columns = self.enrolment_df.columns.str.strip().str.lower()
        self.enrolment_df['date'] = pd.to_datetime(self.enrolment_df['date'], format='%d-%m-%Y', errors='coerce')
        self.enrolment_df['state'] = self.enrolment_df['state'].str.strip()
        self.enrolment_df['district'] = self.enrolment_df['district'].str.strip()
        self.enrolment_df['pincode'] = self.enrolment_df['pincode'].astype(str).str.strip()
        
        # Ensure numeric columns
        for col in ['age_0_5', 'age_5_17', 'age_18_greater']:
            self.enrolment_df[col] = pd.to_numeric(self.enrolment_df[col], errors='coerce').fillna(0)
        
        # Remove duplicates
        before = len(self.enrolment_df)
        self.enrolment_df.drop_duplicates(subset=['date', 'state', 'district', 'pincode'], keep='first', inplace=True)
        after = len(self.enrolment_df)
        print(f"  ✓ Removed {before - after:,} duplicate records")
        print(f"  ✓ Final Enrolment Records: {after:,}\n")
        
        # Clean Demographic Data
        print("Cleaning Demographic Update Data...")
        self.demographic_df.columns = self.demographic_df.columns.str.strip().str.lower()
        self.demographic_df['date'] = pd.to_datetime(self.demographic_df['date'], format='%d-%m-%Y', errors='coerce')
        self.demographic_df['state'] = self.demographic_df['state'].str.strip()
        self.demographic_df['district'] = self.demographic_df['district'].str.strip()
        self.demographic_df['pincode'] = self.demographic_df['pincode'].astype(str).str.strip()
        
        # Ensure numeric columns
        for col in ['demo_age_5_17', 'demo_age_17_']:
            self.demographic_df[col] = pd.to_numeric(self.demographic_df[col], errors='coerce').fillna(0)
        
        before = len(self.demographic_df)
        self.demographic_df.drop_duplicates(subset=['date', 'state', 'district', 'pincode'], keep='first', inplace=True)
        after = len(self.demographic_df)
        print(f"  ✓ Removed {before - after:,} duplicate records")
        print(f"  ✓ Final Demographic Records: {after:,}\n")
        
        # Clean Biometric Data
        print("Cleaning Biometric Update Data...")
        self.biometric_df.columns = self.biometric_df.columns.str.strip().str.lower()
        self.biometric_df['date'] = pd.to_datetime(self.biometric_df['date'], format='%d-%m-%Y', errors='coerce')
        self.biometric_df['state'] = self.biometric_df['state'].str.strip()
        self.biometric_df['district'] = self.biometric_df['district'].str.strip()
        self.biometric_df['pincode'] = self.biometric_df['pincode'].astype(str).str.strip()
        
        # Ensure numeric columns
        for col in ['bio_age_5_17', 'bio_age_17_']:
            self.biometric_df[col] = pd.to_numeric(self.biometric_df[col], errors='coerce').fillna(0)
        
        before = len(self.biometric_df)
        self.biometric_df.drop_duplicates(subset=['date', 'state', 'district', 'pincode'], keep='first', inplace=True)
        after = len(self.biometric_df)
        print(f"  ✓ Removed {before - after:,} duplicate records")
        print(f"  ✓ Final Biometric Records: {after:,}\n")
        
        print("✓ All datasets cleaned and standardized!\n")
    
    
    # =========================================================================
    # SECTION 3: DATA MERGING
    # =========================================================================
    
    def merge_datasets(self):
        """
        Merge all three datasets into a master dataset.
        """
        print("\nSECTION 3: DATA MERGING")
        print("-" * 80)
        
        merge_keys = ['date', 'state', 'district', 'pincode']
        
        # Start with enrolment data
        print("Merging datasets using outer joins...")
        self.master_df = self.enrolment_df.copy()
        
        # Merge with demographic data
        self.master_df = self.master_df.merge(
            self.demographic_df,
            on=merge_keys,
            how='outer',
            suffixes=('', '_demo')
        )
        print(f"  ✓ After demographic merge: {len(self.master_df):,} records")
        
        # Merge with biometric data
        self.master_df = self.master_df.merge(
            self.biometric_df,
            on=merge_keys,
            how='outer',
            suffixes=('', '_bio')
        )
        print(f"  ✓ After biometric merge: {len(self.master_df):,} records")
        
        # Fill missing values with 0 for numeric columns
        numeric_cols = self.master_df.select_dtypes(include=[np.number]).columns
        self.master_df[numeric_cols] = self.master_df[numeric_cols].fillna(0)
        
        # Drop rows with missing date
        self.master_df.dropna(subset=['date'], inplace=True)
        
        print(f"  ✓ Final Master Dataset: {len(self.master_df):,} records")
        print(f"  ✓ Columns: {list(self.master_df.columns)}\n")
        
        print("✓ Datasets merged successfully!\n")
    
    
    # =========================================================================
    # SECTION 4: FEATURE ENGINEERING
    # =========================================================================
    
    def engineer_features(self):
        """
        Create meaningful features including the Aadhaar Stability Index (ASI).
        """
        print("\nSECTION 4: FEATURE ENGINEERING")
        print("-" * 80)
        
        # Total enrolments by age group
        self.master_df['total_enrolments'] = (
            self.master_df['age_0_5'] + 
            self.master_df['age_5_17'] + 
            self.master_df['age_18_greater']
        )
        print("  ✓ Created: total_enrolments")
        
        # Total demographic updates
        self.master_df['total_demo_updates'] = (
            self.master_df['demo_age_5_17'] + 
            self.master_df['demo_age_17_']
        )
        print("  ✓ Created: total_demo_updates")
        
        # Total biometric updates
        self.master_df['total_bio_updates'] = (
            self.master_df['bio_age_5_17'] + 
            self.master_df['bio_age_17_']
        )
        print("  ✓ Created: total_bio_updates")
        
        # Total updates (demographic + biometric)
        self.master_df['total_updates'] = (
            self.master_df['total_demo_updates'] + 
            self.master_df['total_bio_updates']
        )
        print("  ✓ Created: total_updates")
        
        # Update ratio
        self.master_df['update_ratio'] = np.where(
            self.master_df['total_enrolments'] > 0,
            self.master_df['total_updates'] / self.master_df['total_enrolments'],
            0
        )
        print("  ✓ Created: update_ratio")
        
        # =====================================================================
        # AADHAAR STABILITY INDEX (ASI)
        # =====================================================================
        # ASI = 1 - (Total Updates / Total Enrolments)
        # High ASI → Stable Aadhaar records
        # Low ASI → Poor data quality, high rework
        
        self.master_df['asi'] = 1 - self.master_df['update_ratio']
        self.master_df['asi'] = self.master_df['asi'].clip(lower=0, upper=1)  # Ensure between 0 and 1
        print("  ✓ Created: asi (Aadhaar Stability Index)")
        
        # Log-transformed features (for better ML performance)
        self.master_df['log_enrolments'] = np.log1p(self.master_df['total_enrolments'])
        self.master_df['log_updates'] = np.log1p(self.master_df['total_updates'])
        print("  ✓ Created: log_enrolments, log_updates")
        
        # Date features
        self.master_df['year'] = self.master_df['date'].dt.year
        self.master_df['month'] = self.master_df['date'].dt.month
        self.master_df['day_of_week'] = self.master_df['date'].dt.dayofweek
        print("  ✓ Created: year, month, day_of_week")
        
        print(f"\n  Total Features: {len(self.master_df.columns)}")
        print("✓ Feature engineering completed!\n")
    
    
    # =========================================================================
    # SECTION 5: EXPLORATORY DATA ANALYSIS (EDA)
    # =========================================================================
    
    def perform_eda(self):
        """
        Generate comprehensive exploratory data analysis and visualizations.
        """
        print("\nSECTION 5: EXPLORATORY DATA ANALYSIS (EDA)")
        print("-" * 80)
        
        # Calculate district-level aggregations
        district_stats = self.master_df.groupby('district').agg({
            'total_enrolments': 'sum',
            'total_bio_updates': 'sum',
            'total_demo_updates': 'sum',
            'asi': 'mean'
        }).reset_index()
        
        district_stats = district_stats.sort_values('total_enrolments', ascending=False).head(20)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        
        # 1. District-wise Enrolment Bar Chart
        ax1 = plt.subplot(2, 3, 1)
        sns.barplot(data=district_stats, x='total_enrolments', y='district', palette='viridis', ax=ax1)
        ax1.set_title('Top 20 Districts by Total Enrolments', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Total Enrolments', fontsize=12)
        ax1.set_ylabel('District', fontsize=12)
        ax1.ticklabel_format(style='plain', axis='x')
        
        # 2. Biometric Updates by District
        ax2 = plt.subplot(2, 3, 2)
        sns.barplot(data=district_stats, x='total_bio_updates', y='district', palette='Reds_r', ax=ax2)
        ax2.set_title('Top 20 Districts by Biometric Updates', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Total Biometric Updates', fontsize=12)
        ax2.set_ylabel('District', fontsize=12)
        ax2.ticklabel_format(style='plain', axis='x')
        
        # 3. Demographic Updates by District
        ax3 = plt.subplot(2, 3, 3)
        sns.barplot(data=district_stats, x='total_demo_updates', y='district', palette='Blues_r', ax=ax3)
        ax3.set_title('Top 20 Districts by Demographic Updates', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Total Demographic Updates', fontsize=12)
        ax3.set_ylabel('District', fontsize=12)
        ax3.ticklabel_format(style='plain', axis='x')
        
        # 4. Date-wise Total Aadhaar Activity
        ax4 = plt.subplot(2, 3, 4)
        date_stats = self.master_df.groupby('date').agg({
            'total_enrolments': 'sum',
            'total_updates': 'sum'
        }).reset_index()
        ax4.plot(date_stats['date'], date_stats['total_enrolments'], label='Enrolments', linewidth=2)
        ax4.plot(date_stats['date'], date_stats['total_updates'], label='Updates', linewidth=2)
        ax4.set_title('Date-wise Aadhaar Activity Trend', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Date', fontsize=12)
        ax4.set_ylabel('Count', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        # 5. ASI Comparison Across Districts
        ax5 = plt.subplot(2, 3, 5)
        asi_sorted = district_stats.sort_values('asi', ascending=False).head(20)
        colors = ['green' if x > 0.7 else 'orange' if x > 0.5 else 'red' for x in asi_sorted['asi']]
        sns.barplot(data=asi_sorted, x='asi', y='district', palette=colors, ax=ax5)
        ax5.set_title('Aadhaar Stability Index (ASI) by District', fontsize=14, fontweight='bold')
        ax5.set_xlabel('ASI (Higher = More Stable)', fontsize=12)
        ax5.set_ylabel('District', fontsize=12)
        ax5.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, label='High Stability')
        ax5.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium Stability')
        ax5.legend()
        
        # 6. Update Distribution
        ax6 = plt.subplot(2, 3, 6)
        update_data = pd.DataFrame({
            'Type': ['Demographic', 'Biometric'],
            'Count': [
                self.master_df['total_demo_updates'].sum(),
                self.master_df['total_bio_updates'].sum()
            ]
        })
        ax6.pie(update_data['Count'], labels=update_data['Type'], autopct='%1.1f%%', 
                colors=['#3498db', '#e74c3c'], startangle=90, textprops={'fontsize': 12})
        ax6.set_title('Distribution of Update Types', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        eda_path = self.output_dir / 'eda_comprehensive_analysis.png'
        plt.savefig(eda_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {eda_path}")
        plt.close()
        
        # Additional Analysis: State-level Statistics
        state_stats = self.master_df.groupby('state').agg({
            'total_enrolments': 'sum',
            'total_updates': 'sum',
            'asi': 'mean'
        }).reset_index()
        state_stats = state_stats.sort_values('total_enrolments', ascending=False).head(15)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(state_stats))
        width = 0.35
        
        ax.bar(x - width/2, state_stats['total_enrolments'], width, label='Enrolments', color='#2ecc71')
        ax.bar(x + width/2, state_stats['total_updates'], width, label='Updates', color='#e74c3c')
        
        ax.set_xlabel('State', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('State-wise Enrolments vs Updates (Top 15 States)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(state_stats['state'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        state_path = self.output_dir / 'state_wise_analysis.png'
        plt.savefig(state_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {state_path}")
        plt.close()
        
        print("✓ EDA completed and visualizations saved!\n")
    
    
    # =========================================================================
    # SECTION 6: ANOMALY DETECTION
    # =========================================================================
    
    def detect_anomalies(self):
        """
        Identify unstable regions and anomalies in the data.
        """
        print("\nSECTION 6: ANOMALY DETECTION")
        print("-" * 80)
        
        # 1. Districts with Lowest ASI (Most Unstable)
        print("1. Top 10 Districts with LOWEST ASI (Most Unstable):")
        district_asi = self.master_df.groupby('district')['asi'].mean().reset_index()
        district_asi = district_asi.sort_values('asi').head(10)
        print(district_asi.to_string(index=False))
        print()
        
        # 2. PIN Codes with Highest Update Ratio
        print("2. Top 10 PIN Codes with HIGHEST Update Ratio:")
        pincode_ratio = self.master_df.groupby('pincode')['update_ratio'].mean().reset_index()
        pincode_ratio = pincode_ratio.sort_values('update_ratio', ascending=False).head(10)
        print(pincode_ratio.to_string(index=False))
        print()
        
        # 3. Age Groups Causing Most Rework
        print("3. Age Groups Causing Most Updates (Rework Analysis):")
        age_analysis = pd.DataFrame({
            'Age Group': ['5-17', '17+'],
            'Demographic Updates': [
                self.master_df['demo_age_5_17'].sum(),
                self.master_df['demo_age_17_'].sum()
            ],
            'Biometric Updates': [
                self.master_df['bio_age_5_17'].sum(),
                self.master_df['bio_age_17_'].sum()
            ]
        })
        age_analysis['Total Updates'] = age_analysis['Demographic Updates'] + age_analysis['Biometric Updates']
        print(age_analysis.to_string(index=False))
        print()
        
        # 4. States with Highest Instability
        print("4. Top 5 States with LOWEST Average ASI:")
        state_asi = self.master_df.groupby('state')['asi'].mean().reset_index()
        state_asi = state_asi.sort_values('asi').head(5)
        print(state_asi.to_string(index=False))
        print()
        
        # Save anomaly report
        anomaly_report = {
            'Unstable Districts': district_asi,
            'High Update PIN Codes': pincode_ratio,
            'Age Group Analysis': age_analysis,
            'Unstable States': state_asi
        }
        
        with pd.ExcelWriter(self.output_dir / 'anomaly_detection_report.xlsx') as writer:
            for sheet_name, df in anomaly_report.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"  ✓ Saved: {self.output_dir / 'anomaly_detection_report.xlsx'}")
        print("✓ Anomaly detection completed!\n")
        
        return anomaly_report
    
    
    # =========================================================================
    # SECTION 7: MACHINE LEARNING MODELS COMPARISON
    # =========================================================================
    
    def build_ml_model(self):
        """
        Build and compare multiple ML models to predict future biometric update load.
        """
        print("\nSECTION 7: MACHINE LEARNING MODELS COMPARISON")
        print("=" * 80)
        
        print("Target Variable: bio_age_17_ (Biometric updates for age 17+)")
        print("Comparing Multiple Regression Models\n")
        
        # Prepare features and target
        # Remove rows where target is 0 to focus on actual update patterns
        ml_df = self.master_df[self.master_df['bio_age_17_'] > 0].copy()
        
        feature_cols = [
            'age_0_5', 'age_5_17', 'age_18_greater',
            'total_enrolments', 'total_demo_updates', 
            'total_updates', 'update_ratio', 'asi',
            'log_enrolments', 'log_updates',
            'month', 'day_of_week'
        ]
        
        X = ml_df[feature_cols].fillna(0)
        y = ml_df['bio_age_17_']
        
        print(f"Training Dataset Size: {len(X):,} records")
        print(f"Features: {len(feature_cols)}")
        print(f"Feature List: {feature_cols}\n")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training Set: {len(X_train):,} | Test Set: {len(X_test):,}\n")
        
        # Initialize models dictionary
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=10, min_samples_leaf=5, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42),
            'Linear Regression': LinearRegression(n_jobs=-1),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Lasso Regression': Lasso(alpha=1.0, random_state=42),
            'Decision Tree': DecisionTreeRegressor(max_depth=15, min_samples_split=10, min_samples_leaf=5, random_state=42)
        }
        
        # Train and evaluate all models
        results = []
        trained_models = {}
        
        print("Training and evaluating models...")
        print("-" * 80)
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            
            # Train model
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            # Store results
            results.append({
                'Model': name,
                'Train MAE': train_mae,
                'Test MAE': test_mae,
                'Train R²': train_r2,
                'Test R²': test_r2,
                'Train RMSE': train_rmse,
                'Test RMSE': test_rmse,
                'Training Time (s)': training_time
            })
            
            trained_models[name] = {
                'model': model,
                'predictions': y_test_pred
            }
            
            print(f"✓ {name} completed in {training_time:.2f}s")
            print(f"  Test R²: {test_r2:.4f} | Test MAE: {test_mae:.2f} | Test RMSE: {test_rmse:.2f}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Find best model based on Test R²
        best_idx = results_df['Test R²'].idxmax()
        best_model_name = results_df.loc[best_idx, 'Model']
        best_model = trained_models[best_model_name]['model']
        best_y_pred = trained_models[best_model_name]['predictions']
        
        print("\n" + "=" * 80)
        print(f"BEST MODEL: {best_model_name}")
        print(f"Test R² Score: {results_df.loc[best_idx, 'Test R²']:.4f}")
        print(f"Test MAE: {results_df.loc[best_idx, 'Test MAE']:.2f}")
        print(f"Test RMSE: {results_df.loc[best_idx, 'Test RMSE']:.2f}")
        print("=" * 80)
        
        # Display comparison table
        print("\n" + "MODEL COMPARISON TABLE:")
        print("-" * 80)
        print(results_df.to_string(index=False))
        print("-" * 80 + "\n")
        
        # Feature importance (from best model if it supports it)
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("FEATURE IMPORTANCE (Top 10):")
            print("-" * 40)
            print(feature_importance.head(10).to_string(index=False))
            print()
        else:
            feature_importance = None
            print("Note: Best model does not support feature importance.\n")
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Test R² Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        results_sorted = results_df.sort_values('Test R²', ascending=False)
        colors = ['#2ecc71' if m == best_model_name else '#3498db' for m in results_sorted['Model']]
        ax1.barh(results_sorted['Model'], results_sorted['Test R²'], color=colors)
        ax1.set_xlabel('Test R² Score', fontsize=11, fontweight='bold')
        ax1.set_title('Model Comparison: Test R² Score', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        for i, v in enumerate(results_sorted['Test R²']):
            ax1.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9)
        
        # 2. Test MAE Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        results_sorted_mae = results_df.sort_values('Test MAE')
        colors_mae = ['#2ecc71' if m == best_model_name else '#e74c3c' for m in results_sorted_mae['Model']]
        ax2.barh(results_sorted_mae['Model'], results_sorted_mae['Test MAE'], color=colors_mae)
        ax2.set_xlabel('Test MAE (Lower is Better)', fontsize=11, fontweight='bold')
        ax2.set_title('Model Comparison: Test MAE', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        for i, v in enumerate(results_sorted_mae['Test MAE']):
            ax2.text(v + max(results_sorted_mae['Test MAE'])*0.01, i, f'{v:.2f}', va='center', fontsize=9)
        
        # 3. Test RMSE Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        results_sorted_rmse = results_df.sort_values('Test RMSE')
        colors_rmse = ['#2ecc71' if m == best_model_name else '#f39c12' for m in results_sorted_rmse['Model']]
        ax3.barh(results_sorted_rmse['Model'], results_sorted_rmse['Test RMSE'], color=colors_rmse)
        ax3.set_xlabel('Test RMSE (Lower is Better)', fontsize=11, fontweight='bold')
        ax3.set_title('Model Comparison: Test RMSE', fontsize=12, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        for i, v in enumerate(results_sorted_rmse['Test RMSE']):
            ax3.text(v + max(results_sorted_rmse['Test RMSE'])*0.01, i, f'{v:.2f}', va='center', fontsize=9)
        
        # 4. Training Time Comparison
        ax4 = fig.add_subplot(gs[1, 0])
        results_sorted_time = results_df.sort_values('Training Time (s)')
        colors_time = ['#2ecc71' if m == best_model_name else '#9b59b6' for m in results_sorted_time['Model']]
        ax4.barh(results_sorted_time['Model'], results_sorted_time['Training Time (s)'], color=colors_time)
        ax4.set_xlabel('Training Time (seconds)', fontsize=11, fontweight='bold')
        ax4.set_title('Model Comparison: Training Time', fontsize=12, fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
        for i, v in enumerate(results_sorted_time['Training Time (s)']):
            ax4.text(v + max(results_sorted_time['Training Time (s)'])*0.01, i, f'{v:.2f}s', va='center', fontsize=9)
        
        # 5. Metrics Heatmap
        ax5 = fig.add_subplot(gs[1, 1:])
        metrics_for_heatmap = results_df[['Model', 'Test R²', 'Test MAE', 'Test RMSE', 'Training Time (s)']].set_index('Model')
        # Normalize for better visualization
        metrics_normalized = (metrics_for_heatmap - metrics_for_heatmap.min()) / (metrics_for_heatmap.max() - metrics_for_heatmap.min())
        sns.heatmap(metrics_normalized.T, annot=metrics_for_heatmap.T, fmt='.3f', cmap='RdYlGn', 
                    cbar_kws={'label': 'Normalized Score'}, ax=ax5, linewidths=0.5)
        ax5.set_title('All Metrics Comparison (Normalized)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('')
        ax5.set_ylabel('Metrics', fontsize=11, fontweight='bold')
        
        # 6. Actual vs Predicted (Best Model)
        ax6 = fig.add_subplot(gs[2, :2])
        sample_size = min(1000, len(y_test))
        sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
        ax6.scatter(y_test.iloc[sample_indices], best_y_pred[sample_indices], 
                    alpha=0.5, s=15, color='#3498db', edgecolors='white', linewidth=0.3)
        ax6.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 'r--', lw=2, label='Perfect Prediction')
        ax6.set_xlabel('Actual Bio Updates (17+)', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Predicted Bio Updates (17+)', fontsize=11, fontweight='bold')
        best_test_r2 = results_df.loc[best_idx, 'Test R²']
        best_test_mae = results_df.loc[best_idx, 'Test MAE']
        ax6.set_title(f'Best Model ({best_model_name}): Actual vs Predicted\nR² = {best_test_r2:.4f}, MAE = {best_test_mae:.2f}', 
                      fontsize=12, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        # 7. Feature Importance (if available)
        ax7 = fig.add_subplot(gs[2, 2])
        if feature_importance is not None:
            sns.barplot(data=feature_importance.head(10), x='Importance', y='Feature', 
                        palette='viridis', ax=ax7)
            ax7.set_title(f'Top 10 Feature Importance\n({best_model_name})', fontsize=12, fontweight='bold')
            ax7.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
            ax7.set_ylabel('')
        else:
            ax7.text(0.5, 0.5, 'Feature Importance\nNot Available\nfor this model', 
                     ha='center', va='center', fontsize=12, transform=ax7.transAxes)
            ax7.axis('off')
        
        plt.suptitle('MACHINE LEARNING MODELS COMPARISON - Complete Analysis', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        comparison_path = self.output_dir / 'ml_models_comparison.png'
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved comprehensive comparison: {comparison_path}")
        plt.close()
        
        # Save comparison results to CSV
        results_csv_path = self.output_dir / 'ml_models_comparison_results.csv'
        results_df.to_csv(results_csv_path, index=False)
        print(f"✓ Saved comparison results: {results_csv_path}")
        
        print("\n✓ Machine learning models comparison completed!\n")
        
        return best_model, feature_cols, feature_importance, results_df
    
    
    # =========================================================================
    # SECTION 8: PREDICTIONS
    # =========================================================================
    
    def generate_predictions(self, model, feature_cols):
        """
        Generate predictions for the entire dataset.
        """
        print("\nSECTION 8: PREDICTION GENERATION")
        print("-" * 80)
        
        print("Generating predictions for all records...")
        
        # Prepare features
        X_all = self.master_df[feature_cols].fillna(0)
        
        # Generate predictions
        predictions = model.predict(X_all)
        
        # Add to master dataframe
        self.master_df['predicted_bio_load'] = predictions
        self.master_df['predicted_bio_load'] = self.master_df['predicted_bio_load'].clip(lower=0)
        
        print(f"  ✓ Predictions generated for {len(self.master_df):,} records")
        print(f"  ✓ Prediction Statistics:")
        print(f"     - Mean Predicted Load: {predictions.mean():.2f}")
        print(f"     - Median Predicted Load: {np.median(predictions):.2f}")
        print(f"     - Max Predicted Load: {predictions.max():.2f}")
        print(f"     - Min Predicted Load: {predictions.min():.2f}\n")
        
        # Create predictions summary by district
        district_predictions = self.master_df.groupby('district').agg({
            'predicted_bio_load': 'sum',
            'total_enrolments': 'sum',
            'asi': 'mean'
        }).reset_index()
        
        district_predictions = district_predictions.sort_values('predicted_bio_load', ascending=False).head(20)
        
        print("TOP 20 DISTRICTS BY PREDICTED BIOMETRIC LOAD:")
        print("-" * 60)
        print(district_predictions.to_string(index=False))
        print()
        
        # Visualize predictions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Top districts by predicted load
        sns.barplot(data=district_predictions, x='predicted_bio_load', y='district', 
                    palette='Reds_r', ax=ax1)
        ax1.set_title('Top 20 Districts by Predicted Biometric Service Load', 
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted Biometric Load', fontsize=12)
        ax1.set_ylabel('District', fontsize=12)
        
        # Correlation: ASI vs Predicted Load
        scatter_data = self.master_df[self.master_df['predicted_bio_load'] > 0].sample(min(5000, len(self.master_df)))
        ax2.scatter(scatter_data['asi'], scatter_data['predicted_bio_load'], 
                    alpha=0.3, s=20, color='#e74c3c')
        ax2.set_xlabel('Aadhaar Stability Index (ASI)', fontsize=12)
        ax2.set_ylabel('Predicted Biometric Load', fontsize=12)
        ax2.set_title('ASI vs Predicted Service Load\n(Lower ASI → Higher Predicted Load)', 
                      fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pred_path = self.output_dir / 'predictions_analysis.png'
        plt.savefig(pred_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {pred_path}")
        plt.close()
        
        print("✓ Prediction generation completed!\n")
        
        return district_predictions
    
    
    # =========================================================================
    # SECTION 9: SAVE OUTPUTS
    # =========================================================================
    
    def save_outputs(self):
        """
        Save all outputs including cleaned data, master dataset, and predictions.
        """
        print("\nSECTION 9: SAVING OUTPUTS")
        print("-" * 80)
        
        # 1. Save cleaned enrolment data
        enrolment_path = self.output_dir / 'cleaned_enrolment_data.csv'
        self.enrolment_df.to_csv(enrolment_path, index=False)
        print(f"  ✓ Saved: {enrolment_path}")
        
        # 2. Save cleaned demographic data
        demographic_path = self.output_dir / 'cleaned_demographic_data.csv'
        self.demographic_df.to_csv(demographic_path, index=False)
        print(f"  ✓ Saved: {demographic_path}")
        
        # 3. Save cleaned biometric data
        biometric_path = self.output_dir / 'cleaned_biometric_data.csv'
        self.biometric_df.to_csv(biometric_path, index=False)
        print(f"  ✓ Saved: {biometric_path}")
        
        # 4. Save master dataset
        master_path = self.output_dir / 'master_dataset_with_asi.csv'
        self.master_df.to_csv(master_path, index=False)
        print(f"  ✓ Saved: {master_path}")
        
        # 5. Save predictions only
        predictions_cols = [
            'date', 'state', 'district', 'pincode',
            'total_enrolments', 'total_updates', 'asi',
            'predicted_bio_load'
        ]
        predictions_df = self.master_df[predictions_cols].copy()
        predictions_path = self.output_dir / 'predictions_biometric_load.csv'
        predictions_df.to_csv(predictions_path, index=False)
        print(f"  ✓ Saved: {predictions_path}")
        
        # 6. Create summary statistics
        summary_stats = {
            'Total Records': len(self.master_df),
            'Total Districts': self.master_df['district'].nunique(),
            'Total States': self.master_df['state'].nunique(),
            'Total PIN Codes': self.master_df['pincode'].nunique(),
            'Date Range': f"{self.master_df['date'].min()} to {self.master_df['date'].max()}",
            'Total Enrolments': self.master_df['total_enrolments'].sum(),
            'Total Updates': self.master_df['total_updates'].sum(),
            'Average ASI': self.master_df['asi'].mean(),
            'Total Predicted Bio Load': self.master_df['predicted_bio_load'].sum()
        }
        
        summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
        summary_path = self.output_dir / 'summary_statistics.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"  ✓ Saved: {summary_path}")
        
        print("\n✓ All outputs saved successfully!\n")
    
    
    # =========================================================================
    # MAIN EXECUTION PIPELINE
    # =========================================================================
    
    def run_complete_pipeline(self):
        """
        Execute the complete end-to-end pipeline.
        """
        try:
            # Step 1: Load data
            self.load_all_datasets()
            
            # Step 2: Clean data
            self.clean_and_standardize()
            
            # Step 3: Merge datasets
            self.merge_datasets()
            
            # Step 4: Feature engineering
            self.engineer_features()
            
            # Step 5: EDA
            self.perform_eda()
            
            # Step 6: Anomaly detection
            self.detect_anomalies()
            
            # Step 7: Build ML model (now returns 4 values)
            model, feature_cols, feature_importance, results_df = self.build_ml_model()
            
            # Step 8: Generate predictions
            self.generate_predictions(model, feature_cols)
            
            # Step 9: Save outputs
            self.save_outputs()
            
            print("=" * 80)
            print("AADHAAR INTELLIGENCE SYSTEM - PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"\n📁 All outputs saved in: {self.output_dir}")
            print("\n✅ DELIVERABLES:")
            print("   1. Cleaned datasets (3 files)")
            print("   2. Master dataset with ASI")
            print("   3. Predictions dataset")
            print("   4. EDA visualizations")
            print("   5. Anomaly detection report")
            print("   6. ML models comparison analysis")
            print("   7. Summary statistics")
            print("\n🎯 KEY INSIGHTS:")
            print(f"   • Total Records Analyzed: {len(self.master_df):,}")
            print(f"   • Average ASI: {self.master_df['asi'].mean():.4f}")
            print(f"   • Districts Covered: {self.master_df['district'].nunique()}")
            print(f"   • Best ML Model: {results_df.loc[results_df['Test R²'].idxmax(), 'Model']}")
            print(f"   • Best Model Test R²: {results_df['Test R²'].max():.4f}")
            print(f"   • Predicted Future Bio Load: {self.master_df['predicted_bio_load'].sum():,.0f}")
            print("\n" + "=" * 80 + "\n")
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Initialize the system
    base_directory = r"d:\uidai hack"
    
    system = AadhaarIntelligenceSystem(base_directory)
    
    # Run the complete pipeline
    system.run_complete_pipeline()
