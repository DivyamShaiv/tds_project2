# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scikit-learn", "chardet", "requests", "seaborn", "matplotlib", "python-dotenv"]
# ///

import os
import sys
import json
import logging
import subprocess
from typing import Dict, Any, List, Optional

# Automated dependency management
REQUIRED_PACKAGES = [
    "numpy", "pandas", "scikit-learn", "chardet", 
    "requests", "seaborn", "matplotlib", "python-dotenv"
]

def install_dependencies(packages):
    """
    Install required Python packages safely.
    
    Args:
        packages (List[str]): List of package names to install
    """
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install '{package}'. Error: {e}")
            sys.exit(1)

# Install dependencies before importing
install_dependencies(REQUIRED_PACKAGES)

import pandas as pd
import numpy as np
import chardet
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from dotenv import load_dotenv

class AdvancedDataAnalyzer:
    """
    A comprehensive data analysis framework with advanced analytical, 
    visualization, and narrative generation capabilities.
    """
    
    def __init__(self, 
                 dataset_path: str, 
                 api_key: str, 
                 log_level: int = logging.INFO):
        """
        Initialize the data analysis framework.
        
        Args:
            dataset_path (str): Path to the input dataset
            api_key (str): API key for LLM services
            log_level (int): Logging verbosity level
        """
        self.dataset_path = dataset_path
        self.api_key = api_key
        self.df = None
        self.analysis_results = {}
        
        # Advanced logging configuration
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s | %(levelname)8s | %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('data_analysis.log')
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Output management
        self.output_dir = self._create_output_directory()
    
    def _create_output_directory(self) -> str:
        """
        Create a structured output directory for analysis artifacts.
        
        Returns:
            str: Path to the created output directory
        """
        base_name = os.path.splitext(os.path.basename(self.dataset_path))[0]
        output_path = os.path.join('analysis_outputs', base_name)
        os.makedirs(output_path, exist_ok=True)
        return output_path
    
    def load_data(self) -> pd.DataFrame:
        """
        Robustly load data with encoding detection and preprocessing.
        
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        try:
            # Detect encoding
            with open(self.dataset_path, 'rb') as file:
                raw_data = file.read()
                encoding = chardet.detect(raw_data)['encoding']
            
            # Load with detected encoding
            self.df = pd.read_csv(self.dataset_path, encoding=encoding)
            
            # Basic preprocessing
            self.df.dropna(how='all', inplace=True)  # Drop entirely empty rows
            self.df.reset_index(drop=True, inplace=True)
            
            self.logger.info(f"Loaded dataset: {self.df.shape}")
            return self.df
        
        except Exception as e:
            self.logger.error(f"Data loading error: {e}")
            raise
    
    def advanced_preprocessing(self):
        """
        Advanced data preprocessing with multiple techniques.
        """
        # Imputation strategies
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        # Numeric imputation
        numeric_imputer = SimpleImputer(strategy='median')
        self.df[numeric_cols] = numeric_imputer.fit_transform(self.df[numeric_cols])
        
        # Scale numeric features
        scaler = StandardScaler()
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        
        return self
    
    def feature_importance(self):
        """
        Calculate feature importance using mutual information.
        
        Returns:
            Dict: Feature importance scores
        """
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        # Placeholder for target - in real-world, this would be specified
        target_proxy = self.df[numeric_cols].mean(axis=1)
        
        importances = mutual_info_classif(
            self.df[numeric_cols], 
            target_proxy
        )
        
        importance_dict = dict(zip(numeric_cols, importances))
        self.analysis_results['feature_importances'] = importance_dict
        
        return importance_dict
    
    def generate_visualizations(self):
        """
        Create a comprehensive set of visualizations with advanced styling.
        """
        plt.style.use('seaborn')
        
        # Correlation Heatmap
        plt.figure(figsize=(12, 10))
        corr_matrix = self.df.corr()
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            linewidths=0.5,
            fmt=".2f",
            square=True,
            cbar_kws={"shrink": .8}
        )
        plt.title('Feature Correlation Matrix', fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_heatmap.png'))
        plt.close()
        
        # PCA Visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.df.select_dtypes(include=['number']))
        
        plt.figure(figsize=(10, 8))
        plt.scatter(
            pca_result[:, 0], 
            pca_result[:, 1], 
            alpha=0.7,
            c=pca_result[:, 0],  # Color by first principal component
            cmap='viridis'
        )
        plt.title('PCA Visualization', fontsize=15)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(label='PC1 Value')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pca_visualization.png'))
        plt.close()
        
        return self
    
    def generate_narrative_prompt(self) -> Dict[str, Any]:
        """
        Create a structured, context-rich prompt for narrative generation.
        
        Returns:
            Dict: Comprehensive data context for LLM
        """
        narrative_context = {
            "dataset_metadata": {
                "name": os.path.basename(self.dataset_path),
                "rows": self.df.shape[0],
                "columns": self.df.shape[1],
                "column_types": {col: str(dtype) for col, dtype in self.df.dtypes.items()}
            },
            "statistical_summary": {
                "numeric_summary": self.df.describe().to_dict(),
                "feature_importances": self.analysis_results.get('feature_importances', {})
            },
            "key_observations": {
                "top_correlated_features": dict(sorted(
                    {k: v for k, v in self.df.corr().unstack().items() 
                     if abs(v) > 0.5 and k != v}, 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )[:5])
            }
        }
        
        return narrative_context
    
    def generate_llm_narrative(self, context: Dict[str, Any]) -> str:
        """
        Generate a narrative using an LLM with efficient token usage.
        
        Args:
            context (Dict): Prepared data context
        
        Returns:
            str: Generated narrative in Markdown
        """
        try:
            endpoint = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": """You are a data storyteller. Create a compelling, 
                        structured narrative that:
                        1. Introduces the dataset's context
                        2. Highlights key statistical insights
                        3. Explains significant patterns and correlations
                        4. Provides actionable recommendations
                        5. Uses clear, engaging Markdown formatting"""
                    },
                    {
                        "role": "user",
                        "content": json.dumps(context, indent=2)
                    }
                ],
                "max_tokens": 1000  # Efficient token management
            }
            
            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            
            return response.json()['choices'][0]['message']['content']
        
        except Exception as e:
            self.logger.error(f"Narrative generation failed: {e}")
            return "## Data Story Generation Error\n\nUnable to generate narrative."
    
    def generate_comprehensive_report(self):
        """
        Orchestrate the entire analysis workflow.
        """
        try:
            (self.load_data()
                .advanced_preprocessing()
                .feature_importance()
                .generate_visualizations())
            
            narrative_context = self.generate_narrative_prompt()
            narrative = self.generate_llm_narrative(narrative_context)
            
            # Combine narrative with visualizations
            full_report = f"""# Data Analysis Report

{narrative}

## Visualizations

### Correlation Heatmap
![Correlation Heatmap](correlation_heatmap.png)

### Principal Component Analysis
![PCA Visualization](pca_visualization.png)
"""
            
            report_path = os.path.join(self.output_dir, 'comprehensive_report.md')
            with open(report_path, 'w') as f:
                f.write(full_report)
            
            self.logger.info(f"Comprehensive report generated at {report_path}")
        
        except Exception as e:
            self.logger.error(f"Comprehensive report generation failed: {e}")

def main():
    """
    Main script execution point with robust error handling.
    """
    if len(sys.argv) < 2:
        print("Usage: python script.py <dataset.csv>")
        sys.exit(1)
    
    load_dotenv()  # Load environment variables
    
    try:
        api_key = os.environ["AIPROXY_TOKEN"]
        dataset_path = sys.argv[1]
        
        analyzer = AdvancedDataAnalyzer(dataset_path, api_key)
        analyzer.generate_comprehensive_report()
    
    except KeyError:
        print("Error: AIPROXY_TOKEN environment variable not set.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
