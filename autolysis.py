# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scikit-learn", "chardet", "requests", "seaborn", "matplotlib", "python-dotenv","openai","networkx","pyarrow"]
# ///

import os
import sys
import logging
from typing import Dict, Any, Optional

try:
    import numpy as np
    import pandas as pd
    import chardet
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from dotenv import load_dotenv

    import matplotlib.pyplot as plt
    import seaborn as sns
    import networkx as nx

    from openai import OpenAI
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install required packages using:")
    print("pip install numpy pandas scikit-learn chardet python-dotenv "
          "matplotlib seaborn networkx openai pyarrow")
    sys.exit(1)

class EfficientDataAnalyzer:
    """
    Efficient, streamlined data analysis framework with optimized processing.
    """
    
    def __init__(self, 
                 dataset_path: str, 
                 api_key: str, 
                 log_level: int = logging.INFO):
        """
        Initialize the data analysis agent with efficient configuration.
        """
        self.dataset_path = dataset_path
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        
        # Optimized logging
        logging.basicConfig(
            level=log_level, 
            format='%(asctime)s | %(levelname)s: %(message)s',
            handlers=[logging.StreamHandler(), logging.FileHandler('analysis.log')]
        )
        self.logger = logging.getLogger(__name__)
        
        # Efficient output management
        self.output_dir = self._setup_output_directory()
        self.df = None
    
    def _setup_output_directory(self) -> str:
        """
        Create output directory with minimal overhead.
        """
        base_name = os.path.splitext(os.path.basename(self.dataset_path))[0]
        output_path = os.path.join('analysis_outputs', base_name)
        os.makedirs(output_path, exist_ok=True)
        return output_path
    
    def load_data(self) -> 'EfficientDataAnalyzer':
        """
        Efficiently load and detect encoding with minimal memory usage.
        """
        try:
            # Optimized encoding detection
            with open(self.dataset_path, 'rb') as file:
                raw_data = file.read(10000)  # Sample first 10KB for encoding
                encoding = chardet.detect(raw_data)['encoding']
            
            # Load with memory-efficient parameters
            self.df = pd.read_csv(
                self.dataset_path, 
                encoding=encoding, 
                low_memory=True,  # Reduce memory usage
                dtype_backend='pyarrow'  # Use PyArrow for efficient parsing
            )
            
            self._preprocess_data()
            return self
        
        except Exception as e:
            self.logger.error(f"Data loading failed: {e}")
            raise
    
    def _preprocess_data(self):
        """
        Efficient, comprehensive data preprocessing pipeline.
        """
        # Remove excessive missing data rows
        self.df.dropna(thresh=len(self.df.columns) * 0.5, inplace=True)
        
        # Identify column types efficiently
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols)
            ],
            remainder='drop'  # Efficiently handle non-numeric columns
        )
        
        # Fit transform in one step
        self.df = pd.DataFrame(
            preprocessor.fit_transform(self.df),
            columns=preprocessor.get_feature_names_out()
        )
    
    def generate_visualizations(self) -> 'EfficientDataAnalyzer':
        """
        Create optimized, informative visualizations.
        """
        plt.style.use('seaborn')
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.df)
        
        # Efficient clustering
        clustering = DBSCAN(eps=0.5, min_samples=3)
        clusters = clustering.fit_predict(pca_result)
        
        # Visualization with reduced computational complexity
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            pca_result[:, 0], 
            pca_result[:, 1], 
            c=clusters, 
            cmap='viridis', 
            alpha=0.7
        )
        plt.title('Data Clustering via PCA', fontsize=15)
        plt.colorbar(scatter, label='Cluster')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pca_clustering.png'), dpi=150)
        plt.close()
        
        # Correlation heatmap with efficient computation
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlation Matrix', fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_heatmap.png'), dpi=150)
        plt.close()
        
        return self
    
    def generate_report(self) -> None:
        """
        Generate a concise, informative analysis report.
        """
        try:
            (self.load_data()
                .generate_visualizations())
            
            # Create markdown report
            report_content = f"""# Data Analysis Report

## Dataset Overview
- Total Rows: {self.df.shape[0]}
- Total Columns: {self.df.shape[1]}

## Statistical Summary
{self.df.describe().to_markdown()}

## Visualizations
![PCA Clustering](pca_clustering.png)
![Correlation Heatmap](correlation_heatmap.png)
"""
            
            report_path = os.path.join(self.output_dir, 'analysis_report.md')
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            self.logger.info(f"Report generated: {report_path}")
        
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")

def main():
    """
    Efficient main execution with robust error handling.
    """
    load_dotenv()
    
    if len(sys.argv) < 2:
        print("Usage: python script.py <dataset.csv>")
        sys.exit(1)
    
    try:
        api_key = os.getenv("AIPROXY_TOKEN")
        if not api_key:
            raise ValueError("API token not found")
        
        dataset_path = sys.argv[1]
        analyzer = EfficientDataAnalyzer(dataset_path, api_key)
        analyzer.generate_report()
    
    except Exception as e:
        print(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
