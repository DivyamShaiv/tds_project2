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
    
    def load_data(self):
        """
        Robustly load data with encoding detection and preprocessing.
        
        Returns:
            self: Returns the instance for method chaining
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
            return self
        
        except Exception as e:
            self.logger.error(f"Data loading error: {e}")
            raise


    def read_data(self):
        """
        Read the CSV file with automatic encoding detection.
        """
        try:
            with open(self.dataset_path, 'rb') as file:
                result = chardet.detect(file.read())
                encoding = result['encoding']

            self.df = pd.read_csv(self.dataset_path, encoding=encoding)
            if self.df is None or self.df.empty:
                self.logger.error("Dataset is empty or could not be loaded.")
                sys.exit("Dataset is empty or could not be loaded.")
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            sys.exit(1)

    def extract_headers(self):
        """
        Extract headers from the dataset and save as JSON.
        """
        try:
            if self.df is None:
                raise ValueError("Data not loaded. Call read_data() first.")
            
            # Save headers to JSON
            self.headers_json = self.df.columns.tolist()
            headers_path = os.path.join(self.output_dir, "headers.json")
            
            with open(headers_path, "w") as f:
                json.dump(self.headers_json, f, indent=2)
            
            self.logger.info(f"Headers saved to {headers_path}")
        except Exception as e:
            self.logger.error(f"Error extracting headers: {e}")
    
    def advanced_preprocessing(self):
        """
        Advanced data preprocessing with multiple techniques.
        
        Returns:
            self: Returns the instance for method chaining
        """
        # Ensure data is loaded
        if self.df is None:
            raise ValueError("Data must be loaded first. Call load_data() before preprocessing.")
        
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
        Calculate feature importance using alternative methods for continuous data.
        
        Returns:
            self: Returns the instance for method chaining
        """
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        try:
            # Option 1: Use correlation-based feature importance
            correlations = self.df[numeric_cols].corr().abs().mean()
            importance_dict = correlations.to_dict()
            
            # Option 2: Use variance as a simple importance metric
            variances = self.df[numeric_cols].var()
            
            # Combine methods for a more robust importance score
            combined_importance = {}
            for col in numeric_cols:
                combined_importance[col] = (
                    importance_dict.get(col, 0) * 0.7 +  # Correlation weight
                    variances.get(col, 0) * 0.3  # Variance weight
                )
            
            # Normalize to 0-1 range
            max_importance = max(combined_importance.values())
            normalized_importance = {
                k: v / max_importance for k, v in combined_importance.items()
            }
            
            self.analysis_results['feature_importances'] = normalized_importance
            
            # Log feature importances
            self.logger.info("Feature Importances:")
            for feature, importance in sorted(normalized_importance.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"{feature}: {importance:.4f}")
            
            return self
        
        except Exception as e:
            self.logger.error(f"Feature importance calculation failed: {e}")
            return self
    
    def construct_scatterplot(self):
        """
        Generate a scatter plot of two numeric columns.
        """
        try:
            if self.df is None:
                raise ValueError("Data not loaded. Call read_data() first.")
            
            # Select two numeric columns
            numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
            
            if len(numeric_cols) < 2:
                self.logger.warning("Not enough numeric columns for scatter plot.")
                return
            
            plt.figure(figsize=(10, 6))
            plt.scatter(self.df[numeric_cols[0]], self.df[numeric_cols[1]], alpha=0.7)
            plt.title(f"Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}")
            plt.xlabel(numeric_cols[0])
            plt.ylabel(numeric_cols[1])
            
            scatter_path = os.path.join(self.output_dir, "scatter_plot.png")
            plt.savefig(scatter_path)
            plt.close()
            
            self.logger.info(f"Scatter plot saved to {scatter_path}")
        except Exception as e:
            self.logger.error(f"Error generating scatter plot: {e}")

    def generate_correlation_heatmap(self):
        """
        Generate a correlation heatmap for numeric columns.
        """
        try:
            if self.df is None:
                raise ValueError("Data not loaded. Call read_data() first.")
            
            # Select numeric columns
            numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
            
            if numeric_df.empty:
                self.logger.warning("No numeric columns found for correlation heatmap.")
                return
            
            # Compute correlation matrix
            corr_matrix = numeric_df.corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title("Correlation Heatmap")
            
            heatmap_path = os.path.join(self.output_dir, "correlation_heatmap.png")
            plt.savefig(heatmap_path)
            plt.close()
            
            self.logger.info(f"Correlation heatmap saved to {heatmap_path}")
        except Exception as e:
            self.logger.error(f"Error generating correlation heatmap: {e}")

    def generate_cluster_plot(self):
        """
        Perform K-means clustering and visualize results.
        """
        try:
            if self.df is None:
                raise ValueError("Data not loaded. Call read_data() first.")
            
            # Select numeric columns
            numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
            
            if numeric_df.empty or numeric_df.shape[1] < 2:
                self.logger.warning("Not enough numeric columns for clustering.")
                return
            
            # Prepare data for clustering
            scaler = StandardScaler()
            imputer = SimpleImputer(strategy='mean')
            
            # Impute missing values
            numeric_data = imputer.fit_transform(numeric_df)
            scaled_data = scaler.fit_transform(numeric_data)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            
            # Plot clusters
            plt.figure(figsize=(10, 6))
            
            # Use first two columns for visualization
            scatter = plt.scatter(
                scaled_data[:, 0], 
                scaled_data[:, 1], 
                c=clusters, 
                cmap='viridis'
            )
            
            plt.title("K-means Clustering")
            plt.xlabel(numeric_df.columns[0])
            plt.ylabel(numeric_df.columns[1])
            plt.colorbar(scatter, label='Cluster')
            
            cluster_path = os.path.join(self.output_dir, "cluster_plot.png")
            plt.savefig(cluster_path)
            plt.close()
            
            self.logger.info(f"Clustering plot saved to {cluster_path}")
        except Exception as e:
            self.logger.error(f"Error generating cluster plot: {e}")

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
    
    def generate_llm_narrative(self, prompt: Dict[str, Any]) -> Optional[str]:
        """
        Generate narrative using an advanced LLM based on the prompt context.
        
        Args:
            prompt (Dict): Structured context for narrative generation
            
        Returns:
            Optional[str]: Generated narrative, or None if unsuccessful
        """
        try:
            # LLM API Integration
            llm_url = "https://api.example-llm.com/v1/generate"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "model": "text-davinci-003",
                "prompt": json.dumps(prompt),
                "max_tokens": 500
            }
            
            response = requests.post(llm_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                narrative = response.json().get('choices', [{}])[0].get('text', '').strip()
                self.analysis_results['narrative'] = narrative
                return narrative
            else:
                self.logger.error(f"LLM API error: {response.status_code} - {response.text}")
                return None
        
        except Exception as e:
            self.logger.error(f"Failed to generate narrative: {e}")
            return None
    
    def save_analysis_results(self):
        """
        Save all analysis results, visualizations, and narratives to output directory.
        """
        try:
            # Save results to a JSON file
            results_path = os.path.join(self.output_dir, 'analysis_results.json')
            with open(results_path, 'w') as file:
                json.dump(self.analysis_results, file, indent=4)
            
            self.logger.info(f"Analysis results saved to {results_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to save analysis results: {e}")
            raise

# Example usage
if __name__ == "__main__":
    #load_dotenv()  # Load environment variables from a .env file

    if len(sys.argv) < 2:
        print("Usage: python script.py <dataset.csv>")
        sys.exit(1)

    dataset_file = sys.argv[1]

    try:
        load_dotenv()
        api_key = os.environ["AIPROXY_TOKEN"]
    except KeyError:
        raise ValueError("AIPROXY_TOKEN environment variable not set.")


    #api_key = os.getenv("API_KEY")
    #dataset_path = "path/to/your/dataset.csv"
    
    analyzer = AdvancedDataAnalyzer(dataset_path=dataset_file, api_key=api_key)
    analyzer.load_data().advanced_preprocessing().feature_importance().read_data()
    analyzer.construct_scatterplot()
    analyzer.generate_correlation_heatmap()
    analyzer.generate_cluster_plot()
    
    prompt = analyzer.generate_narrative_prompt()
    narrative = analyzer.generate_llm_narrative(prompt)
    
    if narrative:
        analyzer.logger.info(f"Narrative: {narrative}")
    
    analyzer.save_analysis_results()

