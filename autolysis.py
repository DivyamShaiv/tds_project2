# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scikit-learn", "chardet", "requests", "seaborn", "matplotlib", "python-dotenv"]
# ///

import os
import sys
import logging
import subprocess
from typing import Dict, Any, List, Optional, Tuple

# Automated dependency management
REQUIRED_PACKAGES = [
    "numpy", "pandas", "scikit-learn", "chardet", 
    "requests", "seaborn", "matplotlib", "python-dotenv",
    "openai", "PIL"
]

def install_dependencies(packages):
    """
    Safely install required Python packages.
    
    Args:
        packages (List[str]): List of package names to install
    """
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install '{package}'. Error: {e}")

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
from openai import OpenAI
from PIL import Image
import base64
import io

class AdvancedDataAnalysisAgent:
    """
    A multi-modal, agentic data analysis framework with comprehensive capabilities.
    """
    
    def __init__(self, 
                 dataset_path: str, 
                 api_key: str, 
                 log_level: int = logging.INFO):
        """
        Initialize the advanced data analysis agent.
        
        Args:
            dataset_path (str): Path to the input dataset
            api_key (str): API key for LLM and vision services
            log_level (int): Logging verbosity level
        """
        self.dataset_path = dataset_path
        self.api_key = api_key
        self.df = None
        self.analysis_results = {}
        self.vision_client = OpenAI(api_key=api_key)
        
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
    
    def load_and_validate_data(self) -> pd.DataFrame:
        """
        Robustly load data with advanced validation.
        
        Returns:
            pd.DataFrame: Processed and validated DataFrame
        """
        try:
            # Encoding detection
            with open(self.dataset_path, 'rb') as file:
                raw_data = file.read()
                encoding = chardet.detect(raw_data)['encoding']
            
            # Load with detected encoding
            self.df = pd.read_csv(self.dataset_path, encoding=encoding)
            
            # Advanced data validation
            self._validate_data()
            
            self.logger.info(f"Loaded dataset: {self.df.shape}")
            return self.df
        
        except Exception as e:
            self.logger.error(f"Data loading error: {e}")
            raise
    
    def _validate_data(self):
        """
        Perform comprehensive data validation.
        """
        # Check for duplicates
        duplicate_count = self.df.duplicated().sum()
        if duplicate_count > 0:
            self.logger.warning(f"Found {duplicate_count} duplicate rows")
            self.df.drop_duplicates(inplace=True)
        
        # Check data types and convert if necessary
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='raise')
                except ValueError:
                    pass  # Keep as categorical if conversion fails
        
        # Remove rows with too many missing values
        self.df.dropna(thresh=len(self.df.columns)*0.5, inplace=True)
    
    def agentic_preprocessing(self):
        """
        Multi-stage, adaptive preprocessing with dynamic strategy selection.
        """
        # Dynamic imputation strategy
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        # Adaptive imputation
        for col in numeric_cols:
            impute_strategy = 'median' if self.df[col].skew() > 1 else 'mean'
            imputer = SimpleImputer(strategy=impute_strategy)
            self.df[col] = imputer.fit_transform(self.df[[col]])
        
        # Scaling with dynamic method
        scaler = StandardScaler() if len(numeric_cols) > 5 else MinMaxScaler()
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        
        return self
    
    def generate_advanced_visualizations(self):
        """
        Create multi-modal, context-rich visualizations.
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Correlation Network
        plt.figure(figsize=(15, 12))
        corr_matrix = self.df.corr()
        
        # Use networkx for correlation visualization
        import networkx as nx
        
        G = nx.Graph()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                correlation = corr_matrix.iloc[i, j]
                if abs(correlation) > 0.5:
                    G.add_edge(
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        weight=abs(correlation)
                    )
        
        pos = nx.spring_layout(G, k=0.5)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos)
        plt.title('Feature Correlation Network', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_network.png'))
        plt.close()
        
        # PCA Visualization with Clustering
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.df.select_dtypes(include=['number']))
        
        # Use DBSCAN for dynamic clustering
        clustering = DBSCAN(eps=0.5, min_samples=3)
        clusters = clustering.fit_predict(pca_result)
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            pca_result[:, 0], 
            pca_result[:, 1], 
            c=clusters, 
            cmap='viridis', 
            alpha=0.7
        )
        plt.title('PCA with Dynamic Clustering', fontsize=16)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter, label='Cluster')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pca_clustering.png'))
        plt.close()
        
        return self
    
    def vision_analysis(self, image_path: str) -> Dict[str, Any]:
        """
        Perform vision-based analysis of an image.
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            Dict: Vision analysis results
        """
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = self.vision_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "Analyze the image from a data science perspective."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": "Provide a detailed analysis of this visualization, highlighting key insights and potential implications."
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            return {
                "vision_analysis": response.choices[0].message.content
            }
        
        except Exception as e:
            self.logger.error(f"Vision analysis failed: {e}")
            return {"error": str(e)}
    
    def generate_dynamic_narrative(self) -> str:
        """
        Generate a dynamic, context-aware narrative.
        
        Returns:
            str: Generated narrative in Markdown
        """
        try:
            # Prepare narrative content
            narrative = """# Dataset Narrative

## Overview
Dataset contains {rows} rows and {columns} columns.

### Numeric Columns
{numeric_columns}

### Summary Statistics
{summary_statistics}
""".format(
                rows=self.df.shape[0],
                columns=self.df.shape[1],
                numeric_columns=", ".join(self.df.select_dtypes(include=['number']).columns),
                summary_statistics=str(self.df.describe())
            )
            
            return narrative
        
        except Exception as e:
            self.logger.error(f"Dynamic narrative generation failed: {e}")
            return "## Narrative Generation Error\n\nUnable to generate comprehensive narrative."
    
    def generate_comprehensive_report(self):
        """
        Orchestrate the entire advanced analysis workflow.
        """
        try:
            (self.load_and_validate_data()
                .agentic_preprocessing()
                .generate_advanced_visualizations())
            
            # Generate dynamic narrative
            narrative = self.generate_dynamic_narrative()
            
            # Combine narrative with visualizations
            full_report = f"""# Advanced Data Analysis Report

{narrative}

## Visualizations

### Correlation Network
![Correlation Network](correlation_network.png)

### PCA with Dynamic Clustering
![PCA Clustering](pca_clustering.png)
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
        
        analyzer = AdvancedDataAnalysisAgent(dataset_path, api_key)
        analyzer.generate_comprehensive_report()
    
    except KeyError:
        print("Error: AIPROXY_TOKEN environment variable not set.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
