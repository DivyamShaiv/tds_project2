# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scikit-learn", "chardet", "requests", "seaborn", "matplotlib", "python-dotenv"]
# ///

import os
import sys
import logging
import json
from typing import Dict, Any, List

# Use more explicit dependency management
try:
    import pandas as pd
    import numpy as np
    import chardet
    import requests
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score
    from dotenv import load_dotenv
except ImportError:
    print("Installing required packages...")
    import subprocess
    packages = [
        "numpy", "pandas", "scikit-learn", "chardet", 
        "requests", "seaborn", "matplotlib", "python-dotenv"
    ]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Retry imports
    import pandas as pd
    import numpy as np
    import chardet
    import requests
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score
    from dotenv import load_dotenv

class AdvancedDataAnalysis:
    def __init__(self, dataset_path: str, api_key: str):
        """
        Initialize the data analysis pipeline with comprehensive setup.
        
        Args:
            dataset_path (str): Path to the input CSV file
            api_key (str): API key for LLM generation
        """
        self.dataset_path = dataset_path
        self.api_key = api_key
        
        # Enhanced logging configuration
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler('data_analysis.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Dynamic output directory
        self.output_dir = self._create_output_directory()
        
        # Core data structures
        self.df = None
        self.processed_data = None
        self.analysis_results = {}

    def _create_output_directory(self) -> str:
        """
        Create a timestamped output directory for analysis results.
        
        Returns:
            str: Path to the created output directory
        """
        base_dir = os.path.splitext(self.dataset_path)[0]
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{base_dir}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Created output directory: {output_dir}")
        return output_dir

    def load_data(self) -> None:
        """
        Robustly load data with encoding detection and basic preprocessing.
        """
        try:
            # Detect encoding
            with open(self.dataset_path, 'rb') as file:
                result = chardet.detect(file.read())
            
            # Load with detected encoding
            self.df = pd.read_csv(self.dataset_path, encoding=result['encoding'])
            
            # Basic data validation
            if self.df.empty:
                raise ValueError("Empty dataset")
            
            self.logger.info(f"Loaded dataset: {self.dataset_path}")
            self._initial_data_exploration()
        
        except Exception as e:
            self.logger.error(f"Data loading error: {e}")
            raise

    def _initial_data_exploration(self) -> None:
        """
        Perform initial data exploration and logging.
        """
        # Data overview logging
        self.logger.info(f"Dataset Shape: {self.df.shape}")
        self.logger.info("Column Types:")
        for col, dtype in self.df.dtypes.items():
            self.logger.info(f"  {col}: {dtype}")
        
        # Save initial profile
        self._save_data_profile()

    def _save_data_profile(self) -> None:
        """
        Save comprehensive dataset profile.
        """
        profile = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "non_null_counts": self.df.count().to_dict(),
            "descriptive_stats": self.df.describe().to_dict()
        }
        
        profile_path = os.path.join(self.output_dir, "dataset_profile.json")
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2, default=str)

    def preprocess_data(self) -> None:
        """
        Advanced data preprocessing with multiple techniques.
        """
        # Select numeric columns
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        if len(numeric_cols) < 2:
            self.logger.warning("Insufficient numeric columns for advanced analysis")
            return
        
        # Imputation and scaling
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        # Prepare data
        numeric_data = self.df[numeric_cols]
        imputed_data = pd.DataFrame(
            imputer.fit_transform(numeric_data), 
            columns=numeric_cols
        )
        
        self.processed_data = pd.DataFrame(
            scaler.fit_transform(imputed_data), 
            columns=numeric_cols
        )
        
        self.logger.info("Data preprocessed successfully")

    def generate_visualizations(self) -> None:
        """
        Generate multiple advanced visualizations.
        """
        # Scatter plot matrix
        plt.figure(figsize=(15, 10))
        sns.pairplot(self.df.select_dtypes(include=['float64', 'int64']))
        plt.suptitle("Pairplot of Numeric Variables", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "pairplot.png"))
        plt.close()
        
        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.df.select_dtypes(include=['float64', 'int64']).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "correlation_heatmap.png"))
        plt.close()

    def advanced_clustering(self) -> None:
        """
        Perform advanced clustering with multiple algorithms.
        """
        if self.processed_data is None:
            self.preprocess_data()
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(self.processed_data)
        
        # Multiple clustering algorithms
        clustering_methods = {
            'KMeans': KMeans(n_clusters=3, random_state=42),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
        }
        
        results = {}
        for name, algorithm in clustering_methods.items():
            if name == 'KMeans':
                labels = algorithm.fit_predict(self.processed_data)
                score = silhouette_score(self.processed_data, labels)
            else:
                labels = algorithm.fit_predict(self.processed_data)
                # Handle potential noise points
                score = silhouette_score(
                    self.processed_data[labels != -1], 
                    labels[labels != -1]
                ) if len(labels[labels != -1]) > 0 else 0
            
            results[name] = {
                'labels': labels,
                'score': score
            }
        
        # Visualization
        plt.figure(figsize=(15, 6))
        for i, (name, result) in enumerate(results.items(), 1):
            plt.subplot(1, 2, i)
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=result['labels'], cmap='viridis')
            plt.title(f"{name} Clustering (Score: {result['score']:.2f})")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "clustering_comparison.png"))
        plt.close()

    def generate_narrative_readme(self) -> None:
        """
        Generate a comprehensive narrative README using LLM.
        """
        analysis_context = {
            "dataset_name": os.path.basename(self.dataset_path),
            "dataset_profile": {
                "shape": self.df.shape,
                "columns": list(self.df.columns),
                "data_types": self.df.dtypes.astype(str).to_dict(),
                "missing_values": self.df.isnull().sum().to_dict(),
                "numeric_summary": self.df.describe().to_dict()
            },
            "visualizations": [
                "pairplot.png", 
                "correlation_heatmap.png", 
                "clustering_comparison.png"
            ]
        }
        
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
                        "content": """Create a comprehensive data analysis narrative:
                        1. Provide a clear, engaging dataset overview
                        2. Highlight key statistical insights
                        3. Interpret visualizations meaningfully
                        4. Suggest potential research or business implications
                        5. Use professional, academic research writing style
                        6. Use markdown formatting
                        7. Be concise yet comprehensive"""
                    },
                    {
                        "role": "user",
                        "content": json.dumps(analysis_context, indent=2)
                    }
                ]
            }
            
            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            
            readme_content = response.json()['choices'][0]['message']['content']
            
            # Comprehensive README with visualizations
            full_readme = f"# Data Analysis Report: {os.path.basename(self.dataset_path)}\n\n"
            full_readme += readme_content + "\n\n"
            
            # Add visualization references
            full_readme += "## Visualizations\n\n"
            for viz in analysis_context["visualizations"]:
                full_readme += f"### {os.path.splitext(viz)[0]}\n"
                full_readme += f"![{viz}]({viz})\n\n"
            
            # Write README
            readme_path = os.path.join(self.output_dir, "README.md")
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(full_readme)
            
            self.logger.info(f"Narrative README generated: {readme_path}")
        
        except Exception as e:
            self.logger.error(f"README generation failed: {e}")
            self._fallback_readme()

    def _fallback_readme(self):
        """
        Generate a basic README if LLM generation fails.
        """
        readme_content = f"# Data Analysis Report: {os.path.basename(self.dataset_path)}\n\n"
        readme_content += "## Dataset Overview\n"
        readme_content += f"- Total Rows: {self.df.shape[0]}\n"
        readme_content += f"- Total Columns: {self.df.shape[1]}\n\n"
        
        readme_path = os.path.join(self.output_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <dataset.csv>")
        sys.exit(1)

    dataset_file = sys.argv[1]

    try:
        load_dotenv()
        api_key = os.environ.get("AIPROXY_TOKEN")
        if not api_key:
            raise ValueError("AIPROXY_TOKEN environment variable not set")

        analysis = AdvancedDataAnalysis(dataset_file, api_key)
        analysis.load_data()
        analysis.preprocess_data()
        analysis.generate_visualizations()
        analysis.advanced_clustering()
        analysis.generate_narrative_readme()

    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
