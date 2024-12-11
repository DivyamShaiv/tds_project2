# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scikit-learn", "chardet", "requests", "seaborn", "matplotlib", "python-dotenv"]
# ///

import os
import sys
import base64
import subprocess
import json
import logging
from typing import Optional, List, Dict, Any

packages = ["numpy", "pandas", "scikit-learn", "chardet", "requests", "seaborn", "matplotlib", "python-dotenv"]
for package in packages:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install '{package}'. Error: {e}")

import pandas as pd
import chardet
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dotenv import load_dotenv

class DataAnalyis:
    def __init__(self, dataset_path, api_key):
        self.dataset_path = dataset_path
        self.api_key = api_key
        self.df = None
        self.headers_json = None
        self.profile = None
        self.output_dir = os.path.splitext(self.dataset_path)[0]
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        self.ensure_output_dir()

    def ensure_output_dir(self):
        """
        Create the output directory for storing analysis results.
        """
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.logger.info(f"Output directory created: {self.output_dir}")
        except Exception as e:
            self.logger.error(f"Error creating output directory: {e}")
            sys.exit(1)

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

    def create_profile(self):
        """
        Create a comprehensive profile of the dataset.
        """
        try:
            if self.df is None:
                raise ValueError("Data not loaded. Call read_data() first.")
            
            # Compute basic statistics
            profile = {
                "shape": self.df.shape,
                "columns": self.df.columns.tolist(),
                "dtypes": self.df.dtypes.to_dict(),
                "missing_values": self.df.isnull().sum().to_dict(),
                "descriptive_stats": self.df.describe().to_dict()
            }
            
            self.profile = profile
            
            # Save profile to JSON
            profile_path = os.path.join(self.output_dir, "dataset_profile.json")
            with open(profile_path, "w") as f:
                json.dump(profile, f, indent=2, default=str)
            
            self.logger.info(f"Dataset profile saved to {profile_path}")
        except Exception as e:
            self.logger.error(f"Error creating dataset profile: {e}")

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

    def readme(self):
        """
        Generate a comprehensive README with insights and visualizations.
        """
        if not os.path.exists(self.output_dir):
            self.logger.error(f"Directory does not exist: {self.output_dir}")
            return

        # Prepare visualization paths
        image_files = [
            f for f in os.listdir(self.output_dir) 
            if f.endswith('.png')
        ]

        try:
            # Prepare comprehensive context for the LLM
            analysis_context = {
                "dataset_name": os.path.basename(self.dataset_path),
                "dataset_profile": {
                    "shape": self.df.shape,
                    "columns": self.df.columns.tolist(),
                    "data_types": self.df.dtypes.apply(str).to_dict(),
                    "missing_values": self.df.isnull().sum().to_dict(),
                    "numeric_summary": self.df.describe().to_dict()
                },
                "visualizations": image_files
            }

            # Generate README content via LLM
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
                        "content": """You are a professional data analyst creating a comprehensive README.md file. 
                        Your goal is to provide:
                        1. A clear overview of the dataset
                        2. Detailed insights from the data analysis
                        3. Key statistical observations
                        4. Potential business or research implications
                        5. Recommendations for further investigation

                        Use markdown formatting. Be analytical, concise, and provide actionable insights.
                        Include references to the visualizations where relevant.
                        Write in a professional, academic-research style tone."""
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

            # Add visualizations to README
            full_readme_content = "# Dataset Analysis Report\n\n"
            full_readme_content += readme_content + "\n\n"
            
            # Add visualization references
            full_readme_content += "## Visualizations\n\n"
            for image in image_files:
                full_readme_content += f"### {os.path.splitext(image)[0]}\n"
                full_readme_content += f"![{image}]({image})\n\n"

            # Write README
            readme_path = os.path.join(self.output_dir, "README.md")
            with open(readme_path, "w", encoding="utf-8") as readme_file:
                readme_file.write(full_readme_content)
            
            self.logger.info(f"Comprehensive README generated at {readme_path}")

        except Exception as e:
            self.logger.error(f"Failed to generate README: {e}")
            # Fallback README if LLM generation fails
            self._create_fallback_readme(image_files)

    def _create_fallback_readme(self, image_files):
        """
        Create a basic README if LLM generation fails.
        """
        try:
            readme_content = f"# Dataset Analysis Report for {os.path.basename(self.dataset_path)}\n\n"
            readme_content += "## Dataset Overview\n"
            readme_content += f"- Total Rows: {self.df.shape[0]}\n"
            readme_content += f"- Total Columns: {self.df.shape[1]}\n\n"
            
            readme_content += "## Column Types\n"
            for col, dtype in self.df.dtypes.items():
                readme_content += f"- {col}: {dtype}\n"
            
            readme_content += "\n## Visualizations\n"
            for image in image_files:
                readme_content += f"### {os.path.splitext(image)[0]}\n"
                readme_content += f"![{image}]({image})\n\n"

            readme_path = os.path.join(self.output_dir, "README.md")
            with open(readme_path, "w", encoding="utf-8") as readme_file:
                readme_file.write(readme_content)
            
            self.logger.info(f"Fallback README generated at {readme_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to create fallback README: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <dataset.csv>")
        sys.exit(1)

    dataset_file = sys.argv[1]

    try:
        load_dotenv()
        api_key = os.environ["AIPROXY_TOKEN"]
    except KeyError:
        raise ValueError("AIPROXY_TOKEN environment variable not set.")

    analysis = DataAnalyis(dataset_file, api_key)
    analysis.read_data()
    analysis.extract_headers()
    analysis.create_profile()
    analysis.construct_scatterplot()
    analysis.generate_correlation_heatmap()
    analysis.generate_cluster_plot()
    analysis.readme()

if __name__ == "__main__":
    main()
