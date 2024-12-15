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

   def generate_data_narrative(self):
        """
        Generate a comprehensive narrative about the dataset using LLM.
        """
        if self.df is None:
            self.logger.error("Data not loaded. Cannot generate narrative.")
            return None

        try:
            # Prepare a rich context for narrative generation
            narrative_context = {
                "dataset_name": os.path.basename(self.dataset_path),
                "dataset_overview": {
                    "total_rows": self.df.shape[0],
                    "total_columns": self.df.shape[1],
                    "column_types": self.df.dtypes.apply(str).to_dict(),
                    "missing_values": self.df.isnull().sum().to_dict(),
                },
                "statistical_summary": self.df.describe().to_dict(),
                "correlation_insights": self.df.corr().to_dict() if len(self.df.select_dtypes(include=['float64', 'int64']).columns) > 0 else None
            }

            # Prepare LLM request
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
                        "content": """You are a professional data storyteller. Create a compelling narrative that:
                        1. Introduces the dataset as if it were a character with a story
                        2. Explains the data's origin, context, and potential significance
                        3. Highlights key statistical insights like a detective uncovering clues
                        4. Provides a forward-looking perspective on potential insights
                        5. Use metaphors, engaging language, and a narrative arc

                        Write in markdown, maintaining a balance between technical accuracy and storytelling."""
                    },
                    {
                        "role": "user",
                        "content": json.dumps(narrative_context, indent=2)
                    }
                ]
            }

            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            
            narrative = response.json()['choices'][0]['message']['content']
            return narrative

        except Exception as e:
            self.logger.error(f"Failed to generate data narrative: {e}")
            return None

    def readme(self):
        """
        Generate a comprehensive README with data story and visualizations.
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
            # Generate the data narrative
            data_story = self.generate_data_narrative()

            # Fallback to default content if narrative generation fails
            if data_story is None:
                self._create_fallback_readme(image_files)
                return

            # Compile the full README
            full_readme_content = "# Data Journey: A Story of Discovery\n\n"
            full_readme_content += data_story + "\n\n"
            
            # Add visualization references
            full_readme_content += "## Visual Chapters of Our Data Story\n\n"
            for image in image_files:
                full_readme_content += f"### {os.path.splitext(image)[0].replace('_', ' ').title()}\n"
                full_readme_content += f"![{image}]({image})\n\n"

            # Write README
            readme_path = os.path.join(self.output_dir, "README.md")
            with open(readme_path, "w", encoding="utf-8") as readme_file:
                readme_file.write(full_readme_content)
            
            self.logger.info(f"Data story README generated at {readme_path}")

        except Exception as e:
            self.logger.error(f"Failed to generate data story README: {e}")
            self._create_fallback_readme(image_files)

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
