# /// script
# requires-python = ">=3.11"
# dependencies = [ "pandas","numpy","matplotlib","seaborn","requests","Pillow","tk","dotenv"]
# ///

import os
import sys
import json
import warnings
import base64
import requests

import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv

# Suppress warnings globally
warnings.filterwarnings("ignore")
matplotlib.use('Agg')

class DataAnalysisConfig:
    """Configuration class for data analysis settings."""
    load_dotenv()
    api_key = os.environ["AIPROXY_TOKEN"]
    if api_key is None:
        print("Error: Authentication token is required.")
        sys.exit(1)

    API_ENDPOINT = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    REQUEST_HEADERS = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    LANGUAGE_MODEL = "gpt-4o-mini"

class DataAnalyzer:
    """Main class to handle data analysis workflow."""

    @staticmethod
    def clean_dataset(dataframe):
        """
        Perform basic data cleaning.
        
        Args:
            dataframe (pd.DataFrame): Input dataframe
        
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        cleaned_df = dataframe.drop_duplicates()
        cleaned_df = cleaned_df.dropna(how='all')
        return cleaned_df

    @staticmethod
    def analyze_numeric_columns(numeric_columns):
        """
        Generate comprehensive summary for numeric columns.
        
        Args:
            numeric_columns (pd.DataFrame): Numeric columns of the dataset
        
        Returns:
            pd.DataFrame: Detailed numeric column analysis
        """
        if numeric_columns.empty:
            return None

        numeric_analysis = numeric_columns.describe().transpose()
        numeric_analysis['missing_percentage'] = numeric_columns.isnull().mean() * 100
        numeric_analysis['unique_values'] = numeric_columns.nunique()
        numeric_analysis['skewness'] = numeric_columns.skew()
        numeric_analysis['kurtosis'] = numeric_columns.kurtosis()

        # Detect potential outliers
        q1 = numeric_columns.quantile(0.25)
        q3 = numeric_columns.quantile(0.75)
        iqr = q3 - q1
        outlier_flags = ((numeric_columns < (q1 - 1.5 * iqr)) | (numeric_columns > (q3 + 1.5 * iqr))).sum()
        numeric_analysis['outliers'] = outlier_flags

        return numeric_analysis

    @staticmethod
    def analyze_categorical_columns(categorical_columns):
        """
        Generate comprehensive summary for categorical columns.
        
        Args:
            categorical_columns (pd.DataFrame): Categorical columns of the dataset
        
        Returns:
            pd.DataFrame: Detailed categorical column analysis
        """
        if categorical_columns.empty:
            return None

        categorical_analysis = categorical_columns.describe().transpose()
        categorical_analysis['missing_percentage'] = categorical_columns.isnull().mean() * 100
        categorical_analysis['unique_values'] = categorical_columns.nunique()

        # Detect categorical imbalance
        imbalance_flags = categorical_columns.apply(lambda col: col.value_counts(normalize=True).iloc[0] > 0.8)
        categorical_analysis['imbalance'] = imbalance_flags

        return categorical_analysis

    @staticmethod
    def generate_dataset_summary(dataframe):
        """
        Generate comprehensive dataset summary.
        
        Args:
            dataframe (pd.DataFrame): Input dataframe
        
        Returns:
            dict: Detailed dataset summary
        """
        cleaned_df = DataAnalyzer.clean_dataset(dataframe)
        summary = {}

        # Numeric data summary
        numeric_columns = cleaned_df.select_dtypes(exclude='object')
        if not numeric_columns.empty:
            summary['numeric'] = DataAnalyzer.analyze_numeric_columns(numeric_columns)

        # Categorical data summary
        categorical_columns = cleaned_df.select_dtypes(include='object')
        if not categorical_columns.empty:
            summary['categorical'] = DataAnalyzer.analyze_categorical_columns(categorical_columns)

        # Correlation matrix for numeric columns
        if numeric_columns.shape[1] > 1:
            correlation_matrix = numeric_columns.corr()
            high_corr_cols = correlation_matrix.columns[correlation_matrix.abs().gt(0.9).any()]
            correlation_matrix = correlation_matrix[high_corr_cols].loc[high_corr_cols]
            summary['correlation'] = correlation_matrix

        # High missing value features
        high_missing = cleaned_df.isnull().mean() * 100
        high_missing_features = high_missing[high_missing > 50].sort_values(ascending=False)
        if not high_missing_features.empty:
            summary['high_missing'] = high_missing_features

        return summary

    @staticmethod
    def request_column_analysis(dataset_summary):
        """
        Request LLM to suggest columns for analysis.
        
        Args:
            dataset_summary (dict): Comprehensive dataset summary
        
        Returns:
            dict: Suggested columns for visualization
        """
        analysis_functions = [{
            "name": "suggest_analysis_columns",
            "description": "Recommend columns for histogram, bar chart, and pairplot",
            "parameters": {
                "type": "object",
                "properties": {
                    "histogram": {"type": "string", "description": "Column for histogram"},
                    "barchart": {"type": "string", "description": "Column for bar chart"},
                    "pairplot1": {"type": "string", "description": "First column for pairplot"},
                    "pairplot2": {"type": "string", "description": "Second column for pairplot"}
                },
                "required": ["histogram", "barchart", "pairplot1", "pairplot2"]
            }
        }]

        prompt = f'''Suggest columns for analysis based on the dataset summary.
                     \nSummary: {dataset_summary}'''

        request_payload = {
            "model": DataAnalysisConfig.LANGUAGE_MODEL,
            "functions": analysis_functions,
            "function_call": {"name": "suggest_analysis_columns"},
            "messages": [{"role": "user", "content": prompt}]
        }

        response = requests.post(
            DataAnalysisConfig.API_ENDPOINT, 
            headers=DataAnalysisConfig.REQUEST_HEADERS, 
            json=request_payload
        )
        result = response.json()
        return json.loads(result["choices"][0]["message"]["function_call"]["arguments"])

    @staticmethod
    def create_visualizations(dataframe, selected_columns, output_dir):
        """
        Generate and save various data visualizations.
        
        Args:
            dataframe (pd.DataFrame): Input dataframe
            selected_columns (dict): Columns for visualization
            output_dir (str): Directory to save visualizations
        """
        histogram_column = selected_columns["histogram"]
        barchart_column = selected_columns["barchart"]
        pairplot_columns = (selected_columns["pairplot1"], selected_columns["pairplot2"])

        visualization_specs = [
            {
                'type': 'histogram',
                'column': histogram_column,
                'title': f'Distribution of {histogram_column}',
                'x_label': histogram_column,
                'y_label': 'Frequency'
            },
            {
                'type': 'bar',
                'column': barchart_column,
                'title': f'Top Categories of {barchart_column}',
                'x_label': barchart_column,
                'y_label': 'Counts'
            },
            {
                'type': 'scatter',
                'columns': pairplot_columns,
                'title': f'Scatterplot of {pairplot_columns[0]} vs {pairplot_columns[1]}',
                'x_label': pairplot_columns[0],
                'y_label': pairplot_columns[1]
            }
        ]

        for viz in visualization_specs:
            plt.figure(figsize=(8, 6))
            
            if viz['type'] == 'histogram':
                dataframe[viz['column']].dropna().hist(bins=30, color='skyblue', edgecolor='black')
            
            elif viz['type'] == 'bar':
                value_counts = dataframe[viz['column']].value_counts()
                top_values = value_counts[:10] if len(value_counts) > 10 else value_counts
                top_values.plot(kind='bar', color='skyblue', edgecolor='black')
            
            elif viz['type'] == 'scatter':
                plt.scatter(
                    dataframe[viz['columns'][0]], 
                    dataframe[viz['columns'][1]], 
                    color='skyblue', 
                    edgecolor='black', 
                    alpha=0.7
                )

            plt.title(viz['title'])
            plt.xlabel(viz['x_label'])
            plt.ylabel(viz['y_label'])
            plt.legend(title="Legend", loc="upper right")
            plt.tight_layout()
            
            output_filename = f"{viz['x_label'].replace(' ', '_')}_{viz['type']}.png"
            plt.savefig(os.path.join(output_dir, output_filename))
            plt.close()

    @staticmethod
    def generate_readme(dataset_summary, visualization_paths):
        """
        Generate README using LLM.
        
        Args:
            dataset_summary (dict): Comprehensive dataset summary
            visualization_paths (list): Paths to generated visualizations
        
        Returns:
            str: Generated README content
        """
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        encoded_images = [f"data:image/jpeg;base64,{encode_image(path)}" for path in visualization_paths]
        image_filenames = [os.path.basename(f) for f in visualization_paths]

        readme_function = [{
            "name": "generate_readme",
            "description": "Write a comprehensive dataset analysis report",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Analysis report"}
                },
                "required": ["text"]
            }
        }]

        prompt = f'''Generate a comprehensive README.md for the dataset. 
                     Include dataset purpose, key findings, insights, and recommendations.
                     Reference visualizations: {image_filenames}'''

        request_payload = {
            "model": DataAnalysisConfig.LANGUAGE_MODEL,
            "functions": readme_function,
            "function_call": {"name": "generate_readme"},
            "messages": [{"role": "user", "content": prompt}] +
                        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}}]} for image in encoded_images]
        }

        response = requests.post(
            DataAnalysisConfig.API_ENDPOINT, 
            headers=DataAnalysisConfig.REQUEST_HEADERS, 
            json=request_payload
        )
        result = response.json()
        return json.loads(result["choices"][0]["message"]["function_call"]["arguments"])['text']

def main():
    """Main execution function for the data analysis script."""
    if len(sys.argv) < 2:
        print("Usage: python script.py <dataset.csv>")
        sys.exit(1)

    dataset_file = sys.argv[1]

    try:
        load_dotenv()
        api_key = os.environ["AIPROXY_TOKEN"]
    except KeyError:
        raise ValueError("AIPROXY_TOKEN environment variable not set.")
    
    # Create output directory
    output_directory = os.path.splitext(os.path.basename(dataset_file))[0]
    os.makedirs(output_directory, exist_ok=True)

    try:
        # Load dataset
        dataframe = pd.read_csv(dataset_file, encoding='ISO-8859-1')

        # Generate dataset summary
        dataset_summary = DataAnalyzer.generate_dataset_summary(dataframe)

        # Request column analysis
        selected_columns = DataAnalyzer.request_column_analysis(dataset_summary)

        # Create visualizations
        DataAnalyzer.create_visualizations(dataframe, selected_columns, output_directory)

        # Resize generated images
        for image_filename in os.listdir(output_directory):
            if image_filename.endswith('.png'):
                image_path = os.path.join(output_directory, image_filename)
                img = Image.open(image_path)
                img_resized = img.resize((512, 512))
                img_resized.save(image_path)

        # Generate README
        visualization_paths = [os.path.join(output_directory, f) for f in os.listdir(output_directory) if f.endswith('.png')]
        readme_content = DataAnalyzer.generate_readme(dataset_summary, visualization_paths)

        # Save README
        with open(os.path.join(output_directory, "README.md"), "w") as readme_file:
            readme_file.write(f"# Automated Analysis of {dataset_file}\n\n")
            readme_file.write(readme_content)

        print("README.md saved. Analysis completed successfully.")

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
