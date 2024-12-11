# Dataset Analysis Report

# README.md for Goodreads Dataset Analysis

## Dataset Overview

The dataset, titled `goodreads.csv`, consists of a total of **10,000 entries** and **23 variables** that detail various attributes of books listed on Goodreads. This comprehensive dataset includes information related to book identification, author details, publication specifics, and reader engagement metrics, making it an invaluable resource for data analysis in the literary domain.

### Dataset Structure
- **Shape**: (10,000, 23)
- **Columns**: 
    - `book_id`, `goodreads_book_id`, `best_book_id`, `work_id`, `books_count`, `isbn`, `isbn13`, `authors`, 
    - `original_publication_year`, `original_title`, `title`, `language_code`, `average_rating`, `ratings_count`, 
    - `work_ratings_count`, `work_text_reviews_count`, `ratings_1`, `ratings_2`, `ratings_3`, `ratings_4`, `ratings_5`, 
    - `image_url`, `small_image_url`.

### Data Types
- **Numerical**: With types including `int64` and `float64` for various numerical measures. 
- **Categorical**: Observed as strings for author names, titles, and language codes.

### Missing Values
- Notably, several columns contain missing values:
    - `isbn`: 700 missing entries
    - `isbn13`: 585 missing entries
    - `original_publication_year`: 21 missing entries
    - `original_title`: 585 missing entries
    - `language_code`: 1084 missing entries

## Data Analysis Insights

The analysis of the `goodreads.csv` dataset has yielded several key insights into book performance metrics:

1. **Distribution of Ratings**:
   - The average book rating in this dataset is approximately **4.00**, with a standard deviation of approximately **0.25**, indicating a positively skewed distribution in average ratings.
   - A vast majority of books, approximately **65%**, received an average rating of **4.0 or higher**, suggesting a likely preference for high-quality literature among readers.

2. **Engagement Metrics**:
   - Ratings count (`ratings_count`) has a mean of around **54,000**, with some books receiving as high as **4.78 million ratings**. This emphasizes the active engagement of readers in rating books on the platform.
   - The cumulative total of `work_ratings_count` also shows an average of around **59,687**, underscoring general positive engagement metrics.

3. **Publication Timeline**:
   - Most original publication years center around **2004** to **2011**, which may suggest trends in publishing practices or a surge in popular genres during these years. The dataset includes entries as early as the **18th century** but trends favor more contemporary publications.

4. **Rating Breakdown**:
   - The distribution across the ratings (1 through 5) indicates that the highest occurrence lies within the **4-star** and **5-star** ratings. Specifically:
       - `ratings_4` has a mean value of approximately **19,966**.
       - `ratings_5` has a mean value of **23,790**, substantially higher than those of lower ratings.

## Key Statistical Observations

- **Correlation Analysis**:
   - A **correlation heatmap** visual (refer to `correlation_heatmap.png`) indicates a strong correlation (r > 0.5) between `ratings_count` and `average_rating`, suggesting that books with more ratings tend to have higher average ratings.
  
- **Publication Year vs. Ratings**:
   - Findings suggest that newer publications attract more ratings, illustrated in the **scatter plot** (refer to `scatter_plot.png`), which visually emphasizes the clustering of ratings around more recently published books.

## Potential Business and Research Implications

The insights derived offer valuable perspectives for stakeholders in the book industry:

- **Publishing Trends**: Understanding the relationship between publication years and reader engagement can help publishers identify genres that resonate with audiences over time.

- **Market Strategy**: Analysis can inform marketing strategies, especially for new book releases by leveraging works with high ratings to drive traffic and engagement.

- **Reading Preferences**: By probing deeper into genres associated with high ratings, stakeholders can tailor offerings to meet the evolving preferences of readers.

## Recommendations for Further Investigation

To enhance the understanding surrounding the reading habits and preferences captured in this dataset, it is recommended to explore:

1. **Sentiment Analysis on Reviews**: Utilizing natural language processing techniques on user text reviews to ascertain sentiments and themes present in popular books.

2. **Exploration of Genre-Specific Trends**: Analyzing how different genres perform in terms of reviews and ratings to identify eventual market gaps.

3. **Time-Series Analysis**: This analysis should focus on the publication years to study how ratings change over time for specific books and genres, which could further contribute to understanding the lifecycle of popular literature.

4. **Broader Dataset Integration**: Consider merging this dataset with external data, such as author popularity metrics or sales information, to give a more holistic view of the literary market.

The overall analysis underscores the significance of the dataset as a tool for understanding reader preferences and engagement on the Goodreads platform, providing a foundation for further research and strategic planning in the book industry.

## Visualizations

### correlation_heatmap
![correlation_heatmap.png](correlation_heatmap.png)

### scatter_plot
![scatter_plot.png](scatter_plot.png)

### cluster_plot
![cluster_plot.png](cluster_plot.png)

