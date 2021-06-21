## Summary

- Project Overview:
    - A music recommendation system using an unsupervised, K-means clustering algorithm implemented in Scikit learn.
    - Developed a frontend HTML, CSS, Django & Bootstrap.
    - Developed a backend using Flask.

## Introduction

## Data Gathering

The data used consists of audio features of ~600K+ songs from the Kaggle dataset, covering a range of different genres of music. Although this doesn't contain all of the songs on Spotify, it provides a good starting point to see whether the recommendation system created is functioning as expected.

### About the data

There were 5 different CSV files that give information on a host of useful information that can be used. These files are:

spotify_df → Contains information on the songs from the list below.

artist_data → Contains information on the artists.

data_by_genres → Contains information by the genres.

data_w_genres → Same information as spotify_df, however the genre of each song is included too.

year_data → Contains information about each year from 1920 - 2021.

Spotify provides a host of information about the data that allows for us to analyse the songs. The data consists of information shown below as stated by Spotify:

- Instrumentalness: This value represents the amount of vocals in the song. The closer it is to 1.0, the more instrumental the song is.
- Acousticness: This value describes how acoustic a song is. A score of 1.0 means the song is most likely to be an acoustic one.
- Liveness: This value describes the probability that the song was recorded with a live audience. According to the official documentation “*a value above 0.8 provides strong likelihood that the track is live”*.
- Speechiness: “*Speechiness detects the presence of spoken words in a track*”. If the speechiness of a song is above 0.66, it is probably made of spoken words, a score between 0.33 and 0.66 is a song that may contain both music and words, and a score below 0.33 means the song does not have any speech.
- Energy: “*(energy) represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy*”.
- Danceability: “*Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable*”.
- Valence: “*A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry)*”.

## Data Visualisation

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bb593c46-a890-4b2c-b7c5-c662a7bfa83d/line_graph.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bb593c46-a890-4b2c-b7c5-c662a7bfa83d/line_graph.png)

A visual representation of the change in features over time. Key points that can be seen in the data consists of:

- Acousticness decreasing from 1920 to present.
- A steady increase in energy from 1920 to present.
- Valence, liveness and danceability maintaining value over time.

***Pearson Correlation***

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1f277b0e-497a-480b-b4dc-6a93a8a59bba/pearson.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1f277b0e-497a-480b-b4dc-6a93a8a59bba/pearson.png)

***Top 20 Most Popular Genres***

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3e9eb49a-b0db-4eb1-bc99-f1e1ad505b3f/top_genres.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3e9eb49a-b0db-4eb1-bc99-f1e1ad505b3f/top_genres.png)

An Overview of the most popular genres in the dataset.

***Top 20 Most Popular Artists***

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/87b02137-b7b1-49b7-8c32-74598da10a97/top_artists.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/87b02137-b7b1-49b7-8c32-74598da10a97/top_artists.png)

An Overview of the most popular artists in the dataset.

***Top 20 Most Popular Songs***

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/86fd29bb-0ecb-4b17-b4e1-84fa4af1ba3a/top_songs.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/86fd29bb-0ecb-4b17-b4e1-84fa4af1ba3a/top_songs.png)

An Overview of the most popular songs in the dataset.

## Implementing Machine Learning

In order to get song recommendations for the user, an unsupervised machine learning method will be implemented. Principle Component Analysis (PCA) was used to flatten out the features. K Means clustering will then be used to cluster songs with similar features together, where the Eucidelean distance is calculated to find all the songs that are neighbours to the users input. In this scenario we will be using Scikit Learn to implement the Machine Learning model. 

### Principle Component Analysis (PCA)

**What is PCA?**

Principle Component Analysis (PCA) is a method used to reduce the the dimensionality of large datasets, by transforming the data from large to small while ensuring to keep most of the important details. The reason for this is that smaller datasets are easier to visualise and makes implementing machine learning models much faster. In this scenario we have 10 features that we will scale down to allow for the clustering method to be applied.

**How PCA was implemented:**

```python
# pca pipeline
pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)

projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = spotify_df['name']
projection['cluster'] = spotify_df['cluster_label']

fig = px.scatter(projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
fig.show()
```

**Standardization** 

StandardScaler() displayed in the code above is a method to standardize the data in Scikit learn. This ensures that all the values are evenly distributed from 0 to 1, ensuring that biased results are reduced. For example if the data wasn't scaled, the data with a higher range would be more dominant and therefore wouldn't truly represent the data evenly. 

To read more about this:

[A Step-by-Step Explanation of Principal Component Analysis (PCA)](https://builtin.com/data-science/step-step-explanation-principal-component-analysis)

### K-means clustering

Once the data has been standardized, the next step is to cluster the data. But first, lets have an overview of what K-nearest neighbour is.

**What is K-means clustering?**

K-means is a type of unsupervised learning algorithm. This means that all is known is the input vectors and no knowledge of labels or the outcome. K-means is a method for clustering data points that are aggregated together baed on similarities. 

For read more about this:

[Understanding K-means Clustering in Machine Learning](https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1)

[The Most Comprehensive Guide to K-Means Clustering You'll Ever Need](https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/)

**How K-means clustering was implemented:**

```python
#%% K-Means Clustering
cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=20, n_jobs=-1))])
X = spotify_df.select_dtypes(np.number)
number_cols = list(X.columns)
cluster_pipeline.fit(X)
cluster_labels = cluster_pipeline.predict(X)
spotify_df['cluster_label'] = cluster_labels
```
