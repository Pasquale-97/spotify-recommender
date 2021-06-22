# Spotify Music Recommendation System in Scikit Learn
## Demo


https://user-images.githubusercontent.com/66837999/122845974-c4c88800-d2fc-11eb-9e22-0d5024fc314e.mov





## Summary

- Project Overview:
    - A music recommendation system using an unsupervised, K-means clustering algorithm implemented in Scikit learn.
    - Developed a frontend HTML, CSS, Django & Bootstrap.
    - Developed a backend using Flask.

## Introduction

Spotify has been responsible for many new songs that enter my life on a weekly basis through their discover weekly playlists and this is all done through using AI. This is my attempt at using values of songs labelled with a whole host of features (see below) from Spotify themselves and using these features to create recommendations on new songs. Creating my own way of discovering new songs. This is an end to end project, from analysing the data to creating a frontend, to deploying on Google Cloud. I hope you enjoy! 

## Data Gathering

The data used consists of audio features of ~600K+ songs from [Kaggle](https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks/code?datasetId=670134&sortBy=voteCount) downloaded using the Spotify API, covering a range of different genres of music. Although this doesn't contain all of the songs on Spotify, it provides a good starting point to see whether the recommendation system created is functioning as expected.

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

![line_graph](https://user-images.githubusercontent.com/66837999/122840493-1d465800-d2f2-11eb-8542-26f2f44da9ca.png)


A visual representation of the change in features over time. Key points that can be seen in the data consists of:

- Acousticness decreasing from 1920 to present.
- A steady increase in energy from 1920 to present.
- Valence, liveness and danceability maintaining value over time.

***Pearson Correlation***

![pearson](https://user-images.githubusercontent.com/66837999/122840537-351ddc00-d2f2-11eb-9957-3f470dba49ad.png)


***Top 20 Most Popular Genres***

![top_genres](https://user-images.githubusercontent.com/66837999/122840550-3cdd8080-d2f2-11eb-88ab-34af958642ae.png)


An Overview of the most popular genres in the dataset.

***Top 20 Most Popular Artists***

![top_artists](https://user-images.githubusercontent.com/66837999/122840555-41a23480-d2f2-11eb-99c5-c1b8389e1a58.png)


An Overview of the most popular artists in the dataset.

***Top 20 Most Popular Songs***

![top_songs](https://user-images.githubusercontent.com/66837999/122840562-4666e880-d2f2-11eb-9ad4-6e064b97b6d9.png)


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

To read more about this:

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

The result shows how the data was clustered and as you can see there are some clear distinctions between the clusters available

![spotify-cluster](https://user-images.githubusercontent.com/66837999/122840684-89c15700-d2f2-11eb-97b7-0f64d1ae8b2e.png)


### Eucidelean Distance

**What is Eucidelean Distance?**

Eucidelean Distance is a method used to determine the shortest distance between two points and is a common method used in K-means clustering. 

**How it was implemented**

We've standardized the data and clustered them using K-means which has now allowed for us to determine the distance from the users input and the closest points to their input. We can do that by calculating the Eucidelean distance and display all the closes points around the users input.

To read about how this was implemented I have attached the Kaggle notebook that helped me complete this part:

[Music Recommendation System using Spotify Dataset](https://www.kaggle.com/vatsalmavani/music-recommendation-system-using-spotify-dataset#Clustering-Genres-with-K-Means)

To learn more about what **Eucidelean** distance is here is a short video:

[Euclidean Distance - Practical Machine Learning Tutorial with Python p.15](https://www.youtube.com/watch?v=hl3bQySs8sM&ab_channel=sentdex)

## Creation of web application

One important thing I've learned when creating machine learning applications is how important it is to create a frontend to the model created. A frontend provides a more interactive experience for the user compared to leaving it in a Jupyter notebook and allows anyone to play around with the model created. To create the frontend for the application, a Flask backend was implemented with a HTML, CSS and Bootstrap frontend to provide an interactive experience for the user. 

### Backend → Flask

- Get → Retrieve an element

```python
if request.method == 'GET':
        return(render_template('index.html'))
```

Get is used for retrieving elements. In this case we use it to retrieve the index.html file that contains the home screen that will be shown to the user when they access the site.

- Post → Add an element

The post method is used for adding elements. In this scenario we use the post element whenever there is updates to the website. This includes when the user interacts with the submit button, add song button and the clear button. By using the POST method we are able to display to the user the updates to the website based on their input. The functionality of how each button is implemented is explained below.

### Frontend → HTML, CSS & Bootstrap

*Insert screenshot of application here*

**Index.html**

The Index.html file contains the structure of the website as well as the additional code for updating the website based on the users input.

```python
<!DOCTYPE html>

<head>
  <link rel="stylesheet" type="text/css" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
  <link rel="stylesheet" type="text/css" href="./static/css/index.css">
  <title>Spotify Recommender System</title>
</head>

<div class="container">
<div class="row justify-content-center mt-5">
<form class="text-center border border-light p-5" action="{{ url_for('main') }}" method="POST">
  <fieldset>
      <h1><mark class="green">Spotify</mark>Recommender System</h1>
      <br>
      <!-- <label>Enter Song:</label> -->
      <input name="song" type="text" required>
      <br><br>
      <input type="submit" name="submit_button" value="Add Song" id="submit_button">
      <input type="submit" name="submit_button" value="Create Playlist" formnovalidate>
      <input type="submit" name="submit_button" value="Clear" formnovalidate>
  </fieldset>
</form>
</div>
</div>
<br>

<div class="input_songs" align="center">
  <table class="table">
      <tr class="table_header">
        {% if input_songs %}
          <th class="table_cell">Songs</th>
        {% endif %}
      </tr>

      {% for row in input_songs %} 
      <tr class="table_row">
          <td class="table_cell"> {{ row }} </td>
      </tr>
      {% endfor %}
  </table>
</div>

<div class="result" align="center">
<table class="table">
    <tr class="table_header">
      {% for header in headings %}
      <th class="table_cell">{{ header }}</th>
      {% endfor %}
    </tr>

    {% for row in result %} 
    <tr class="table_row">
      {% for cell in row %}
        <td class="table_cell"> {{ cell }} </td>
      {% endfor %}
    </tr>
    {% endfor %}
</table>
</div>

<div class="songs_list" align="center">
  {% if songs_list %}
     <p style="font-size:16px">{{ songs_list }}</p>
  {% endif %}
</div>

</html>
```

**Index.css**

CSS was used to provide a clean site for the user to interact with. The CSS used can be found below:

```python
h1{
  padding-top: 7%;
  color: #757575;
}

mark.green{
  color: #1DB954;
  background-color: transparent;
}

label{
  color: #757575;
}

input{
  border: none;
  padding: 10px;
  padding-left: 3.5%;
  margin: 10px;
  width: 50%;
  border:1px solid #eaeaea;
  border-radius: 50px 50px 50px 50px;
  outline:none;
  background-image:url(../images/search.png); 
  background-repeat: no-repeat;
  background-position: left 10px center;
  background-size: 20px;
  

  /* background-image: url('../images/search.png');
  background-repeat: no-repeat;
  background-position-y: center;
  padding-left:45px;  */
  
}
input:hover{
  border-color: #a0a0a0 #b9b9b9 #b9b9b9 #b9b9b9;
}
input:focus{
  border-color:#4d90fe;
}

input[type="submit"] {
  background: #f2f2f2;
  border: 1px solid #f2f2f2;
  color: #757575;
  cursor: default;
  font-size: 14px;
  font-weight: bold;
  width: 140px;
  padding: 0 16px;
  height:36px;
  border-radius: 7px 7px 7px 7px;
}

input[type="submit"]:hover {
  box-shadow: 0 1px 1px rgba(0,0,0,0.1);
  background: #f8f8f8;
  border: 1px solid #1DB954;
  box-shadow: 0 1px 1px rgba(0,0,0,0.1);
  color: #1DB954;
}

.table_row:nth-child(even){
  background-color: #f8f8f8;
}

.table_cell {
  padding: 10px;
  text-align: center;
}
```

**Submit button**

The submit button takes all the songs that the user has input and runs it through the machine learning model. By pairing this with the index.html file using Django, the results of the model is displayed to the user in a table. Regex is used for ensuring that all unwanted function is removed to provide a cleaner look.

```python
if request.form['submit_button'] == 'Create Playlist':
            song = request.form['song']

            if not list_of_songs:
                message = "Please enter at least one song."
                return render_template('index.html', songs_list=message)

            prediction = recommend_songs(list_of_songs, spotify_df)
            order_prediction = sorted(prediction, key=lambda k: k['popularity'], reverse=True) 

            headings = ("Songs", "Artists", "Year", "Popularity")
            
            tuple_of_songs = [(d['name'], re.sub(r"[\[\]\'\"]", "", d['artists']),d['year'],d['popularity']) for d in order_prediction]

            
            return render_template('index.html',
                            headings = headings,
                            result=tuple_of_songs)
```

**Add song button**

The add button is implemented so that when the user inputs their songs it is appended to a dictionary. The dictionary was used so that all the songs that are input are ready to be run through the K-means clustering model. This request is paired with the HTML file with Django to display the results to the user. Some filtering was applied to ensure that the song is displayed cleanly.

```python
if request.form['submit_button'] == 'Add Song':
            song = request.form['song']
            dictionary_song = {'name': song}
            if dictionary_song not in list_of_songs:
                list_of_songs.append(dictionary_song)
            
            filter_songs = [re.sub(r"[\[\]\'\"]", "", song['name']) for song in list_of_songs]
            
            return render_template('index.html', input_songs=filter_songs)
```

**Clear button**

The clear button is implemented so that the user is able to clear all the songs they have input.

```python
if request.form['submit_button'] == 'Clear':
            list_of_songs.clear()

            return render_template('index.html', result=list_of_songs)
```

## Deploying web application to Cloud Service

*Coming Soon*
