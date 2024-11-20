from re import L
#importing libraries
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain
import numpy as numpy
import pandas as pd

#importing datasets
movies = pd.read_csv('/content/movies.csv')
ratings = pd.read_csv('/content/ratings.csv')
links = pd.read_csv('/content/links.csv')
tags = pd.read_csv('/content/tags.csv')

#movies.head()

#merging datasets
datasets = [movies, ratings, tags, links]
merged_ds = datasets[0]

for dataset in datasets[1:]:
    merged_ds = pd.merge(merged_ds, dataset, on='movieId')

#movies.shape
#merged_ds.shape #11 cols as movieId is the common col in all the datasets

#merged_ds.head()
#merged_ds.info()

#to check whether the data has missing and duplicate values
#merged_ds.isnull().sum() #data doesnt have any missing values but if it had then use
#merged_ds.dropna(inplace = True)

merged_ds.duplicated().sum() #no duplicate values

#merged_ds.iloc[0].genres

#to put genres in list
def movie_genres(data):
    L = []
    for i in data.split('|'):
        L.append(i)
    return L

merged_ds['genres'] = merged_ds['genres'].apply(movie_genres)
merged_ds.head()

#distribution of movie ratings in the rating column
plt.hist(merged_ds['rating'], bins=5, edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Number of Movies')
plt.title('Distribution of Movie Ratings in merged dataset')
plt.show()

#top 10 most rated movies
top_rated = merged_ds.groupby('title')['rating'].count().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_rated.values, y=top_rated.index)
plt.title('Top 10 Most Rated Movies')
plt.xlabel('Number of Ratings')
plt.ylabel('Movie Titles')
plt.show()

#genre distribution
all_genres = list(chain.from_iterable(merged_ds['genres']))
genre_counts = pd.Series(all_genres).value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=genre_counts.values[:10], y=genre_counts.index[:10])
plt.title('Top 10 Genres')
plt.xlabel('Frequency')
plt.ylabel('Genres')
plt.show()

#avg rating by genre
genre_avg_rating = merged_ds.explode('genres').groupby('genres')['rating'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_avg_rating.values, y=genre_avg_rating.index)
plt.title('Top 10 Genres by Average Rating')
plt.xlabel('Average Rating')
plt.ylabel('Genres')
plt.show()

#ratings over time
merged_ds['timestamp_x'] = pd.to_datetime(merged_ds['timestamp_x'], unit='s')
ratings_time = merged_ds.groupby(merged_ds['timestamp_x'].dt.year)['rating'].mean()

plt.figure(figsize=(10, 6))
ratings_time.plot(kind='line', color='green')
plt.title('Average Ratings Over Years')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.grid()
plt.show()

#correlation heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(merged_ds[['rating', 'userId_x', 'timestamp_x', 'imdbId', 'tmdbId']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#movies released over year
merged_ds['year'] = merged_ds['title'].str.extract(r'\((\d{4})\)').astype(float)
movies_per_year = merged_ds['year'].value_counts().sort_index()

plt.figure(figsize=(12, 6))
movies_per_year.plot(kind='line', color='skyblue')
plt.title('Number of Movies Released Over Years')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.grid()
plt.show()

#scatter plot betw rating and imdbids
plt.figure(figsize=(10, 6))
sns.scatterplot(x='imdbId', y='rating', data=merged_ds, alpha=0.5, color='purple')
plt.title('Ratings vs IMDb IDs')
plt.xlabel('IMDb ID')
plt.ylabel('Ratings')
plt.show()

#removing and checking for duplicates
merged_ds = merged_ds.drop_duplicates(subset='movieId', keep='first')
duplicate_count = merged_ds.duplicated(subset='movieId').sum()


#created function to display top 20 movie suggestions by genre and showing tags for selected movie
def suggest_movies_by_genre(df, tags_df):
    genres = sorted(set([genre for sublist in df['genres'] for genre in sublist]))
    print("\nAvailable genres:", ", ".join(genres))

    user_genre = input("\nSelect a genre: ").strip().lower()
    filtered_movies = df[df['genres'].apply(lambda x: user_genre in [genre.lower() for genre in x])]
    
    top_20_movies = filtered_movies[['title', 'genres', 'rating']].head(20)
    top_20_movies['Serial No.'] = range(1, 21)
    
    print(f"\nTop 20 Movie Suggestions for '{user_genre.capitalize()}' genre:")
    print(top_20_movies[['Serial No.', 'title']].to_string(index=False)) 

    movie_serial = int(input("\nEnter the serial number to see the tags associated with the movie: "))
    selected_movie = top_20_movies.iloc[movie_serial - 1]
    selected_movie_id = df[df['title'] == selected_movie['title']]['movieId'].values[0]

    movie_tags = tags_df[tags_df['movieId'] == selected_movie_id]
    print(f"\nTags for '{selected_movie['title']}':")
    print(", ".join(movie_tags['tag'].values) if not movie_tags.empty else "No tags available")

suggest_movies_by_genre(merged_ds, tags)


#created function to get movie suggestions based on rating
def suggest_movies_by_rating(df):
    user_rating = float(input("\nEnter a rating to get movie suggestions (1 to 5): ").strip())
    filtered_movies = df[df['rating'] == user_rating]
    
    print(f"\nTop 10 Movie Suggestions with rating {user_rating}:")
    print(filtered_movies[[ 'title', 'genres']].head(10).to_string(index=False))  # Top 10 rows

suggest_movies_by_rating(merged_ds)




