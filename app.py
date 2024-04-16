import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
from nltk.stem.porter import PorterStemmer

# Load data
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

# Define converter functions
def converter1(obj):
    # Convert string representation of list to actual list and extract 'name' key
    return [entry['name'] for entry in ast.literal_eval(obj)]

def converter2(obj):
    # Convert string representation of list to actual list and extract 'name' key, limiting to 3 entries
    return [entry['name'] for entry in ast.literal_eval(obj)][:3]

def converter3(obj):
    # Extract 'name' key where 'job' is 'Director'
    return [entry['name'] for entry in ast.literal_eval(obj) if entry['job'] == 'Director']

# Preprocess data
def preprocess_data(movies):
    movies['genres'] = movies['genres'].apply(converter1)
    movies['keywords'] = movies['keywords'].apply(converter1)
    movies['cast'] = movies['cast'].apply(converter2)
    movies['crew'] = movies['crew'].apply(converter3)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['overview'] = movies['overview'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['tags'] = movies['overview'] + movies['cast'] + movies['crew'] + movies['keywords'] + movies['genres']
    return movies

movies = preprocess_data(movies)

# Stemming
ps = PorterStemmer()
def stem(text):
    if isinstance(text, list):
        return " ".join([stem(word) for word in text])  # Join stemmed words into a single string
    else:
        return str(text).lower()  # Convert to lowercase and ensure it's a string


movies['tags'] = movies['tags'].apply(stem)

# Create CountVectorizer
cv = CountVectorizer(max_features=500, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Calculate cosine similarity
similarity = cosine_similarity(vectors)

# Define recommendation function
def recommend(movie):
    if movie not in movies['title'].values:
        return "Movie not found in the database"
    
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = []
    
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]]['title'])
    
    return recommended_movies

# Streamlit app
st.title("Movie Recommender System")

# Sidebar for user input
movie_input = st.text_input("Enter a movie title", "Avatar")

# Button to trigger recommendation
if st.button("Recommend"):
    recommendations = recommend(movie_input)
    st.write("Recommended Movies:")
    for movie in recommendations:
        st.write("-", movie)

# Display session info
if st.session_state.is_active:
    st.session_info()
