import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load datasets
users_data = pd.read_csv(r"C:\Users\User\Desktop\MovieLens Project\data\movielens_users_data.csv.gz")
movies_data = pd.read_csv(r"C:\Users\User\Desktop\MovieLens Project\data\movielens_movies_ext_filtered.csv.gz")
movie_attributes = pd.read_csv(r"C:\Users\User\Desktop\MovieLens Project\data\movielens_movies_ext_filtered_attributes.csv.gz")

# Merge movie attributes with the main movie data
movies_data = pd.merge(
    movies_data,
    movie_attributes,
    on="movie_id",
    how="left",
    suffixes=("_movies", "_attributes")
)

# Keep necessary columns and avoid duplicates
movies_data = movies_data[[
    "tconst_movies", "movie_id", "primaryTitle", "year_movies",
    "runtimeminutes", "genres_movies", "imdb_avg_rating", "imdb_num_votes", "genres_code"
]]

# Rename columns for consistency
movies_data.rename(columns={
    "tconst_movies": "tconst",
    "year_movies": "year",
    "genres_movies": "genres"
}, inplace=True)

# Fill missing titles
movies_data["primaryTitle"].fillna("Unknown Title", inplace=True)

# Create a combined title with the year
movies_data["title_with_year"] = movies_data["primaryTitle"] + " (" + movies_data["year"].astype(str) + ")"

# Load ratings data
ratings_data = pd.read_csv(r"C:\Users\User\Desktop\MovieLens Project\data\movielens_ratings_data.csv.gz")
watch_history = ratings_data[['user_id', 'movie_id']]

# Load the recommendation model
model = load(r"C:\Users\User\Desktop\MovieLens Project\models\movielens_user_movies_rating_hist_gradient_boosting.joblib")

# Age bucketing function
def bucketize_age(age):
    bins = [0, 18, 25, 35, 45, 50, 56, 100]
    labels = [1, 18, 25, 35, 45, 50, 56]
    return labels[np.digitize(age, bins) - 1]

# Create user watch list
def create_user_watch_list(users_data, watch_history):
    watch_list = watch_history.groupby("user_id")["movie_id"].apply(set).to_dict()
    return watch_list

# Recommendation function
def recommend_movies(user_features, movies_data, seen_movies, model, threshold, max_size):
    feature_order = [
        "year", "runtimeminutes", "genres_code", "imdb_avg_rating", "imdb_num_votes",
        "user_zip_code", "bucketized_user_age", "user_occupation_label", "user_gender_code"
    ]

    movie_features = movies_data.drop("movie_id", axis=1)
    user_repeated = pd.concat([user_features] * len(movies_data), ignore_index=True)

    input_features = pd.concat([movie_features.reset_index(drop=True), user_repeated], axis=1)
    input_features = input_features[feature_order]  # Ensure correct column order

    predicted_ratings = model.predict(input_features)

    movie_scores = sorted(zip(movies_data["movie_id"], predicted_ratings), key=lambda x: x[1], reverse=True)

    recommendations = []
    for movie_id, rating in movie_scores:
        if len(recommendations) >= max_size:
            break
        if movie_id in seen_movies:
            continue
        imdb_votes = movies_data.loc[movies_data["movie_id"] == movie_id, "imdb_num_votes"].values[0]
        if rating > threshold and imdb_votes >= 1000:
            recommendations.append(movie_id)
    
    return recommendations

# Get movie attributes
def get_movie_attributes(recommendation_list, movie_data):
    return movie_data[movie_data["movie_id"].isin(recommendation_list)]

# Streamlit UI
st.title("🎬 Movie Recommender")

# User input
st.subheader("Enter Your Details")
gender = st.radio("Select Gender:", ["Male", "Female"])
age = st.number_input("Enter Age:", min_value=1, max_value=100, step=1)
bucketized_age = bucketize_age(age)
occupation = st.selectbox("Select Occupation:", users_data["user_occupation_text"].unique())
zip_code = st.selectbox("Select Zip Code:", users_data["user_zip_code"].unique())

# Movie selection
st.subheader("Select Movies You Have Watched")
watched_movies = st.multiselect("Choose movies:", movies_data["title_with_year"].tolist())

# Recommendation preferences
st.subheader("Recommendation Preferences")
max_recommendations = st.slider("Number of recommendations:", 1, 10, 5)
threshold = st.slider("Minimum rating threshold:", 1.0, 5.0, 3.5, 0.1)

# Generate recommendations
if st.button("Get Recommendations"):
    user_features = pd.DataFrame([{
        "user_gender_code": 1 if gender == "Male" else 0,
        "bucketized_user_age": bucketized_age,
        "user_occupation_label": users_data[users_data["user_occupation_text"] == occupation]["user_occupation_label"].values[0],
        "user_zip_code": zip_code
    }])

    watch_list = set(movies_data[movies_data["title_with_year"].isin(watched_movies)]["movie_id"].tolist())
    recommendations = recommend_movies(user_features, movies_data, watch_list, model, threshold, max_recommendations)
    recommended_movies = get_movie_attributes(recommendations, movies_data)

    st.subheader("🎥 Recommended Movies")
    st.write(recommended_movies[["primaryTitle", "year", "genres", "imdb_avg_rating"]])
