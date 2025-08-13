import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re


def load_data():
    """
    Loads the data within the movies.dat, ratings.dat and users.dat files and stores them in a table. It also one-hot econdes 
    the genres within movies.dat.

    Returns:
        users (pd.DataFrame): User dataset
        movies (pd.DataFrame): Movie dataset with genre one=hot encoding
        ratings (pd.DataFrame): User-Movie dataset
    
    
    """
    users = pd.read_csv("film_recommendation/data/users.dat", sep = "::", engine="python", header=None)
    users.columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']  # Assigns column names

    movies = pd.read_csv("film_recommendation/data/movies.dat", sep = "::", engine="python", header=None, encoding='latin-1')
    movies.columns = ['MovieID', 'Title', 'Genres'] # Assigns column names

    # One-hot encodes the genres column (split by '|') into seperate binary columns
    genre_dummies = movies['Genres'].str.get_dummies(sep='|')
    movies = pd.concat([movies, genre_dummies], axis=1)
    movies.drop('Genres', axis=1, inplace=True) # Removes the genres column for read-ability

    ratings = pd.read_csv("film_recommendation/data/ratings.dat", sep= "::", engine="python", header=None)
    ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'] # Assigns column names
    return users, movies, ratings


def create_user_item_matrix(ratings):
    """
    Creates a user item matrix in which users are the rows and the films are the columns

    Args:
        ratings (pd.DataFrame): DataFrame containing users and their film ratings

    Returns:
        pd.DataFrame: User-Item matrix
    """
    return ratings.pivot(index = 'UserID', columns = 'MovieID', values = 'Rating')


def normalise_ratings(rating_matrix):
    """
    Normalises the ratings given by users by centering them around 0 to remove bais

    Args:
        ratings_matrix (pd.DataFram): DataFrame containing the user rating matrix
    
    Returns:
        rating_norm (pd.DataFrame): Normalised ratings
        user_mean (pd.Series): Users means
    """
    user_mean = rating_matrix.mean(axis = 1) # Compute the mean rating for each user
    # Subtract user mean from their ratings to center data
    rating_norm = rating_matrix.sub(user_mean, axis = 0) 
    return rating_norm, user_mean


def compute_user_similarity(rating_norm):
    """
    Uses Pearsons correlation to calculates how similiar users ratings are to each others
    
    Args:
        rating_norm (pd.DataFrame): contains the normalised ratings of each user
    
    Returns:
        similarity (pd.dataFrame): A matrix containing the pearson correlation between each user
    """  
    # Transposes the rating_norm matrix so the rows are the films and calculates the pearsons correlation between said films
    similarity = rating_norm.T.corr(method = 'pearson') 
    # checks to see if the shape is square so there are no missing values
    if similarity.shape[0] == similarity.shape[1]:
        np.fill_diagonal(similarity.to_numpy(), 0) # fills diagonals with 0s
    return similarity


def predict_ratings(rating_matrix, similarity_matrix, user_means, k=10):
    """
    Calculates predicted ratings for films using the k most similar users.

    Args:
        rating_matrix (pd.DataFrame): DataFrame of users (rows) and their film ratings (columns).
        similarity_matrix (pd.DataFrame): DataFrame containing similarity scores between users.
        user_means (pd.Series): Series with the mean rating for each user.
        k (int): Number of nearest neighbors to consider for prediction.

    Returns:
        pd.DataFrame: Predicted ratings for all users and films.
    """
    rating_filled = rating_matrix.fillna(0)  # Replace missing ratings with zero
    pred = []
    for user in rating_matrix.index:
        sim_score = similarity_matrix.loc[user]  # Similarity scores of current user to all others
        top_k_users = sim_score.sort_values(ascending=False).head(k)  # Top k most similar users
        sim_weights = top_k_users.values  # Similarity weights for these neighbors
        ratings = rating_filled.loc[top_k_users.index]  # Ratings by top k neighbors
        weighted_sum = ratings.T.dot(sim_weights)  # Weighted sum of neighbors' ratings per movie
        sum_of_weights = np.abs(sim_weights).sum()  # Sum of absolute similarity weights

        # Predicted normalized ratings; fallback to user's mean if no neighbors found
        pred_ratings = weighted_sum / sum_of_weights if sum_of_weights > 0 else user_means[user]

        # Add back user's mean rating to denormalize and store predictions
        pred.append(pred_ratings + user_means[user])

    # Return predictions as DataFrame with users as rows and movies as columns
    return pd.DataFrame(pred, index=rating_matrix.index, columns=rating_matrix.columns)


def get_rmse(predictions, test_data):
    """
    Calculates the Root Mean Squared Error (RMSE) between the predicted ratings of user film pairs and their actual ratings.

    Args:
        predictions (pd.DataFrame): A DataFrame containing predicted ratings indexed by user and movie IDs.
        test_data (pd.DataFrame): A DataFrame containing the actual user ratings with 'UserID', 'MovieID', and 'Rating' columns.

    Returns:
        float: RMSE score indicating the accuracy of the predictions DataFrame.
    """
    pred = []
    actual = []

    for row in test_data.itertuples():
        try:
            # Get predicted rating for the rows user film pair
            pred_rating = predictions.loc[row.UserID, row.MovieID]

            # Only consider if there is a prediction for the rating of a film  
            if not np.isnan(pred_rating):
                pred.append(pred_rating)
                actual.append(row.Rating)

        except KeyError:
            # If a prediction is missing, skip this pair
            continue

    # Calculate RMSE between actual and predicted ratings
    return np.sqrt(mean_squared_error(actual, pred))


def calc_rmse():
    """
    Runs the full pipeline for user-based collaborative filtering and then computes RMSE to give an insight into how well the 
    model has performed.

    Steps:
    - Load the .dat files to get the user, movie, and rating data
    - Split ratings into training and test sets
    - Create user-item matrix from the training data
    - Normalize the newly created user ratings
    - Compute user similarity using Pearson correlation
    - Predict ratings using top-k neighbors
    - Evaluate the accuracy of the predictions using RMSE

    Returns:
        None (prints RMSE to console)
    """
    # Load the data
    users, movies, ratings = load_data()

    # Split into training and test sets
    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

    # Create user-item matrix from training data
    ratings_matrix = create_user_item_matrix(train_data)

    # Normalize ratings to remove user bias
    rating_norm, user_mean = normalise_ratings(ratings_matrix)

    # Compute user similarity (Pearson correlation)
    user_similarity = compute_user_similarity(rating_norm)

    # Predict ratings using top 20 most similar users
    predictions = predict_ratings(ratings_matrix, user_similarity, user_mean, k=20)

    # Evaluate prediction accuracy using RMSE
    rmse = get_rmse(predictions, test_data)

    print(f"User-based CF RMSE: {rmse:.4f}")


def prep_model():
    """
    Loads and prepares data for collaborative filtering.

    This function performs the following steps:
    - Loads users, movies, and ratings data
    - Sets MovieID as the index of the movies DataFrame
    - Creates a user-item rating matrix
    - Normalizes ratings by subtracting each user's mean
    - Computes user-user similarity matrix using Pearson correlation

    Returns:
        users (pd.DataFrame): User dataset
        movies (pd.DataFrame): Movie dataset with MovieID as index
        ratings (pd.DataFrame): Raw user-movie ratings
        ratings_matrix (pd.DataFrame): User-item matrix of ratings
        user_mean (pd.Series): Mean rating for each user
        user_similarity (pd.DataFrame): User-user similarity matrix
    """
    # Load the user, movie, and ratings datasets
    users, movies, ratings = load_data()

    # Set MovieID as index for easier access to the values
    movies = movies.set_index('MovieID')

    # Create user-item rating matrix (users as rows, movies as columns)
    ratings_matrix = create_user_item_matrix(ratings)

    # Normalize ratings by subtracting user means (centering)
    rating_norm, user_mean = normalise_ratings(ratings_matrix)

    # Compute user-user similarity using Pearson correlation
    user_similarity = compute_user_similarity(rating_norm)

    return users, movies, ratings, ratings_matrix, user_mean, user_similarity

def extract_year(title):
    """
    Extracts the year from a film title by using regular expression to  search for a 4-digit year enclosed in parentheses,
     e.g. "Toy Story (1995)". If no year is present, it returns NaN.

    Args:
        title (str): The full movie title (e.g. "The Matrix (1999)")

    Returns:
        int or float: The extracted year as an integer, or np.nan if not found
    """
    match = re.search(r'\((\d{4})\)', title)  # Look for a 4-digit year encased within parentheses
    return int(match.group(1)) if match else np.nan


def recommend_from_title(
    liked_films, movies, ratings, ratings_matrix, user_mean, user_similarity, 
    k=10, n_recs=5, genre_weight=0.6, year_weight=0.2, rating_weight=0.2, genre_threshold=0.3
):
    """
    Generates movie recommendations based on a list of liked titles.

    Combines collaborative filtering (user-based) with content-based factors (genre and release year).
    
    Args:
        liked_films (list): Titles of movies the user likes
        movies (pd.DataFrame): Movie metadata (indexed by MovieID)
        ratings (pd.DataFrame): Ratings data
        ratings_matrix (pd.DataFrame): User-item matrix of ratings
        user_mean (pd.Series): Mean rating for each user
        user_similarity (pd.DataFrame): Similarity matrix between users
        k (int): Number of top similar users to consider
        n_recs (int): Number of final recommendations to return
        genre_weight (float): Weight for genre similarity
        year_weight (float): Weight for year similarity
        rating_weight (float): Weight for predicted rating
        genre_threshold (float): Minimum genre similarity required to consider a movie

    Returns:
        pd.DataFrame: Top recommended movies with final scores
    """

    # Gets the MovieIDs of the liked films
    liked_id = movies[movies['Title'].isin(liked_films)].index.tolist()
    if not liked_id:
        return pd.DataFrame(columns=['Title', 'final_score'])
       
    # Initialize similarity scores (to accumulate collaborative scores)
    sim_scores = pd.Series(0, index=user_similarity.columns)

    # Accumulate similarity scores from users who liked the same movies
    for movie_id in liked_id:
        # gathers users who rated the current movie_id highly
        users_who_like = ratings[(ratings['MovieID'] == movie_id) & (ratings['Rating'] >= 4)]['UserID']
        if users_who_like.empty:
            continue
        # For each user who liked this movie, add their similarity scores to sim_scores 
        sim_scores += user_similarity.loc[users_who_like].sum(axis=0)

    # handles the case where no similiar users were found
    if sim_scores.sum() == 0:
        return pd.DataFrame(columns=['Title', 'final_score'])

    # gathers the k most similiar users to the current user
    top_k_users = sim_scores.sort_values(ascending=False).head(k).index
    # Compute predicted weighted ratings by taking the dot product of top k users' ratings and their similarity scores
    weighted_ratings = ratings_matrix.loc[top_k_users].T.dot(sim_scores.loc[top_k_users])
    sum_sims = sim_scores[top_k_users].sum()
    if sum_sims == 0:
        return pd.DataFrame(columns=['Title', 'final_score'])

    predicted_ratings = weighted_ratings / sum_sims # calculates predicted rating
    predicted_ratings = predicted_ratings.drop(liked_id, errors='ignore') # Don't recommend already liked movies

    # Merge predicted ratings with movie titles
    recommendations_df = movies.loc[movies.index.intersection(predicted_ratings.index), ['Title']].copy() 
    recommendations_df['predicted_rating'] = predicted_ratings.loc[recommendations_df.index]

    # Get all genre columns (exclude 'Title')
    genre_cols = movies.columns.difference(['Title'])

    # Compute the average genre profile across all liked movies
    liked_genres = movies.loc[liked_id, genre_cols].mean()
    liked_genres_vector = liked_genres.values.reshape(1, -1)

    # Extract genre vectors for candidate movies
    candidate_genres = movies.loc[recommendations_df.index, genre_cols].values

    # Compute cosine similarity between liked genres and candidate genres
    genre_sim_scores = cosine_similarity(candidate_genres, liked_genres_vector).flatten()

    # Add genre similarity scores to recommendation DataFrame
    recommendations_df['genre_sim'] = genre_sim_scores

    # Filter out movies that are too dissimilar in genre to the liked films
    recommendations_df = recommendations_df[recommendations_df['genre_sim'] >= genre_threshold]
    if recommendations_df.empty:
        return pd.DataFrame(columns=['Title', 'final_score'])

    # Extract release years from liked and candidate movie titles for year similarity scoring
    liked_years = np.array([extract_year(title) for title in movies.loc[liked_id, 'Title']])
    candidate_years = np.array([extract_year(title) for title in recommendations_df['Title']])

    def year_similarity(candidate_years, liked_years, sigma=5):
        """
        Computes similarity scores between candidate movie years and the average year of liked movies,
        using a Gaussian (RBF) kernel to model similarity.

        Args:
            candidate_years (np.ndarray): Array of years corresponding to candidate movies.
            liked_years (np.ndarray): Array of years from movies the user has liked.
            sigma (float, optional): Standard deviation parameter controlling the sharpness of similarity decay.
            Lower values penalize larger year differences more strongly. Default is 5.

        Returns:
            np.ndarray: An array of similarity scores (between 0 and 1) for each candidate year.
        """
        liked_avg_year = np.nanmean(liked_years)  # Mean release year of the liked movies ignoring any NaNs.
        dist = np.abs(candidate_years - liked_avg_year)  # Year difference from each candidate to liked mean
        sim = np.exp(- (dist ** 2) / (2 * sigma ** 2))  # Apply Gaussian similarity function
        return sim

    # Compute year similarity scores and assign to DataFrame
    recommendations_df['year_sim'] = year_similarity(candidate_years, liked_years)

    # Ensure there's data to work with before continuing
    if recommendations_df.empty or len(recommendations_df) < 1:
        return pd.DataFrame(columns=['Title', 'final_score'])

    try:
        # Normalizes the scores components to the range [0, 1] using the MinMaxScaler, in order to make them comparable
        scaler = MinMaxScaler()
        recommendations_df[['predicted_rating', 'genre_sim', 'year_sim']] = scaler.fit_transform(
            recommendations_df[['predicted_rating', 'genre_sim', 'year_sim']]
        )
    except ValueError:
        # Handle edge case where scaling fails (e.g., all columns have the same value)
        return pd.DataFrame(columns=['Title', 'final_score'])

    # Compute the final recommendation score as a weighted sum of rating, genre, and year similarities
    recommendations_df['final_score'] = (
        rating_weight * recommendations_df['predicted_rating'] +
        genre_weight * recommendations_df['genre_sim'] +
        year_weight * recommendations_df['year_sim']
    )

    # Sort by score and return top recommendations
    recommendations_df = recommendations_df.sort_values('final_score', ascending=False).head(n_recs)

    return recommendations_df[['Title', 'final_score']]




lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def process(val):
    """
    Processes a text string by tokenizing, converting to lowercase, removing stopwords,
    and lemmatizing each token.

    Args:
        val (str): Input text string to process.

    Returns:
        set: A set of processed tokens (lemmas) without stopwords.
    """
    tokens = word_tokenize(val.lower())  # Tokenize and convert to lowercase
    lemmas = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]  # Lemmatize and filters out any stopwords
    return set(lemmas)

def jaccard_similarity(set1, set2):
    """
    Calculates the Jaccard similarity between two sets.

    Args:
        set1 (set): First set of tokens.
        set2 (set): Second set of tokens.

    Returns:
        float: Jaccard similarity coefficient (intersection / union).
               Returns 0 if both sets are empty.
    """
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0
    return len(intersection) / len(union)

def jaccard_similarity_titles(user_title, movie_titles, threshold=0.2):
    """
    Finds movie titles from a list that closely match what the user typed based on Jaccard similarity.

    Args:
        user_title (list of str): List of titles provided by the user.
        movie_titles (list of str): List of movie titles to match against.
        threshold (float): Minimum Jaccard similarity score required to consider a match.

    Returns:
        list of str: Movie titles from `movie_titles` that match user titles above the threshold.
    """
    matched = []

    # Preprocess all movie titles once for efficient comparison
    processed_films = [(title, process(title)) for title in movie_titles]

    for raw_title in user_title:
        tokens_input = process(raw_title)  # Preprocess user title
        
        best_match = None
        best_score = 0

        # Compare with each movie title's processed tokens to find the best match
        for title, tokens_films in processed_films:
            score = jaccard_similarity(tokens_input, tokens_films)
            if score > best_score:
                best_match = title
                best_score = score
        
        # Add the best match if it meets the similarity threshold
        if best_score >= threshold:
            matched.append(best_match)

    return matched

def recommend_content_based(liked_films, movies, n_recs=5):
    """
    Generates content-based movie recommendations based on genre similarity.

    Args:
        liked_films (list of str): List of movie titles the user likes.
        movies (pd.DataFrame): DataFrame containing movie data including one-hot encoded genres.
        n_recs (int): Number of recommendations to return.

    Returns:
        pd.DataFrame: DataFrame with recommended movie titles and their content-based similarity scores.
    """
    # Ensure 'MovieID' is set as the index for consistent lookup
    if movies.index.name != 'MovieID':
        movies = movies.set_index('MovieID')

    genre_cols = movies.columns.difference(['Title'])  # Extract one-hot encoded genre columns

    # Get only the films that are liked and in the DataFrame
    liked_movies = movies[movies['Title'].isin(liked_films)]
    if liked_movies.empty:
        return pd.DataFrame()  # Return empty DataFrame if no liked films found

    # Extract genre vectors
    liked_genres = liked_movies[genre_cols].values
    all_genres = movies[genre_cols].values

    # Compute cosine similarity between all movies and liked movies
    sim_matrix = cosine_similarity(all_genres, liked_genres)

    # Use the highest similarity score for each candidate movie
    sim_scores = sim_matrix.max(axis=1)

    sim_scores_df = pd.Series(sim_scores, index=movies.index)

    # Remove already liked movies from recommendations
    liked_ids = liked_movies.index
    sim_scores_df = sim_scores_df.drop(liked_ids, errors='ignore')

    # Get top N recommendations
    top_scores = sim_scores_df.sort_values(ascending=False).head(n_recs * 2)

    return pd.DataFrame({
        'Title': movies.loc[top_scores.index, 'Title'],
        'cb_score': top_scores.values
    })

def hybrid_recommendation(liked_films, movies, ratings, rating_matrix, user_mean, user_similarity, 
                          k=20, n_recs=10, cf_weight=0.5, cb_weight=0.5):
    """
    Generates hybrid film recommendations by combining collaborative filtering (CF)
    and content-based filtering (CBF) scores.

    Args:
        liked_films (list of str): List of film titles the user likes.
        movies (pd.DataFrame): Film metadata with one-hot encoded genres.
        ratings (pd.DataFrame): User-movie rating interactions.
        rating_matrix (pd.DataFrame): User-item rating matrix.
        user_mean (pd.Series): Average rating per user.
        user_similarity (pd.DataFrame): Similarity matrix between users.
        k (int): Number of similar users to consider in CF.
        n_recs (int): Number of final recommendations to return.
        cf_weight (float): Weight for collaborative filtering score.
        cb_weight (float): Weight for content-based filtering score.

    Returns:
        list of str: List of recommended movie titles sorted by hybrid score.
    """

    # Get collaborative filtering recommendations
    cf_recs_df = recommend_from_title(
        liked_films, movies, ratings, rating_matrix, user_mean, user_similarity,
        k=k, n_recs=n_recs * 2  # Over-generate for more overlap with CBF
    )
    
    # Handle case where CF gives no recommendations
    if cf_recs_df is None or cf_recs_df.empty:
        cf_recs_df = pd.DataFrame(columns=['Title', 'final_score'])
    cf_recs_df = cf_recs_df.rename(columns={'final_score': 'cf_score'})

    # Get content-based filtering recommendations
    cb_recs_df = recommend_content_based(liked_films, movies, n_recs=n_recs * 2)

    # Handle case where CBF gives no recommendations
    if cb_recs_df is None or cb_recs_df.empty:
        cb_recs_df = pd.DataFrame(columns=['Title', 'cb_score'])

    # Merge CF and CBF recommendations by movie title
    hybrid_df = pd.merge(cf_recs_df, cb_recs_df, on='Title', how='outer')

    # Replace missing scores with 0 to avoid NaNs during scaling
    hybrid_df['cf_score'] = hybrid_df['cf_score'].fillna(0)
    hybrid_df['cb_score'] = hybrid_df['cb_score'].fillna(0)

    # Normalize scores to ensure fair contribution to final score
    scaler = MinMaxScaler()
    hybrid_df[['cf_score', 'cb_score']] = scaler.fit_transform(hybrid_df[['cf_score', 'cb_score']])

    # Compute the weighted hybrid score
    hybrid_df['hybrid_score'] = (
        cf_weight * hybrid_df['cf_score'] + cb_weight * hybrid_df['cb_score']
    )

    # Sort by hybrid score and return top N titles
    top_recs = hybrid_df.sort_values('hybrid_score', ascending=False).head(n_recs)
    return top_recs['Title'].tolist()