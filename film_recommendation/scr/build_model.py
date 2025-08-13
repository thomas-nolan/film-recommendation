import pickle
import recommendation  # your recommendation code file

def build_and_save():
    """
    Builds and serializes the recommendation model components for later use.

    This function:
    - Loads user, movie, and rating data
    - Constructs a user-item matrix
    - Normalizes user ratings by centering them
    - Computes user-user similarity using Pearson correlation
    - Saves all relevant model artifacts to a pickle file (model_data.pkl)
    """

    print("Starting to load data...")
    users, movies, ratings = recommendation.load_data()
    print("Data loaded.")

    print("Creating user-item matrix...")
    ratings_matrix = recommendation.create_user_item_matrix(ratings)
    print("User-item matrix created.")

    print("Normalising ratings...")
    rating_norm, user_mean = recommendation.normalise_ratings(ratings_matrix)
    print("Ratings normalised.")

    print("Computing user similarity...")
    user_similarity = recommendation.compute_user_similarity(rating_norm)
    print("User similarity computed.")

    # Serialize all key components for reuse without recomputing
    print("Saving model...")
    with open('model_data.pkl', 'wb') as f:
        pickle.dump({
            'users': users,
            'movies': movies,
            'ratings': ratings,
            'ratings_matrix': ratings_matrix,
            'user_mean': user_mean,
            'user_similarity': user_similarity
        }, f)
    print("Model saved.")


if __name__ == '__main__':
    # Run the model-building process only if this script is executed directly
    build_and_save()