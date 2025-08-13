from flask import Flask, render_template, request, session, redirect, url_for
import recommendation
import pickle

app = Flask(__name__)
app.secret_key = 'Daisy3'  # Secret key for session management

# Load precomputed model data for recommendations
with open('model_data.pkl', 'rb') as f:
    model_data = pickle.load(f)

users = model_data['users']
movies = model_data['movies']
ratings = model_data['ratings']
rating_matrix = model_data['ratings_matrix']
user_mean = model_data['user_mean']
user_similarity = model_data['user_similarity']


@app.route('/', methods=["GET", "POST"])
def home():
    """
    Handles the main page requests.

    GET:
        - Displays recommendations based on films the user has liked (stored in session).
        - Shows the liked films and top content-based recommendations.

    POST:
        - Receives user input of liked films as a comma-separated string.
        - Matches user input against movie titles via Jaccard similarity.
        - Updates session with matched liked films.
        - Generates hybrid recommendations combining collaborative and content-based filtering.
        - Renders page with updated recommendations and liked films.

    Also handles removal of liked films via URL query parameter.
    """
    # Initialize liked films in session if not present
    if 'liked_films' not in session:
        session['liked_films'] = []

    all_titles = movies['Title'].tolist()
    error_message = None  # To display errors when no matches found

    # Handle removal of a liked film via query parameter
    remove_title = request.args.get('remove')
    if remove_title:
        session['liked_films'] = [title for title in session['liked_films'] if title != remove_title]
        return redirect(url_for('home'))

    if request.method == 'POST':
        raw_likes = request.form.get('liked_movies', '')
        user_titles = [t.strip() for t in raw_likes.split(",") if t.strip()]

        # Match user input titles to movie titles using fuzzy Jaccard similarity
        matched_titles = recommendation.jaccard_similarity_titles(user_titles, all_titles)

        if not matched_titles:
            error_message = "No matching films found. Please check the titles and try again."
            return render_template('index.html', recommendations=[], liked_films=session.get('liked_films', []), error_message=error_message)

        # Append newly matched titles to session, avoiding duplicates
        liked_films = session.get('liked_films', [])
        for title in matched_titles:
            if title not in liked_films:
                liked_films.append(title)
        session['liked_films'] = liked_films

        # Generate hybrid recommendations using collaborative and content-based filtering
        recommendations = recommendation.hybrid_recommendation(
            liked_films, movies, ratings, rating_matrix, user_mean, user_similarity,
            k=40, n_recs=10, cf_weight=0.6, cb_weight=0.4
        )

        return render_template('index.html', recommendations=recommendations, liked_films=liked_films)

    # Handle GET request: show content-based recommendations for liked films
    liked_films = session.get('liked_films', [])
    if liked_films:
        rec_df = recommendation.recommend_content_based(liked_films, movies, n_recs=5)
        recommendations = rec_df['Title'].tolist() if not rec_df.empty else [] 
    else:
        recommendations = []

    return render_template('index.html', recommendations=recommendations, liked_films=liked_films)


if __name__ == '__main__':
    import os
    os.environ["FLASK_ENV"] = "development"
    app.run(debug=True, use_reloader=False)
