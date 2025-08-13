# 🎬 Film Recommendation System

This project is a hybrid film recommendation system that combines **collaborative filtering** and **content-based filtering** to suggest films a user may enjoy based on their previously liked titles.

## 💡 Features

- Matches user-input movie titles using nlp and **Jaccard similarity**.
- Extracts release years from movie titles.
- Uses **collaborative filtering** based on user-user similarity, genres, and year of release.
- Uses **content-based filtering** focusing on genre similarity.
- Combines both approaches in a **hybrid model** with user-controlled weighting
- Handles noisy inputs and missing values gracefully.

## 🛠️ Methods Used

### 🔍 1. Jaccard Similarity Matching
Matches fuzzy or partial user inputs to real movie titles, so users don’t need to research exact release years or spellings.

### 🧠 2. Collaborative Filtering (CF)
Based on:
- User-user similarity based on rating patterns
- Ratings matrix analysis
- Preferences of similar users
- Genre similarity via cosine similarity and scaling
- Gaussian similarity function to calculate year-based distance between films
- User-controlled weighting for combining user-user similarity, genre, and year factors

### 🎨 3. Content-Based Filtering (CBF)
Based on:
- Genre overlap using one-hot encoded genre features
- Cosine similarity between user-liked films and candidate movies
- A simpler and faster alternative to CF and hybrid models

### 🔗 4. Hybrid Recommendation
Combines CF and CBF scores using a weighted approach adjustable by the user.

## 🗃️ Inputs

- `liked_films`: A list of movie titles the user likes.
- `movies`: A DataFrame with:
  - `Title` column (with format like *"Toy Story (1995)"*)
  - One-hot encoded genre columns
- `ratings`: User ratings DataFrame
- `ratings_matrix`: Pivot table of users × movies
- `user_similarity`: Precomputed user similarity matrix
- `user_mean`: Series of average ratings per user

## 🧪 Sample Functions

- `recommend_from_title(...)` - collaborative filtering
- `recommend_content_based(...)` - content-based filtering
- `hybrid_recommendation(...)` - combines CF + CBF
- `jaccard_similarity_titles(...)` - fuzzy matching for titles
- `extract_year(title)` - parses release year from titles

## 📊 Data Analysis (annotations folder)

This folder contains a quick exploratory analysis script for the dataset.

### Description:
- Loads user, movie, and rating data using the load_data function from the recommendation module.
- Merges user demographic info with ratings.
- Creates a count plot showing the distribution of ratings by gender using Seaborn and Matplotlib.
- Provides a simple visualization to understand how different genders rate movies.

### How to Run:
1. **Install required dependencies** (if you haven’t already):

    ```bash
    pip install seaborn matplotlib pandas
    ```

2. **Run the script from the project root directory:**

    ```bash
    python annotations/data_analysis.py
    ```

## 🌐 Web Interface

This project includes a simple Flask web app (`scr/app.py`) with an HTML front-end (`templates/index.html`).

### Features:
- A form to enter liked films (comma-separated)
- Dynamically displays a list of recommended films based on the selected recommendation models
- Includes an option to remove previously liked films — useful for correcting fuzzy matches or updating preferences
- Handles errors gracefully, such as unmatched titles or empty input, with clear user feedback when a film isn’t found in the dataset
- Features clean, minimal styling using built-in CSS within the HTML template

The app uses Flask’s Jinja2 templating engine to render results dynamically based on user input and recommendation logic.

Once running, you can access the app at:
http://127.0.0.1:5000/
(Or the localhost address shown in your terminal when the app launches.)

## 📁 Data Source

This project uses the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/) from GroupLens for training and evaluation.

> Note: Due to licensing restrictions, the dataset is not included in this repository.  
> You can download it directly from [MovieLens](https://grouplens.org/datasets/movielens/1m/) and place the extracted files into the appropriate folder (e.g. `data/`).

The dataset is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## 📦 Dependencies

Make sure you have the following Python packages installed (nltk==3.5 is used for compatibility with the punkt tokenizer.):

```bash
pip install pandas numpy scikit-learn nltk==3.5
```
### About `build_and_save` 
The file `scr/build_and_save.py` builds and saves the core data structures required by the recommendation system, including:

- User, movie, and rating datasets
- User-item rating matrix
- Normalized ratings and user mean ratings
- User similarity matrix

This script serializes these components into `model_data.pkl` for efficient loading during app runtime.

### When to run

- Run this script once when setting up the project or when the data is updated.
- If `model_data.pkl` is already present in the correct folder, you can skip this step.

### How to run

```bash
python scr/build_and_save.py
```

## How to run

1.**Run the pre-required scripts in order to dowload features required for nlp**

```bash
    python scr/download.py
```

2.**Run the app**

```bash
    python scr/app.py
```