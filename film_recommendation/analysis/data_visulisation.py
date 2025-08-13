# Import the function to load data from your recommendation module
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scr.recommendation import load_data

# Import seaborn and matplotlib for data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets: users, movies, and ratings
users, movies, ratings = load_data()

# Merge ratings with user info on 'UserID' to get demographic data alongside ratings
ratings_users = ratings.merge(users, on='UserID')

# Create a count plot showing the distribution of ratings, separated by gender
sns.countplot(
    data=ratings_users,    # Data source
    x='Rating',            # Ratings on the x-axis
    hue='Gender',          # Separate bars by gender
    palette='pastel'       # Use a pastel color palette for aesthetics
)


plt.title('Rating Distribution by Gender')
plt.xlabel('Rating')
plt.ylabel('Count')

# Add a legend with the title 'Gender' to clarify the hue distinction
plt.legend(title='Gender')

# Display the plot
plt.show()
