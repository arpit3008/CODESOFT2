# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load Dataset
try:
    # Attempt to read the dataset with a specified encoding
    data = pd.read_csv(r"C:\Users\user\Downloads\archive\IMDb Movies India.csv", encoding='ISO-8859-1')
except UnicodeDecodeError as e:
    print(f"Encoding error: {e}")
    data = pd.read_csv(r"C:\Users\user\Downloads\archive\IMDb Movies India.csv", encoding='utf-8', errors='replace')

# Display the first few rows of the dataset
print("Initial data preview:")
print(data.head())

# Step 3: Data Preprocessing
# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Fill missing values for 'Rating' with the mean of the column
if 'Rating' in data.columns:
    data['Rating'].fillna(data['Rating'].mean(), inplace=True)

# Convert 'Duration' from string to numeric (extracting the numeric part)
data['Duration'] = data['Duration'].str.extract('(\d+)').astype(float)

# Check the data types and convert columns if necessary
print("\nData Types After Preprocessing:")
print(data.dtypes)

# Step 4: Exploratory Data Analysis (EDA)
# Plotting the distribution of ratings
plt.figure(figsize=(10, 5))
sns.histplot(data['Rating'], bins=20, kde=True)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Step 5: Feature Engineering
# Select relevant features for modeling
features = data[['Duration', 'Votes']]  # You can add more features as needed
target = data['Rating']

# Convert 'Votes' to numeric after removing non-numeric characters
features['Votes'] = pd.to_numeric(features['Votes'].str.replace(',', ''), errors='coerce')

# Step 6: Model Training and Evaluation
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Step 7: Prediction Example
example = np.array([[120, 1000]])  # Example: Duration = 120 mins, Votes = 1000
predicted_rating = model.predict(example)
print(f"\nPredicted Rating for the movie with Duration 120 mins and 1000 Votes: {predicted_rating[0]:.2f}")

