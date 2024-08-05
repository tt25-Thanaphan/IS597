import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from tqdm import tqdm
from tqdm.auto import tqdm
tqdm.pandas()
import time
from scipy.sparse import csr_matrix, hstack
from xgboost import XGBClassifier

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load your data
file_path = 'america_north.Canada.eng.1'  # Read the file
df = pd.read_csv(file_path, usecols=['Index', 'City', 'Text'])

print("Original dataset shape:", df.shape)

# Define regions
regions = {
    'Atlantic Canada': ['fredericton', 'saint john', 'moncton', 'halifax', 'charlottetown', 'new glasgow', 'corner brook', 'truro', 'glace bay'],
    'Quebec': ['montreal', 'quebec', 'sherbrooke', 'trois-rivieres', 'levis', 'saint-hyacinthe', 'saint-jean-sur-richelieu', 'saint-jerome', 'shawinigan', 'drummondville', 'granby', 'saint-laurent', 'salaberry-de-valleyfield', 'sorel-tracy', 'victoriaville', 'rouyn-noranda', 'sept-iles', 'alma', 'joliette', 'brossard'],
    'Ontario': ['toronto', 'ottawa', 'hamilton', 'london', 'kingston', 'windsor', 'oshawa', 'barrie', 'guelph', 'kitchener', 'waterloo', 'brantford', 'st. catharines', 'peterborough', 'sudbury', 'thunder bay', 'timmins', 'north bay', 'sault ste. marie', 'stratford', 'orillia', 'brockville', 'chatham-kent', 'sarnia', 'woodstock', 'owen sound', 'midland', 'cobourg', 'orangeville', 'cornwall', 'leamington', 'bradford west gwillimbury', 'grand sudbury'],
    'Prairie Provinces': ['calgary', 'edmonton', 'regina', 'saskatoon', 'winnipeg', 'lethbridge', 'red deer', 'medicine hat', 'grande prairie', 'moose jaw', 'prince albert', 'brandon', 'airdrie', 'spruce grove', 'lloydminster', 'north battleford', 'sherwood park'],
    'British Columbia': ['vancouver', 'victoria', 'kelowna', 'kamloops', 'nanaimo', 'prince george', 'abbotsford', 'chilliwack', 'vernon', 'penticton', 'campbell river', 'courtenay', 'port alberni', 'parksville', 'duncan', 'terrace', 'cranbrook', 'white rock', 'new westminster', 'north vancouver', 'richmond', 'north cowichan'],
    'Northern Territories': ['whitehorse'],
}

# Function to map cities to regions
def map_to_region(city):
    city = city.lower()
    for region, cities in regions.items():
        if city in cities:
            return region
    return 'Unknown'  # Changed from 'Others' to 'Unknown'

# Apply the mapping
print("\nMapping cities to regions...")
df['Region'] = df['City'].apply(map_to_region)

print("\nOriginal regional distribution:")
print(df['Region'].value_counts())

# After applying the mapping
print("\nCities not assigned to a region:")
unassigned_cities = df[df['Region'] == 'Unknown']['City'].unique()
print(unassigned_cities)

# Define the number of samples per region
samples_per_region = 200000  # Increased to 200000 for better performance

# Sample equally from each region
sampled_df = pd.DataFrame()
for region in df['Region'].unique():
    if region != 'Unknown':  # Exclude 'Unknown' region from sampling
        region_df = df[df['Region'] == region]
        if len(region_df) > samples_per_region:
            sampled_region = region_df.sample(n=samples_per_region, random_state=42)
        else:
            sampled_region = region_df  # If a region has fewer samples, take all of them
        sampled_df = pd.concat([sampled_df, sampled_region])

print("\nSampled dataset shape:", sampled_df.shape)
print("\nSampled regional distribution:")
print(sampled_df['Region'].value_counts())

# Replace the original dataframe with the sampled one
df = sampled_df

# Data cleaning function
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stopword_list = set(stopwords.words("english")) - {'north', 'south', 'east', 'west'}
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stopword_list]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)


# Apply cleaning to the Text column
print("\nCleaning text data...")
df['CleanedText'] = df['Text'].progress_apply(clean_text)

class CityNameExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, city_list):
        self.city_list = city_list

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = [[1 if city in text.lower() else 0 for city in self.city_list] for text in X]
        return csr_matrix(features)

# Prepare features (X) and target (y)
X = df['CleanedText']
y = df['Region']

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a list of all cities
all_cities = [city.lower() for cities in regions.values() for city in cities]

# Create the feature extraction pipeline
feature_extractor = FeatureUnion([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 3))),
    ('city_names', CityNameExtractor(all_cities)),
])

# Create the full pipeline
pipeline = Pipeline([
    ('features', feature_extractor),
    ('classifier', XGBClassifier(random_state=42))
])

# Hyperparameter grid
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5],
    'classifier__learning_rate': [0.1, 0.3]
}

# Perform grid search
print("\nPerforming grid search...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print best parameters
print("Best parameters:", grid_search.best_params_)

# Evaluate model
y_pred = grid_search.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Example of predicting new texts
new_texts = [
    "The CN Tower offers a spectacular view of Toronto.",
    "Vancouver's Stanley Park is a beautiful urban oasis.",
    "Montreal's old town is full of historic charm.",
    "The Calgary Stampede is a world-famous rodeo event.",
    "The small town charm of Fredericton is unforgettable.",
    "Whitehorse is the gateway to Yukon's wilderness.",
    "The prairies stretch as far as the eye can see in Saskatchewan."
]

# Clean new texts
print("\nCleaning new texts...")
cleaned_new_texts = [clean_text(text) for text in new_texts]

# Predict
probabilities = grid_search.predict_proba(cleaned_new_texts)

print("\nPredictions for new texts:")
for original_text, cleaned_text, probs in zip(new_texts, cleaned_new_texts, probabilities):
    print(f"Original Text: '{original_text}'")
    print(f"Cleaned Text: '{cleaned_text}'")
    for region, prob in sorted(zip(le.classes_, probs), key=lambda x: x[1], reverse=True):
        print(f"{region}: {prob * 100:.2f}%")
    print()