import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from tqdm import tqdm
from tqdm.auto import tqdm
tqdm.pandas()
import time
import spacy
from scipy.sparse import csr_matrix, hstack

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load spaCy model for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")

# Load your data
file_path = 'america_north.Canada.eng.1'  # Read the file.
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
samples_per_region = 200000  # Increased to 200,000 for better performance

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

# Data cleaning function with progress reporting
def clean_text_with_progress(texts):
    is_series = isinstance(texts, pd.Series)
    if is_series:
        total = len(texts)
        cleaned_texts = texts.copy()
    else:
        total = len(texts)
        cleaned_texts = texts.copy()

    for step in range(1, 7):
        print(f"\nStep {step}/6: ", end="")
        if step == 1:
            print("Converting to lowercase...")
            if is_series:
                cleaned_texts = cleaned_texts.progress_apply(lambda x: x.lower())
            else:
                cleaned_texts = [x.lower() for x in tqdm(cleaned_texts, total=total, ncols=70)]
        elif step == 2:
            print("Removing punctuation...")
            if is_series:
                cleaned_texts = cleaned_texts.progress_apply(
                    lambda x: x.translate(str.maketrans('', '', string.punctuation)))
            else:
                cleaned_texts = [x.translate(str.maketrans('', '', string.punctuation)) for x in
                                 tqdm(cleaned_texts, total=total, ncols=70)]
        elif step == 3:
            print("Tokenizing...")
            if is_series:
                cleaned_texts = cleaned_texts.progress_apply(nltk.word_tokenize)
            else:
                cleaned_texts = [nltk.word_tokenize(x) for x in tqdm(cleaned_texts, total=total, ncols=70)]
        elif step == 4:
            print("Removing stopwords...")
            stopword_list = set(stopwords.words("english"))
            if is_series:
                cleaned_texts = cleaned_texts.progress_apply(
                    lambda x: [word for word in x if word not in stopword_list])
            else:
                cleaned_texts = [[word for word in x if word not in stopword_list] for x in
                                 tqdm(cleaned_texts, total=total, ncols=70)]
        elif step == 5:
            print("Stemming...")
            stemmer = PorterStemmer()
            if is_series:
                cleaned_texts = cleaned_texts.progress_apply(lambda x: [stemmer.stem(word) for word in x])
            else:
                cleaned_texts = [[stemmer.stem(word) for word in x] for x in tqdm(cleaned_texts, total=total, ncols=70)]
        else:
            print("Joining tokens...")
            if is_series:
                cleaned_texts = cleaned_texts.progress_apply(lambda x: " ".join(x))
            else:
                cleaned_texts = [" ".join(x) for x in tqdm(cleaned_texts, total=total, ncols=70)]

        time.sleep(0.5)  # Add a small delay between steps

    return cleaned_texts

# Apply cleaning to the Text column
print("\nCleaning text data...")
df['CleanedText'] = clean_text_with_progress(df['Text'])

# Feature extraction classes
class CityNameExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, city_list):
        self.city_list = city_list

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = [[1 if city in text.lower() else 0 for city in self.city_list] for text in X]
        return csr_matrix(features)

class NamedEntityExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit(self, X, y=None):
        loc_features = self._extract_locations(X)
        self.vectorizer.fit(loc_features)
        return self

    def transform(self, X):
        loc_features = self._extract_locations(X)
        return self.vectorizer.transform(loc_features)

    def _extract_locations(self, X):
        loc_features = []
        for text in tqdm(X, desc="Extracting Named Entities"):
            doc = nlp(text)
            locations = [ent.text.lower() for ent in doc.ents if ent.label_ == 'GPE']
            loc_features.append(' '.join(locations))
        return loc_features

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
    ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
    ('city_names', CityNameExtractor(all_cities)),
    ('city_names_2', CityNameExtractor(all_cities)),
    ('city_names_3', CityNameExtractor(all_cities)),  # Added third instance to increase weight
    ('named_entities', NamedEntityExtractor())
])

# Create the full pipeline
pipeline = Pipeline([
    ('features', feature_extractor),
    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))  # Changed to GradientBoostingClassifier
])

# Train model
print("\nTraining model...")
pipeline.fit(X_train, y_train)

# Evaluate model
y_pred = pipeline.predict(X_test)
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
cleaned_new_texts = clean_text_with_progress(new_texts)

# Predict
probabilities = pipeline.predict_proba(cleaned_new_texts)

print("\nPredictions for new texts:")
for original_text, cleaned_text, probs in zip(new_texts, cleaned_new_texts, probabilities):
    print(f"Original Text: '{original_text}'")
    print(f"Cleaned Text: '{cleaned_text}'")
    for region, prob in sorted(zip(le.classes_, probs), key=lambda x: x[1], reverse=True):
        print(f"{region}: {prob * 100:.2f}%")
    print()