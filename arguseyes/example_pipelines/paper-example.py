import sys
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, label_binarize, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline

from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

print("ARGS", sys.argv)


def load_data(target_categories, start_date, verified_only):
    reviews = pd.read_csv('datasets/amazon-reviews/reviews.csv.gz', compression='gzip', index_col=0)
    products = pd.read_csv('datasets/amazon-reviews/products.csv', index_col=0)
    categories = pd.read_csv('datasets/amazon-reviews/categories.csv', index_col=0)
    ratings = pd.read_csv('datasets/amazon-reviews/ratings.csv', index_col=0)

    # Filter review data and categories
    reviews = reviews[reviews['review_date'] >= start_date]
    categories = categories[categories['category'].isin(target_categories)]
    if verified_only:
        reviews = reviews[reviews['verified_purchase'] == 'Y']

    # Join inputs
    reviews_with_ratings = reviews.merge(ratings, on='review_id')
    products = products.merge(categories, left_on='category_id', right_on='id')
    full_reviews = reviews_with_ratings.merge(products, on='product_id')

    # Create combined text column with title and review text
    full_reviews['product_title'] = \
        full_reviews['product_title'].fillna(value='')
    full_reviews['review_body'] = \
        full_reviews['review_body'].fillna(value='')

    full_reviews['title_and_review'] = full_reviews['product_title'].fillna(value='') \
                                       + ' ' + full_reviews['review_body'].fillna(value='')

    return full_reviews


def temporal_split(full_reviews, split_date):
    train_data = full_reviews[full_reviews['review_date'] <= split_date]
    test_data = full_reviews[full_reviews['review_date'] > split_date]

    train_data['is_helpful'] = train_data['helpful_votes'] > 0
    test_data['is_helpful'] = test_data['helpful_votes'] > 0

    train_labels = label_binarize(train_data['is_helpful'], classes=[True, False])
    test_labels = label_binarize(test_data['is_helpful'], classes=[True, False])

    return train_data, test_data, train_labels, test_labels


def safe_log(x):
    return np.log(x, out=np.zeros_like(x), where=(x != 0))


# Nested estimator/transformer pipeline for feature transformation
def feature_encoding(numerical, categorical, text):
    # Impute and one-hot-encode categorical features
    one_hot = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(handle_unknown='ignore'))])

    # Impute and scale numerical features
    scale = Pipeline([
        ('impute', SimpleImputer(strategy='mean')),
        ('log_transform', FunctionTransformer(lambda x: safe_log(x))),
        ('scale', StandardScaler())])

    # Hash n-grams of textual features
    hashing = HashingVectorizer(ngram_range=(1, 3), n_features=100)

    return ColumnTransformer(transformers=[
        ('numerical_features', scale, numerical),
        ('categorical_features', one_hot, categorical),
        ('textual_features', hashing, text)])


# Define layout of neural network classifier
def create_mlp():
    nn = Sequential([
        Dense(256, activation='relu'), Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')])
    nn.compile(loss='sparse_categorical_crossentropy',
               optimizer='adam', metrics='accuracy')
    return nn


target_categories = ['Digital_Video_Games']
split_date = '2015-07-31'

if len(sys.argv) > 1:
    target_categories = [sys.argv[1]]

if len(sys.argv) > 2:
    split_date = sys.argv[2]

reviews = load_data(target_categories, '2015-01-01', verified_only=True)
train_data, test_data, train_labels, test_labels = temporal_split(reviews, split_date)

feature_transformation = feature_encoding(
    numerical=['star_rating'],
    categorical=['vine', 'verified_purchase', 'category_id'],
    text='title_and_review')

pipeline = Pipeline([
    ('features', feature_transformation),
    ('learner', KerasClassifier(create_mlp))])

model = pipeline.fit(train_data, train_labels)
model.score(test_data, test_labels)

