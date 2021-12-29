import pandas as pd
import numpy as np
import sys

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline

target_categories = ['Digital_Video_Games']
split_date = '2015-07-31'
start_date = '2015-01-01'

if len(sys.argv) > 1:
    target_categories = [sys.argv[1]]

if len(sys.argv) > 2:
    split_date = sys.argv[2]

if len(sys.argv) > 3:
    start_date = sys.argv[3]

reviews = pd.read_csv('datasets/amazon-reviews/reviews.csv.gz', compression='gzip', index_col=0)
products = pd.read_csv('datasets/amazon-reviews/products.csv', index_col=0)
categories = pd.read_csv('datasets/amazon-reviews/categories.csv', index_col=0)
ratings = pd.read_csv('datasets/amazon-reviews/ratings.csv', index_col=0)

reviews = reviews[reviews.verified_purchase == 'Y']
reviews = reviews[reviews.marketplace == 'US']
reviews = reviews[reviews.review_date >= start_date]

reviews_with_ratings = reviews.merge(ratings, on='review_id')

categories_of_interest = categories[categories.category.isin(target_categories)]
products_of_interest = products.merge(left_on='category_id', right_on='id', right=categories_of_interest)

reviews_with_products_and_ratings = reviews_with_ratings.merge(products_of_interest, on='product_id')

reviews_with_products_and_ratings['product_title'] = \
    reviews_with_products_and_ratings['product_title'].fillna(value='')

reviews_with_products_and_ratings['review_headline'] = \
    reviews_with_products_and_ratings['review_headline'].fillna(value='')

reviews_with_products_and_ratings['review_body'] = \
    reviews_with_products_and_ratings['review_body'].fillna(value='')

reviews_with_products_and_ratings['title_and_review_text'] = \
    reviews_with_products_and_ratings.product_title + ' ' + \
    reviews_with_products_and_ratings.review_headline + ' ' + \
    reviews_with_products_and_ratings.review_body

train_data = reviews_with_products_and_ratings[reviews_with_products_and_ratings.review_date <= split_date]
test_data = reviews_with_products_and_ratings[reviews_with_products_and_ratings.review_date > split_date]
train_targets = np.array(train_data['star_rating'])
test_targets = np.array(test_data['star_rating'])

numerical_attributes = ['helpful_votes']
categorical_attributes = ['vine', 'verified_purchase', 'category_id']

feature_transformation = ColumnTransformer(transformers=[
    ('numerical_features', StandardScaler(), numerical_attributes),
    ('categorical_features', OneHotEncoder(handle_unknown='ignore'), categorical_attributes),
    ('textual_features', HashingVectorizer(ngram_range=(1, 3), n_features=100), 'title_and_review_text')
])

pipeline = Pipeline([
    ('features', feature_transformation),
    ('learner', SGDClassifier(loss='squared_loss', penalty='l1', max_iter=1000))])

model = pipeline.fit(train_data, train_targets)
score = model.score(test_data, test_targets)

print(f'MSE on the test set: {score}')
