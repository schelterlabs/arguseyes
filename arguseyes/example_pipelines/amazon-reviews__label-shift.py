import pandas as pd

from sklearn.preprocessing import OneHotEncoder, label_binarize, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline

target_categories = ['Digital_Video_Games']
split_date = '2015-07-31'

reviews = pd.read_csv('datasets/amazon-reviews/reviews.csv.gz', compression='gzip', index_col=0)
products = pd.read_csv('datasets/amazon-reviews/products.csv', index_col=0)
categories = pd.read_csv('datasets/amazon-reviews/categories.csv', index_col=0)
ratings = pd.read_csv('datasets/amazon-reviews/ratings.csv', index_col=0)

reviews = reviews[reviews.verified_purchase == 'Y']
reviews = reviews[reviews.marketplace == 'US']
reviews = reviews[reviews.review_date >= '2015-01-01']

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

train_data['is_helpful'] = train_data['helpful_votes'] > 0
test_data['is_helpful'] = test_data['helpful_votes'] > 0

# Simulate label shift by resampling the test data to have a label proportion of 50/50
test_data_helpful = test_data[test_data['is_helpful'] == True]
test_datano_helpful_sampled = test_data[test_data['is_helpful'] == False].sample(len(test_data_helpful))

test_data = pd.concat([test_data_helpful, test_datano_helpful_sampled])

train_labels = label_binarize(train_data['is_helpful'], classes=[True, False])
test_labels = label_binarize(test_data['is_helpful'], classes=[True, False])

numerical_attributes = ['star_rating']
categorical_attributes = ['vine', 'verified_purchase', 'category_id']

feature_transformation = ColumnTransformer(transformers=[
    ('numerical_features', StandardScaler(), numerical_attributes),
    ('categorical_features', OneHotEncoder(handle_unknown='ignore'), categorical_attributes),
    ('textual_features', HashingVectorizer(ngram_range=(1, 3), n_features=100), 'title_and_review_text')
])

pipeline = Pipeline([
    ('features', feature_transformation),
    ('learner', SGDClassifier(loss='log', penalty='l1', max_iter=1000))])

model = pipeline.fit(train_data, train_labels)

score = model.score(test_data, test_labels)

print(f'Accuracy on the test set: {score}')
