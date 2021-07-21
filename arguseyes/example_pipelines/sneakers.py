import numpy as np
import pandas as pd
import sys

from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, label_binarize
from sklearn.pipeline import Pipeline


def decode_image(img_str):
    return np.array([int(val) for val in img_str.split(':')])

# TODO change this to pyarrow + parquet, which can handle numpy arrays well
train_data = pd.read_csv(f'datasets/sneakers/product_images.csv', converters={'image': decode_image})

product_categories = pd.read_csv('datasets/sneakers/product_categories.csv')
with_categories = train_data.merge(product_categories, on='category_id')

categories_to_distinguish = ['Sneaker', 'Ankle boot']

images_of_interest = with_categories[with_categories['category_name'].isin(categories_to_distinguish)]


def normalise_image(images):
    return images / 255.0


def reshape_images(images):
    return np.concatenate(images['image'].values) \
        .reshape(images.shape[0], 28, 28, 1)


def create_cnn():
    model = Sequential([
        Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=2),
        Dropout(0.3),
        Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
        MaxPooling2D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


pipeline = Pipeline(steps=[
    ('normalisation', FunctionTransformer(normalise_image)),
    ('reshaping', FunctionTransformer(reshape_images)),
    ('model', KerasClassifier(create_cnn))
])

random_seed_for_splitting = 1337
if len(sys.argv) > 1:
    random_seed_for_splitting = int(sys.argv[1])

train, test = train_test_split(images_of_interest, test_size=0.2, random_state=random_seed_for_splitting)

y_train = label_binarize(train['category_name'], classes=categories_to_distinguish)
y_test = label_binarize(test['category_name'], classes=categories_to_distinguish)

model = pipeline.fit(train[['image']], y_train)

model.score(test[['image']], y_test)
