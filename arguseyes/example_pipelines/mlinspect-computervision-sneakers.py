import numpy as np
import pandas as pd

from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.models import Sequential

from sklearn.preprocessing import FunctionTransformer, label_binarize
from sklearn.pipeline import Pipeline

pd.options.mode.chained_assignment = None


def decode_image(img_str):
    return np.array([int(val) for val in img_str.split(':')])


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


data_folder = 'datasets/sneakers'

train_data = pd.read_csv(f'{data_folder}/product_images_train_with_labelerrors.csv', converters={'image': decode_image})

# FIX label errors: Rerun with a cleaned version of the data
#train_data = pd.read_csv(f'{data_location}/product_images_train_clean.csv',
#                         converters={'image': decode_image})


test_data = pd.read_csv(f'{data_folder}/product_images_test.csv', converters={'image': decode_image})

product_categories = pd.read_csv(f'{data_folder}/product_categories.csv')
train_data_with_categories = train_data.merge(product_categories, on='category_id')
test_data_with_categories = test_data.merge(product_categories, on='category_id')

categories_to_distinguish = ['Sneaker', 'Ankle boot']

train_images = train_data_with_categories[train_data_with_categories['category_name']\
    .isin(categories_to_distinguish)]

test_images = test_data_with_categories[test_data_with_categories['category_name']\
    .isin(categories_to_distinguish)]


y_train = label_binarize(train_images['category_name'], classes=categories_to_distinguish)
y_test = label_binarize(test_images['category_name'], classes=categories_to_distinguish)

pipeline = Pipeline(steps=[
    ('normalisation', FunctionTransformer(normalise_image)),
    ('reshaping', FunctionTransformer(reshape_images)),
    ('model', KerasClassifier(create_cnn, epochs=10))
])

model = pipeline.fit(train_images[['image']], y_train)

model.score(test_images[['image']], y_test)
