import tensorflow as tf
from tqdm import tqdm
from itertools import product
from models.model_builder import build_complex_model

def grid_search(param_grid, X_train, y_train, X_test, y_test, input_shape, output_size):
    best_accuracy = 0
    best_params = None
    best_model = None

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    for params in tqdm(product(*param_grid.values()), total=len(list(product(*param_grid.values()))), desc="Grid Search Progress"):
        with strategy.scope():
            model = build_complex_model(input_shape, output_size, *params)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=0)

        _, accuracy = model.evaluate(X_test, y_test, verbose=0)

        tqdm.write(f'Model Accuracy for {params}: {accuracy}')

        if accuracy > best_accuracy:
            best_accuracy = max(accuracy, best_accuracy)
            best_params = params
            best_model = model
    
    model.save('./movie_review_model.h5')

    return best_accuracy, best_params