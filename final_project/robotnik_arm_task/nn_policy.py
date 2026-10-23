"""NumPy-only inference for the Lecture 6 style Linear/ReLU regression MLP."""
import numpy as np


def validate_nn(model):
    steps = np.asarray(model.get('arm_window_steps', 1))
    if steps.shape != () or steps.dtype.kind not in 'iu' or steps.item() < 1:
        raise ValueError('Invalid NN history length')
    inputs = 21 * int(steps.item())
    for key, shape in {'nn_x_mean': (inputs,), 'nn_x_scale': (inputs,),
                       'nn_y_mean': (7,), 'nn_y_scale': (7,)}.items():
        if model[key].shape != shape or not np.isfinite(model[key]).all():
            raise ValueError(f'Invalid NN parameter: {key}')
    if np.any(model['nn_x_scale'] <= 0) or np.any(model['nn_y_scale'] <= 0):
        raise ValueError('NN scales must be positive')
    width = inputs
    for i in range(3):
        w, b = model[f'nn_w{i}'], model[f'nn_b{i}']
        if (w.ndim != 2 or w.shape[0] != width or b.shape != (w.shape[1],)
                or not np.isfinite(w).all() or not np.isfinite(b).all()):
            raise ValueError(f'Invalid NN layer {i}')
        width = w.shape[1]
    if width != 7:
        raise ValueError('Expected seven arm outputs')


def predict_nn(model, X):
    X = np.asarray(X, dtype=np.float32)
    inputs = len(model['nn_x_mean'])
    if X.ndim not in (1, 2) or X.shape[-1] != inputs or not np.isfinite(X).all():
        raise ValueError(f'Expected finite input ({inputs},) or (N, {inputs})')
    h = (X - model['nn_x_mean']) / model['nn_x_scale']
    # CORE CHANGE: two Linear+ReLU hidden layers, linear regression output.
    for i in range(3):
        h = h @ model[f'nn_w{i}'] + model[f'nn_b{i}']
        if i < 2:
            h = np.maximum(h, 0)
    return h * model['nn_y_scale'] + model['nn_y_mean']
