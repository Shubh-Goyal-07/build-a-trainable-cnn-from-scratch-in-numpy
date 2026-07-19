"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
def argmax_rows(matrix):
    return np.argmax(matrix, axis=1)

# Step 2 - row_max
import numpy as np

def row_max(matrix):
    return  np.expand_dims(np.max(matrix, axis=1), axis=1)

# Step 3 - row_sum
import numpy as np

def row_sum(matrix):
    """Return per-row sums of a 2D array with shape (N, 1)."""
    return np.expand_dims(np.sum(matrix, axis=1), axis=1)

# Step 4 - exp_shifted
import numpy as np

def exp_shifted(logits):
    """Subtract per-row max from logits and exponentiate elementwise."""
    max_per_row = row_max(logits)
    return np.exp(logits - max_per_row)

# Step 5 - stable_softmax
def stable_softmax(logits):
    exp_row = exp_shifted(logits)
    sum_row = row_sum(exp_row)
    
    return exp_row / sum_row

# Step 6 - one_hot
def one_hot(labels, num_classes):
    one_hot_enc = np.zeros((len(labels), num_classes))
    one_hot_enc[np.arange(len(labels)), labels] = 1.0
    return one_hot_enc

# Step 7 - gather_true_class_probs
def gather_true_class_probs(probs, labels):
    return probs[np.arange(len(labels)), labels]

# Step 8 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, labels, eps=1e-12):
    pred_probs = gather_true_class_probs(probs, labels)
    pred_probs = np.clip(pred_probs, eps, None)
    log_probs = np.log(pred_probs)
    
    loss = -np.mean(np.log(pred_probs))
    return float(abs(loss))

# Step 9 - accuracy
def accuracy(logits_or_probs, labels):
    pred_cls = argmax_rows(logits_or_probs)

    return np.mean(pred_cls==labels)

# Step 10 - he_std
def he_std(fan_in):
    return np.sqrt(2/fan_in)

# Step 11 - he_init
def he_init(shape, fan_in, seed):
    np.random.seed(seed)
    std = he_std(fan_in)
    return np.random.normal(loc=0, scale=std, size=shape)

# Step 12 - init_zero_bias
import numpy as np

def init_zero_bias(length):
    return np.zeros(length)

# Step 13 - pad_2d
def pad_2d(images, pad):
    N, C, H, W = images.shape
    padded_img = np.zeros((N, C, H+2*pad, W+2*pad), dtype=images.dtype)
    padded_img[:, :, pad:H+pad, pad:W+pad] = images
    
    return padded_img

# Step 14 - output_spatial_size
def output_spatial_size(input_size, kernel, stride, padding):
    return int((input_size+2*padding-kernel)/stride) + 1

# Step 15 - im2col
def im2col(images, kernel_h, kernel_w, stride, padding):
    N, C, H, W = images.shape
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)

    padded_img = pad_2d(images, padding)

    out = np.empty((N, out_h, out_w, C * kernel_h * kernel_w),
                dtype=padded_img.dtype)

    for i in range(out_h):
        for j in range(out_w):
            r = i * stride
            c = j * stride

            # Shape: (N, C, kernel_h, kernel_w)
            patch = padded_img[:, :, r:r+kernel_h, c:c+kernel_w]

            # Shape: (N, C*kernel_h*kernel_w)
            patch = patch.reshape(N, -1)

            # Rows corresponding to this spatial position
            out[:, i, j] = patch

    return out.reshape(N * out_h * out_w, C * kernel_h * kernel_w)

# Step 16 - col2im
def col2im(cols, input_shape, kernel_h, kernel_w, stride, padding):
    N, C, H, W = input_shape

    out_h = (H + 2 * padding - kernel_h) // stride + 1
    out_w = (W + 2 * padding - kernel_w) // stride + 1

    # Image with padding
    padded = np.zeros(
        (N, C, H + 2 * padding, W + 2 * padding),
        dtype=cols.dtype
    )

    row = 0
    for n in range(N):
        for i in range(out_h):
            for j in range(out_w):
                r = i * stride
                c = j * stride

                patch = cols[row].reshape(C, kernel_h, kernel_w)

                # Add because patches overlap
                padded[n, :, r:r+kernel_h, c:c+kernel_w] += patch

                row += 1

    # Remove padding
    if padding > 0:
        return padded[:, :, padding:-padding, padding:-padding]

    return padded

# Step 17 - conv2d_forward
def conv2d_forward(x, weights, bias, stride, padding):
    N, _, H, W = x.shape
    C_out, C_in, K_h, K_w = weights.shape

    out_h = output_spatial_size(H, K_h, stride, padding)
    out_w = output_spatial_size(W, K_w, stride, padding)

    cols = im2col(x, K_h, K_w, stride, padding)

    w_flat = weights.reshape(C_out, -1)

    # (N*out_h*out_w, C_out)
    Y = cols @ w_flat.T + bias

    # -> (N, out_h, out_w, C_out)
    Y = Y.reshape(N, out_h, out_w, C_out)

    # -> (N, C_out, out_h, out_w)
    Y = Y.transpose(0, 3, 1, 2)

    cache = {
        "x_shape": x.shape,
        "weights": weights,
        "cols": cols,
        "stride": stride,
        "padding": padding,
        "kernel_h": K_h,
        "kernel_w": K_w
    }

    return Y, cache

# Step 18 - conv2d_grad_input
def conv2d_grad_input(d_out, cache):
    d_out = d_out.transpose(0, 2, 3, 1)
    N, out_h, out_w, c_out = d_out.shape
    d_out = d_out.reshape(N*out_h*out_w, c_out)

    w_flat = cache["weights"].reshape(c_out, -1)
    d_cols = d_out @ w_flat

    return col2im(
        d_cols,
        cache["x_shape"],
        cache["kernel_h"],
        cache["kernel_w"],
        cache["stride"],
        cache["padding"]
    )

# Step 19 - conv2d_grad_weights
def conv2d_grad_weights(d_out, cache):
    N, c_out, out_h, out_w = d_out.shape
    d_out = d_out.transpose(1, 0, 2, 3)
    d_out = d_out.reshape(c_out, N*out_h*out_w)

    dW = (d_out @ cache["cols"]).reshape(cache["weights"].shape)
    
    return dW

# Step 20 - conv2d_grad_bias
def conv2d_grad_bias(d_out):
    return np.sum(d_out, axis=(0, 2, 3))

# Step 21 - conv2d_backward
def conv2d_backward(d_out, cache):
    dx = conv2d_grad_input(d_out, cache)
    dW = conv2d_grad_weights(d_out, cache)
    db = conv2d_grad_bias(d_out)

    return dx, dW, db

# Step 22 - maxpool2d_forward
def maxpool2d_forward(x, kernel, stride):
    N, C, H, W = x.shape
    out_h = output_spatial_size(H, kernel, stride, 0)
    out_w = output_spatial_size(W, kernel, stride, 0)

    out = np.empty((N, C, out_h, out_w))
    argmax = np.empty((N, C, out_h, out_w), dtype=np.int64)
    
    for i in range(out_h):
        for j in range(out_w):
            r = i * stride
            c = j * stride

            patch = x[:, :, r:r+kernel, c:c+kernel]

            out[:, :, i, j] = patch.max(axis=(2,3))

            flat = patch.reshape(N, C, -1)
            argmax[:, :, i, j] = flat.argmax(axis=2)

    cache = {
        "x_shape": x.shape,
        "argmax": argmax,
        "kernel": kernel,
        "stride": stride
    }

    return out, cache

# Step 23 - scatter_grad_window
import numpy as np

def scatter_grad_window(grad_value, argmax_index, kernel):
    out = np.zeros(kernel*kernel)
    out[argmax_index] = grad_value

    return out.reshape(kernel, kernel)

# Step 24 - maxpool2d_backward
def maxpool2d_backward(d_out, cache):
    x_shape = cache["x_shape"]
    argmax = cache["argmax"]
    kernel = cache["kernel"]
    stride = cache["stride"]

    N, C, H, W = x_shape
    _, _, out_h, out_w = d_out.shape

    dx = np.zeros(x_shape, dtype=d_out.dtype)

    for i in range(out_h):
        for j in range(out_w):
            r = i * stride
            c = j * stride

            for n in range(N):
                for ch in range(C):
                    window_grad = scatter_grad_window(
                        d_out[n, ch, i, j],
                        argmax[n, ch, i, j],
                        kernel,
                    )

                    dx[n, ch, r:r+kernel, c:c+kernel] += window_grad

    return dx

# Step 25 - relu_forward
def relu_forward(x):
    z = np.where(x>0, x, 0)
    return z, {"x": x}

# Step 26 - relu_backward
def relu_backward(d_out, cache):
    return d_out * (cache["x"]>0).astype(np.int8)

# Step 27 - flatten_forward
def flatten_forward(x):
    N, _, _, _ = x.shape
    return x.reshape(N, -1), {"x_shape": x.shape}

# Step 28 - flatten_backward
import numpy as np

def flatten_backward(d_out, cache):
    return d_out.reshape(cache["x_shape"])

# Step 29 - linear_forward
def linear_forward(x, weights, bias):
    out = x @ weights + bias
    cache = {
        "x": x,
        "weights": weights
    }

    return out, cache

# Step 30 - linear_grad_input
import numpy as np

def linear_grad_input(d_out, cache):
    """Gradient of a linear layer w.r.t. its input X."""
    return d_out @ cache["weights"].T

# Step 31 - linear_grad_weights
import numpy as np

def linear_grad_weights(x, dout):
    """Gradient of loss wrt linear-layer weights W of shape (D_in, D_out)."""
    return x.T @ dout

# Step 32 - linear_grad_bias
import numpy as np

def linear_grad_bias(dout):
    return np.sum(dout, axis=0)

# Step 33 - linear_backward
def linear_backward(dout, cache):
    dx = linear_grad_input(dout, cache)
    dW = linear_grad_weights(cache["x"], dout)
    db = linear_grad_bias(dout)

    return dx, dW, db

# Step 34 - softmax_cross_entropy_forward
def softmax_cross_entropy_forward(logits, y):
    probs = stable_softmax(logits)
    loss = cross_entropy_loss(probs, y)

    # Convert signed/tiny zero to positive zero
    if np.isclose(loss, 0.0):
        loss = 0.0

    return loss

# Step 35 - softmax_cross_entropy_backward
def softmax_cross_entropy_backward(logits, y):
    N, num_classes = logits.shape
    probs = stable_softmax(logits)
    one_hot_enc = one_hot(y, num_classes)

    return (probs - one_hot_enc) / N

# Step 36 - sgd_step
import numpy as np

def sgd_step(param, grad, lr):
    return param - grad*lr

# Step 37 - adam_update_m
import numpy as np

def adam_update_m(m, grad, beta_one):
    return beta_one*m + (1-beta_one)*grad

# Step 38 - adam_update_v
import numpy as np

def adam_update_v(v, grad, beta_two):
    return beta_two*v + (1-beta_two)*(grad**2)

# Step 39 - adam_bias_correct
def adam_bias_correct(moment, beta, t):
    return moment / (1 - beta**t)

# Step 40 - adam_param_step
import numpy as np

def adam_param_step(param, m_hat, v_hat, lr, eps):
    return param - lr*(m_hat / (np.sqrt(v_hat) + eps))

# Step 41 - adam_step
import numpy as np

def adam_step(param, grad, m, v, t, lr, beta_one, beta_two, eps):
    mt = adam_update_m(m, grad, beta_one)
    vt = adam_update_v(v, grad, beta_two)

    m_hat = adam_bias_correct(mt, beta_one, t)
    v_hat = adam_bias_correct(vt, beta_two, t)

    return adam_param_step(param, m_hat, v_hat, lr, eps), mt, vt

# Step 42 - init_conv_layer
def init_conv_layer(out_channels, in_channels, kernel_size, seed=0):
    params = {
        "W": he_init((out_channels, in_channels, kernel_size, kernel_size), in_channels*kernel_size*kernel_size, seed),
        "b": init_zero_bias(out_channels)
    }

    return params

# Step 43 - init_linear_layer
def init_linear_layer(in_features, out_features, seed=0):
    params = {
        "W": he_init((in_features, out_features), in_features, seed),
        "b": init_zero_bias((out_features))
    }

    return params

# Step 44 - init_lenet
def init_lenet(in_channels, num_classes, seed=0):
    lenet = {
        "conv1": init_conv_layer(6, in_channels, 5, seed),
        "conv2": init_conv_layer(16, 6, 5, seed+1),
        "fc1": init_linear_layer(256, 120, seed+2),
        "fc2": init_linear_layer(120, num_classes, seed+3)
    }

    return lenet

# Step 45 - forward_conv_block
def forward_conv_block(x, W, b, pool_size, stride, pad):
    out, conv_cache = conv2d_forward(x, W, b, stride, pad)
    out, relu_cache = relu_forward(out)
    out, pool_cache = maxpool2d_forward(out, pool_size, pool_size)

    cache = {
        "conv_cache": conv_cache,
        "relu_cache": relu_cache,
        "pool_cache": pool_cache
    }

    return out, cache

# Step 46 - forward_classifier_block
def forward_classifier_block(x, fc1, fc2):
    x_flat, flatten_cache = flatten_forward(x)
    x1, fc1_cache = linear_forward(x_flat, fc1["W"], fc1["b"])
    z1, relu_cache = relu_forward(x1)
    x2, fc2_cache = linear_forward(z1, fc2["W"], fc2["b"])

    cache = {
        "flatten_cache": flatten_cache,
        "fc1_cache": fc1_cache,
        "relu_cache": relu_cache,
        "fc2_cache": fc2_cache
    }

    return x2, cache

# Step 47 - lenet_forward
def lenet_forward(x, params):
    out, block1_cache = forward_conv_block(
        x,
        params["conv1"]["W"],
        params["conv1"]["b"],
        pool_size=2,
        stride=1,
        pad=0
    )

    out, block2_cache = forward_conv_block(
        out,
        params["conv2"]["W"],
        params["conv2"]["b"],
        pool_size=2,
        stride=1,
        pad=0
    )

    logits, classifier_cache = forward_classifier_block(
        out,
        params["fc1"],
        params["fc2"]
    )

    caches = {
        "block1": block1_cache,
        "block2": block2_cache,
        "classifier": classifier_cache
    }

    return logits, caches

# Step 48 - backward_conv_block
def backward_conv_block(dout, cache):
    dout = maxpool2d_backward(dout, cache["pool_cache"])
    dout = relu_backward(dout, cache["relu_cache"])
    dx, dW, db = conv2d_backward(dout, cache["conv_cache"])

    return dx, dW, db

# Step 49 - backward_classifier_block
def backward_classifier_block(dlogits, cache):
    # FC2 backward
    dout, dW2, db2 = linear_backward(
        dlogits,
        cache["fc2_cache"]
    )

    # ReLU backward
    dout = relu_backward(
        dout,
        cache["relu_cache"]
    )

    # FC1 backward
    dout, dW1, db1 = linear_backward(
        dout,
        cache["fc1_cache"]
    )

    # Flatten backward
    dx = flatten_backward(
        dout,
        cache["flatten_cache"]
    )

    return {
        "dx": dx,
        "fc1": {
            "dW": dW1,
            "db": db1
        },
        "fc2": {
            "dW": dW2,
            "db": db2
        }
    }

# Step 50 - lenet_backward
def lenet_backward(dlogits, caches):
    # Classifier block
    classifier_grads = backward_classifier_block(
        dlogits,
        caches["classifier"]
    )

    # Conv block 2
    dx, dW2, db2 = backward_conv_block(
        classifier_grads["dx"],
        caches["block2"]
    )

    # Conv block 1
    dx, dW1, db1 = backward_conv_block(
        dx,
        caches["block1"]
    )

    grads = {
        "conv1": {
            "dW": dW1,
            "db": db1
        },
        "conv2": {
            "dW": dW2,
            "db": db2
        }, 
        "fc1": classifier_grads["fc1"],
        "fc2": classifier_grads["fc2"]
    }

    return grads

# Step 51 - lenet_predict
def lenet_predict(x, params):
    logits, cache = lenet_forward(x, params)
    return np.argmax(logits, axis=1)

# Step 52 - build_synthetic_image_dataset
def build_synthetic_image_dataset(num_samples, num_classes, image_size, in_channels=1, seed=0):
    rng = np.random.default_rng(seed)

    # Labels first
    y = rng.integers(0, num_classes, size=num_samples)

    # Then images
    x = rng.standard_normal(
        (num_samples, in_channels, image_size, image_size)
    )

    # Class-dependent shift
    shift = y - (num_classes - 1) / 2

    # Broadcast over C, H, W
    x += shift[:, None, None, None]

    return x, y

# Step 53 - shuffle_indices
import numpy as np

def shuffle_indices(n, seed=0):
    rng = np.random.RandomState(seed)
    return rng.permutation(n)

# Step 54 - train_test_split
def train_test_split(x, y, test_fraction=0.2, seed=0):
    N = len(x)

    indices = shuffle_indices(N, seed)

    test_size = int(N * test_fraction)

    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    x_train = x[train_idx]
    y_train = y[train_idx]

    x_test = x[test_idx]
    y_test = y[test_idx]

    return x_train, y_train, x_test, y_test

# Step 55 - iterate_minibatches
def iterate_minibatches(x, y, batch_size, seed=0):
    N = len(x)

    indices = shuffle_indices(N, seed)

    for start in range(0, N - batch_size + 1, batch_size):
        batch_idx = indices[start:start + batch_size]

        xb = x[batch_idx]
        yb = y[batch_idx]

        yield xb, yb

# Step 56 - train_step (not yet solved)
# TODO: implement

# Step 57 - train_one_epoch (not yet solved)
# TODO: implement

# Step 58 - train_loop (not yet solved)
# TODO: implement

# Step 59 - evaluate (not yet solved)
# TODO: implement

