import numpy as np
import utils
import typing
from tqdm import tqdm
np.random.seed(1)

def find_mean_and_deviation(X: np.ndarray):
    """
    Args:
        X: images of shape [training set size, 784] in the range (0, 255)
    Returns:
        mean: mean of the training set
        standard_deviation: standard deviation of the training set
    """
    assert X.shape[1] == 784, \
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # Calculates the mean and standard deviation from the whole training set
    mean = np.sum(X) / (X.shape[0] * X.shape[1])
    variance = np.sum((X - mean) ** 2) / (X.shape[0] * X.shape[1])
    standard_deviation = np.sqrt(variance)

    return mean, standard_deviation

def pre_process_images(X: np.ndarray, mean, standard_deviation):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
        mean: mean of the training set
        standard_deviation: standard deviation of the training set
    Returns:
        X: images of shape [batch size, 785]
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # Normalizes the input X
    X_norm = (X - mean) / standard_deviation

    # Applies the bias trick by adding a column of 1 in the front
    bias_trick = np.ones((np.shape(X)[0], 1))
    X_norm = np.hstack((bias_trick, X_norm))

    return X_norm


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    ce = targets * np.log(outputs)
    return np.sum(ce) / (-1 * ce.shape[0])
    print("test for github")
    raise NotImplementedError


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Define number of input nodes
        self.I = 785 # 28x28=784 pixels + 1 for bias
        self.use_improved_sigmoid = use_improved_sigmoid

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Values computed en forward function and needed in backward function
        self.aj = None
        self.zj = None

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            w = np.zeros(w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # Sigmoid function for the hidden layer
        #hidden_layer_activation = np.zeros((self.ws[0].shape[0], self.ws[0].shape[1]))
        self.zj = X @ self.ws[0]
        self.aj = 1 / (1 + np.exp(-self.zj)) # Hidden layer activation

        # Softmax function for the output layer
        y = np.exp(self.aj @ self.ws[1])
        # Normalize each y^n by the sum of the vector's values
        sum = np.sum(y, axis=1)
        output_layer_activation = y/sum[:, None]

        return output_layer_activation
        return None

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        # self.grads = []

        # Output layer backpropagation
        delta_k = (targets - outputs) / (-X.shape[0])
        self.grads[1] = self.aj.T @ delta_k

        # Hidden layer backpropagation
        exp = np.exp(-self.zj)
        sigmoid_derivative = exp / ((1 + exp) ** 2)
        #print("delta_k.shape = ", delta_k.shape, "self.aj.shape = ", self.aj.shape, "self.zj.shape = ", self.zj.shape, " sigmoid_derivative.shape = ", sigmoid_derivative.shape, "ws.[1].shape = ", self.ws[1].shape)

        delta_j = (delta_k @ self.ws[1].T) * sigmoid_derivative
        #print("delta_j.shape = ", delta_j.shape, " X.shape = ", X.shape)
        self.grads[0] = X.T @ delta_j

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    aux_array = np.zeros((np.shape(Y)[0],num_classes - 1), dtype=int)
    Y = np.hstack((Y, aux_array))
    # For each row it sets array[row][0] to 0 and then writes 1 on the previous array[row][0] value
    for i in range(np.shape(Y)[0]):
        value = Y[i,0]
        Y[i,0] = 0
        Y[i,value] = 1
    return Y
    raise NotImplementedError


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited.
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in tqdm(range(w.shape[0])):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist(0.1)
    mean, standard_deviation = find_mean_and_deviation(X_train)
    #print("mean: ", mean, " / standard deviation: ", standard_deviation)
    #exit()
    X_train = pre_process_images(X_train, mean, standard_deviation)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)
    """logits = model.forward(X_train)
    np.testing.assert_almost_equal(
        logits.mean(), 1/10,
        err_msg="Since the weights are all 0's, the softmax activation should be 1/10")"""

    # Gradient approximation check for 100 images
    print("X_train.shape = ", X_train.shape)
    X_train = X_train[:100]
    print("X_train.shape = ", X_train.shape)
    Y_train = Y_train[:100]

    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    print("Llamando a grandient_approximation_test...")
    gradient_approximation_test(model, X_train, Y_train)
    print("gradient approximation test ya se ha ejecutado, todo bien :)")