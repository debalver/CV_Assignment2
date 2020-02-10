import numpy as np
import utils
import matplotlib.pyplot as plt
import typing
import time
from tqdm import tqdm
from task2a import cross_entropy_loss, SoftmaxModel, one_hot_encode, pre_process_images, find_mean_and_deviation
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray,
                       model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # Compute the outputs
    outputs = model.forward(X)
    # Counting the correct predictions
    nb_predictions = outputs.shape[0]
    nb_correct_predictions = 0
    for row, output in enumerate(outputs):
        index = np.argmax(output)
        if targets[row][index] == 1:
            nb_correct_predictions += 1

    accuracy = nb_correct_predictions / nb_predictions
    return accuracy


def train(
        model: SoftmaxModel,
        datasets: typing.List[np.ndarray],
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        # Task 3 hyperparameters,
        use_shuffle: bool,
        use_momentum: bool,
        momentum_gamma: float):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = datasets

    # Utility variables
    num_batches_per_epoch = X_train.shape[0] // batch_size
    num_steps_per_val = num_batches_per_epoch // 5
    # Tracking variables to track loss / accuracy
    train_loss = {}
    val_loss = {}
    train_accuracy = {}
    val_accuracy = {}
    # Early stop variables
    early_stopped_weight_j = np.zeros((model.ws[0].shape[0], model.ws[0].shape[1]))
    early_stopped_weight_k = np.zeros((model.ws[1].shape[0], model.ws[1].shape[1]))
    early_stop_counter = 0
    best_loss = float("inf")

    global_step = 0
    for epoch in tqdm(range(num_epochs)):
        for step in range(num_batches_per_epoch):
            start = step * batch_size
            end = start + batch_size
            X_batch, Y_batch = X_train[start:end], Y_train[start:end]

            outputs = model.forward(X_batch)
            model.backward(X_batch, outputs, Y_batch)
            # Update the weights
            model.ws[0] = model.ws[0] - learning_rate * model.grads[0]
            model.ws[1] = model.ws[1] - learning_rate * model.grads[1]

            # Track training loss continuously over the entire X_Train and not only the current batch
            #outputs_training = model.forward(X_train)
            #_train_loss = cross_entropy_loss(Y_batch, outputs)
            #train_loss[global_step] = _train_loss

            # Track train / validation loss / accuracy
            # every time we progress 20% through the dataset
            if (global_step % num_steps_per_val) == 0:
                # Test the validation data on the network
                outputs_validation = model.forward(X_val)
                _val_loss = cross_entropy_loss(Y_val, outputs_validation)
                val_loss[global_step] = _val_loss

                # Track training loss over the entire X_Train and not only the current batch
                # once every validation epoch
                outputs_training = model.forward(X_train)
                _train_loss = cross_entropy_loss(Y_train, outputs_training)
                train_loss[global_step] = _train_loss

                # Early stop implementation

                # If the loss does not reduce compared to best loss, increment counter
                # Otherwise, set the counter to 0 and update best loss
                if _val_loss >= best_loss:
                    early_stop_counter += 1
                else:
                    early_stop_counter = 0
                    best_loss = _val_loss
                    early_stopped_weight_j = model.ws[0]
                    early_stopped_weight_k = model.ws[1]
                # If 30 times in a row a new best loss was not achieved, stop the program
                if early_stop_counter == 30:
                    print("The cross entropy loss for validation data increased too much, thus triggering "
                          "the early stop at step : " + str(global_step) + " and epoch : " + str(epoch))
                    model.ws[0] = early_stopped_weight_j
                    model.ws[1] = early_stopped_weight_k
                    return model, train_loss, val_loss, train_accuracy, val_accuracy

                train_accuracy[global_step] = calculate_accuracy(
                    X_train, Y_train, model)
                val_accuracy[global_step] = calculate_accuracy(
                    X_val, Y_val, model)

            global_step += 1
        # Task 3a: Suffle training examples after each epoch
        np.random.shuffle(X_train)
    return model, train_loss, val_loss, train_accuracy, val_accuracy


if __name__ == "__main__":

    # Measure execution time
    start = time.time()

    # Load dataset
    validation_percentage = 0.2
    X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_full_mnist(
        validation_percentage)

    # Preprocess and adapt the data
    mean, standard_deviation = find_mean_and_deviation(X_train)
    X_train = pre_process_images(X_train, mean, standard_deviation)
    X_val = pre_process_images(X_val, mean, standard_deviation)
    X_test = pre_process_images(X_test, mean, standard_deviation)

    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)
    Y_test = one_hot_encode(Y_test, 10)

    # Hyperparameters
    num_epochs = 20
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter

    # Settings for task 3. Keep all to false for task 2.
    use_shuffle = False
    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)

    # Initializing weights before any training
    model.ws[0] = np.random.uniform(-1, 1, (785, 64))
    model.ws[1] = np.random.uniform(-1, 1, (64, 10))

    model, train_loss, val_loss, train_accuracy, val_accuracy = train(
        model,
        [X_train, Y_train, X_val, Y_val, X_test, Y_test],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_shuffle=use_shuffle,
        use_momentum=use_momentum,
        momentum_gamma=momentum_gamma)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Test Cross Entropy Loss:",
          cross_entropy_loss(Y_test, model.forward(X_test)))

    print("Final Train accuracy:",
          calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:",
          calculate_accuracy(X_val, Y_val, model))
    print("Final Test accuracy:",
          calculate_accuracy(X_test, Y_test, model))

    # Execution time calculation
    end = time.time()
    time_in_seconds = end - start
    if (time_in_seconds > 60):
        print("The process took: " + str(int(time_in_seconds / 60)) + "min " + str(int(time_in_seconds % 60)) + "s")
    else:
        print("The process took: " + str(int(time_in_seconds)) + "s")


    # Plot loss
    plt.ylim([0.3, 2.0])
    utils.plot_loss(train_loss, "Training Loss")
    utils.plot_loss(val_loss, "Validation Loss")
    plt.xlabel('Number of gradient steps')
    plt.ylabel("Cross Entropy Loss")
    plt.title('Printout of Cross Entropy Loss when shuffling training examples')
    plt.legend()
    plt.savefig("images/task_3/task_3a_softmax_loss.png", dpi=200)
    plt.show()

    # Plot accuracy
    plt.ylim([0.0, 0.90])
    utils.plot_loss(train_accuracy, "Training Accuracy")
    utils.plot_loss(val_accuracy, "Validation Accuracy")
    plt.xlabel('Number of gradient steps')
    plt.ylabel("Accuracy")
    plt.title('Printout of Accuracy when shuffling training examples')
    plt.legend()
    plt.savefig("images/task_3/task_3a_softmax_accuracy.png", dpi=200)
    plt.show()