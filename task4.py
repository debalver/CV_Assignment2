import numpy as np
import utils
import matplotlib.pyplot as plt
import typing
import time
from sklearn.utils import shuffle
from tqdm import tqdm
from task2a import cross_entropy_loss, SoftmaxModel, one_hot_encode, pre_process_images, find_mean_and_deviation
from task3 import calculate_accuracy, train
np.random.seed(0)


if __name__ == "__main__":

    # Measure execution time
    start = time.time()

    # Choose which mode to run
    all_tricks = False

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
    neurons_per_layer = [8, 10]
    momentum_gamma = .9  # Task 3 hyperparameter

    # Settings for task 3 and 4. Keep all to false for task 2.
    use_shuffle = False
    use_improved_sigmoid = False
    use_improved_weight_init = True
    use_momentum = False

    # Create the network and train it
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init
    )


    model, train_loss, val_loss, train_accuracy, val_accuracy = train(
        model,
        [X_train, Y_train, X_val, Y_val, X_test, Y_test],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_shuffle=use_shuffle,
        use_momentum=use_momentum,
        momentum_gamma=momentum_gamma,
        all_tricks=all_tricks)

    # Process the results in something readable
    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Test Cross Entropy Loss:",
          cross_entropy_loss(Y_test, model.forward(X_test)))

    print("Final Train accuracy:",
          calculate_accuracy(model.forward(X_train), Y_train, model))
    print("Final Validation accuracy:",
          calculate_accuracy(model.forward(X_val), Y_val, model))
    print("Final Test accuracy:",
          calculate_accuracy(model.forward(X_test), Y_test, model))


    title_tricks = str()
    if use_shuffle: title_tricks += "&shuffle"
    if use_improved_sigmoid: title_tricks += "&impr_sigmoid"
    if use_improved_weight_init: title_tricks += "&impr_weights"
    if use_momentum: title_tricks += "&momemtum"

    # Plot loss
    plt.ylim([0.0, 0.5])
    utils.plot_loss(train_loss, "Training Loss")
    utils.plot_loss(val_loss, "Validation Loss")
    plt.xlabel('Number of gradient steps')
    plt.ylabel("Cross Entropy Loss")
    plt.title('Printout of Loss '+title_tricks)
    plt.legend()
    plt.savefig("images/task_4/task_4_loss"+title_tricks+".png", dpi=200)
    plt.show()

    # Plot accuracy
    plt.ylim([0.8, 1.0])
    utils.plot_loss(train_accuracy, "Training Accuracy")
    utils.plot_loss(val_accuracy, "Validation Accuracy")
    plt.xlabel('Number of gradient steps')
    plt.ylabel("Accuracy")
    plt.title('Printout of Accuracy '+title_tricks)
    plt.legend()
    plt.savefig("images/task_4/task_4_accuracy"+title_tricks+".png", dpi=200)
    plt.show()
