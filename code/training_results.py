import matplotlib.pyplot as plt 

def show_results( history, epochs ):

    # print( history.history )
    acc = history.history['functional_1_accuracy']
    val_acc = history.history['val_functional_1_accuracy']

    loss = history.history['functional_1_loss']
    val_loss = history.history['val_functional_1_loss']

    epochs_range = range( epochs )

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy 1')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss 1')
    plt.show()

    acc = history.history['functional_3_accuracy']
    val_acc = history.history['val_functional_3_accuracy']

    loss = history.history['functional_3_loss']
    val_loss = history.history['val_functional_3_loss']

    epochs_range = range( epochs )

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy 2')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss 2')
    plt.show()

    acc = history.history['functional_5_accuracy']
    val_acc = history.history['val_functional_5_accuracy']

    loss = history.history['functional_5_loss']
    val_loss = history.history['val_functional_5_loss']

    epochs_range = range( epochs )

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy 3')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss 3')
    plt.show()
