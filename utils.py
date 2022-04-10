import os
import random
import matplotlib.pyplot as plt

def split_train_val(data_path, val_size):
    data = [os.path.join(data_path, img_name)
            for img_name in os.listdir(data_path)]
    random.shuffle(data)
    return data[:int(val_size*len(data))], data[int(val_size*len(data)):]


def plot_loss(train_logs, valid_logs):
    plt.plot(train_logs, label='train_loss', marker='*')
    plt.plot(valid_logs, label='val_loss',  marker='*')
    plt.title('Loss per epoch')
    plt.ylabel('Loss')
    plt.xlabel('epochs')
    plt.legend(), plt.grid()
    plt.show()