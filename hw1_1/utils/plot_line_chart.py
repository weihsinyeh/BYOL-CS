import matplotlib.pyplot as plt
from os import path
training_loss = []
validation_loss = []
accuracy = []
epochs = []
filepath = '/home/weihsin/project/dlcv-fall-2024-hw1-weihsinyeh/hw1_1/log/finetune-SettingC_925'
with open(filepath, 'r') as f:
    epoch = 0

    for line in f:
        if 'Training Loss' in line:
            loss = float(line.split(' ')[-1])
            training_loss.append(loss)
            epochs.append(epoch)  
            epoch += 1
        elif 'Validation Loss' in line:
            v_loss = float(line.split(' ')[-1])
            validation_loss.append(v_loss)
        elif 'Validation Accuracy' in line:
            v_acc = float(line.split(' ')[-1])
            accuracy.append(v_acc)
            if epoch >= 150:
                break

plt.figure(figsize=(12, 6))
plt.plot(epochs, training_loss, label='Training Loss', color='blue')
plt.plot(epochs, validation_loss, label='Validation Loss', color='orange')
plt.plot(epochs, accuracy, label='Validation Accuracy', color='green')

plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.grid()
finename = path.basename(filepath) + '_149.jpg'
plt.savefig(finename, format='png')