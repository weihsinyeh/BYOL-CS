import matplotlib.pyplot as plt
from os import path
training_loss = []
validation_loss = []
accuracy = []
epochs = []
'''
filepath = './hw1_1/log/finetune-SettingC_927'
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
'''
filepath = './927'
epoch_list = []
with open(filepath, 'r') as f:
    epoch = 0
    for line in f:
        if 'Epoch:' in line:
            # read only 392
            epoch_list.append(int(line.split('.')[0].split('model_epoch')[1]))
        if 'Validation Accuracy' in line:
            v_acc = float(line.split(' ')[-1])
            accuracy.append(v_acc)

filepath = './hw1_1/log/finetune-SettingC_927'
print(len(epoch_list))

with open(filepath, 'r') as f:
    epoch = 0

    for line in f:
        if 'Training Loss' in line:
            loss = float(line.split(' ')[-1])
            if epoch in epoch_list:
                training_loss.append(loss)

        elif 'Validation Loss' in line:
            v_loss = float(line.split(' ')[-1])
            if epoch in epoch_list:
                validation_loss.append(v_loss)
            epoch += 1


print(len(training_loss))
print(len(validation_loss))
plt.figure(figsize=(12, 6))
plt.plot(epoch_list, training_loss, label='Training Loss', color='blue')
plt.plot(epoch_list, validation_loss, label='Validation Loss', color='orange')
plt.plot(epoch_list, accuracy, label='Validation Accuracy', color='green')

plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.grid()
finename = path.basename(filepath) + '.jpg'
plt.savefig(finename, format='png')