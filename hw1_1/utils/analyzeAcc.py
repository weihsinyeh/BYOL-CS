def extract_max_validation_accuracy(file_path):
    max_accuracy = -1 
    max_epoch = -1 
    epoch = 0
    with open(file_path, 'r') as file:
        for line in file:
            if 'Epoch:' in line:
                epoch = (int(line.split('.')[0].split('model_epoch')[1]))
            if "Validation Accuracy:" in line:            
                accuracy = float(line.split("Validation Accuracy:")[1].strip())
                if accuracy > max_accuracy and epoch <= 150:
                    max_accuracy = accuracy
                    max_epoch = epoch
                epoch += 1
    
    return max_epoch, max_accuracy

file_path = './hw1_1/log/finetune-SettingC_927'
max_epoch, max_accuracy = extract_max_validation_accuracy(file_path)
print(f"Max Validation Accuracy: {max_accuracy} at Epoch {max_epoch}")