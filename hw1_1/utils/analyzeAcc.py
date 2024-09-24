def extract_max_validation_accuracy(file_path):
    max_accuracy = -1 
    max_epoch = -1 
    epoch = 0
    with open(file_path, 'r') as file:
        for line in file:
            if "Validation Accuracy:" in line:            
                accuracy = float(line.split("Validation Accuracy:")[1].strip())
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    max_epoch = epoch
                epoch += 1
    
    return max_epoch, max_accuracy

file_path = '/home/weihsin/project/dlcv-fall-2024-hw1-weihsinyeh/hw1_1/log/finetune-SettingA' 
max_epoch, max_accuracy = extract_max_validation_accuracy(file_path)
print(f"Max Validation Accuracy: {max_accuracy} at Epoch {max_epoch}")