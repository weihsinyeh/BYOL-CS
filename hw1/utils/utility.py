import os
from shutil import copyfile


def _create_model_training_folder(files_to_same):
    model_checkpoints_folder = '/home/weihsin/project/dlcv-fall-2024-hw1-weihsinyeh/hw1/checkpoints'
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))