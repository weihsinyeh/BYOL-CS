from PIL import Image
import os

folder_path = '/home/weihsin/project/dlcv-fall-2024-hw1-weihsinyeh/hw1_data/p1_data/mini/train'
folder_path = '/home/weihsin/project/dlcv-fall-2024-hw1-weihsinyeh/hw1_data/p1_data/office/train'
folder_path = '/home/weihsin/project/dlcv-fall-2024-hw1-weihsinyeh/hw1_data/p1_data/office/val'
files = os.listdir(folder_path)

def resize_picture():
    for file_name in files:
        if file_name.lower().endswith(('.jpg')):
            file_path = os.path.join(folder_path, file_name)
            output_file_path = os.path.join(folder_path, file_name)
            try:
                with Image.open(file_path) as img:
                    img_resized = img.resize((128, 128))
                    width, height = img_resized.size
                    img_resized.save(output_file_path)
                    print(f'Resized and saved {file_name} to {width}x{height}')
            except Exception as e:
                print(f'cannot handle {file_name}: {e}')

if __name__ == '__main__':
    resize_picture()