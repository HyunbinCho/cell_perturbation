from pathlib import Path
import os


def best_model_getter(path, model_index):
    """
    load a model checkpoint of the best performance
    model checkpoint was saved at the end of each epoch when average loss across data is lower than previous best point.

    so, this simply brings the highest number among checkpoints
    """
    print('\n>> Finding best deep learning model path')

    files = os.listdir(os.path.join(path, model_index))
    best_file_path = ''
    idx = 0
    for file in files:
        if 'CP' in file:
            f = file.replace('.pth', '')
            f = f.replace('CP', '')
            if idx < int(f):
                idx = int(f)
                best_file_path = os.path.join(path, model_index, file)

    print('---> Done -- Best model path : ', best_file_path)
    return best_file_path
