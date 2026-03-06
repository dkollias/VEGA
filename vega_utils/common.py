"""Shared utilities for VEGA."""

from __future__ import annotations

from tqdm import tqdm


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducibility."""
    print('Seed:', seed)
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print("Is CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Current CUDA device:", torch.cuda.current_device())
        print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("random.randint:", random.randint(0, 100))
    print("numpy.random.randint:", np.random.randint(0, 100))
    print("torch.rand:", torch.rand(1).item())


emotion_labels = {
    'IEMOCAP': ['happy', 'sad', 'neutral', 'anger', 'excited', 'frustration'],
    'MELD': ['neutral', 'surprise', 'fear', 'sad', 'happy', 'disgust', 'anger'],
}


def list_image_file_abs_path_recursive(folder_path, path_remove_content=None, return_format='Path'):
    import os
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif')

    if path_remove_content:
        path_remove_content = path_remove_content.rstrip(os.sep) + os.sep

    file_paths = []
    non_image_files = []
    file_count = 0
    image_file_count = 0
    lock = threading.Lock()

    def process_directory(root, files):
        local_image_count = 0
        local_non_image = []
        local_file_paths = []

        for f in files:
            full_path = os.path.join(root, f)
            if f.lower().endswith(image_extensions):
                processed_path = full_path.replace(path_remove_content, '') if path_remove_content else full_path
                final_path = Path(processed_path) if return_format == 'Path' else processed_path
                local_file_paths.append(final_path)
                local_image_count += 1
            else:
                local_non_image.append(f)

        with lock:
            file_paths.extend(local_file_paths)
            non_image_files.extend(local_non_image)
            nonlocal file_count, image_file_count
            file_count += len(files)
            image_file_count += local_image_count
            pbar.update(1)

    dirs_queue = []
    for root, _, files in os.walk(folder_path):
        dirs_queue.append((root, files))

    with tqdm(total=len(dirs_queue), desc="🚀 Processing directories") as pbar:
        with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
            futures = []
            for root, files in dirs_queue:
                futures.append(executor.submit(process_directory, root, files))
            for future in futures:
                future.result()

    print(f'\n{"=" * 40} Statistics {"=" * 40}')
    print(f"Total files: {file_count} | Image files: {image_file_count}")
    if non_image_files:
        print(f"\nFound {len(non_image_files)} non-image files:")
        for idx, f in enumerate(non_image_files[:5]):
            print(f"  {idx + 1}. {f}")
        if len(non_image_files) > 5:
            print(f"  ...and {len(non_image_files) - 5} more files")

    return file_paths
