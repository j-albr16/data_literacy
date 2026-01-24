import pandas as pd
import os
from argparse import ArgumentParser

def add_bins(bins):
    def bin_value(value):
        for i, bin in enumerate(bins):
            if value <=bin:
                return i

        return len(bins)
    return bin_value

def normalize_filename(filename):
    """Replace German umlauts with ASCII equivalents"""
    import unicodedata

    # Normalize Unicode to composed form (NFC) to ensure umlauts are single characters
    filename = unicodedata.normalize('NFC', str(filename))

    replacements = {
        'ä': 'ae', 'ö': 'oe', 'ü': 'ue',
        'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue',
        'ß': 'ss', 'ğ': 'g', 'ć': 'c', 
        'ç':'c', 'é':'e', 'é': 'e'
    }
    for old, new in replacements.items():
        filename = filename.replace(old, new)
    return filename

def rename_files(base_dir='party_members'):
    """Rename all files with umlauts in the directory tree"""
    for file in os.listdir(base_dir):
        path = os.path.join(base_dir, file)

        if not os.path.isdir(path):
            continue

        for f in os.listdir(path):
            # rename path
            old_path = os.path.join(path, f)
            new_filename = normalize_filename(f)
            new_path = os.path.join(path, new_filename)
            os.rename(old_path, new_path)




def get_process_args():
    parser = ArgumentParser()
    parser.add_argument("--politician_reference_csv",
                        help="Path to the .csv file, that contains the paths to the images of the politicians that are being used as reference for classification", default='politicians/data.csv')
    parser.add_argument("--article_data_csv",
                        help="Directory to the .csv containing the article data. (e.g. image path, newspaper, ...)", default='politician_data_set/politicians.csv')
    parser.add_argument("--politician_base_dir", 
                        help="Base directory of the politician dataset. Contains the .csv pointing to all the images")
    parser.add_argument("--num_images",
                        type=int,
                        help="How many images to process (For e.g. debugging.)",
                        default=-1)
    parser.add_argument("--data_dir", type=str, default='data')
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--out_name", type=str, required=False, default="out.csv")
    parser.add_argument("--omit_tqdm", action="store_true", default=False)
    return parser.parse_args()
