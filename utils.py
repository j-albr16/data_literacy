import os

def normalize_filename(filename):
    """Replace German umlauts with ASCII equivalents"""
    import unicodedata

    # Normalize Unicode to composed form (NFC) to ensure umlauts are single characters
    filename = unicodedata.normalize('NFC', str(filename))

    replacements = {
        'ä': 'ae', 'ö': 'oe', 'ü': 'ue',
        'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue',
        'ß': 'ss'
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


