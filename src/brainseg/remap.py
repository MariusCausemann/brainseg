import nibabel as nib
import fastremap
import numpy as np
import sys

def load_label_map(file_path):
    """Parses the txt file to create a dictionary of {name: id}."""
    label_to_id = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                # ID is the first element, Name is the second
                label_id = int(parts[0])
                label_name = parts[1]
                label_to_id[label_name] = label_id
        return label_to_id
    except FileNotFoundError:
        print(f"Error: Label file not found at {file_path}")
        sys.exit(1)

def remap(img, old_labels, new_labels):
    mapping = {}
    data = img.get_fdata().astype(np.int32)

    for name, old_id in old_labels.items():
        if name in new_labels:
            mapping[old_id] = new_labels[name]
        else:
            # Default missing labels to 0 (Background)
            mapping[old_id] = 0

    remapped_data = fastremap.remap(data, mapping)
    
    return nib.Nifti1Image(remapped_data, img.affine, img.header)

def remap_file(infile, old_label_txt, new_label_txt, outfile):
    img = nib.load(infile)
    
    old_labels = load_label_map(old_label_txt)
    new_labels = load_label_map(new_label_txt)
    
    new_img = remap(img, old_labels, new_labels)
    nib.save(new_img, outfile)