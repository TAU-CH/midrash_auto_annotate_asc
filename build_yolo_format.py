import os
import json


def list_images(dir_path, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif")):
    """
    Return a list of image filenames (not full paths) found directly in `dir_path`.

    Parameters
    ----------
    dir_path : str
        Folder to scan.
    exts : tuple[str]
        File extensions treated as images (case-insensitive).

    Returns
    -------
    list[str]
        Sorted list of image filenames.
    """
    return sorted(
        f
        for f in os.listdir(dir_path)
        if f.lower().endswith(exts) and os.path.isfile(os.path.join(dir_path, f))
    )

def is_low_res(img):
    file_name = img['file_name']
    SKIP_MS_IDS = {
        "000", "002", "003", "030", "034", "035",
        "036", "037", "038", "039", "047", "048", "054",
        "055", "057",
    }
    ms_id = file_name.split('_')[0]
    return ms_id in SKIP_MS_IDS


def create_annotations(image_dir, coco_annotation_file, category_id):
    # Define the base directory for output labels
    labels_dir = image_dir.replace("images", "labels")
    
    # Ensure the output labels directory exists
    os.makedirs(labels_dir, exist_ok=True)
    
    # Load COCO annotations
    with open(coco_annotation_file, "r") as file:
        coco_data = json.load(file)

    skip_img_ids = {
        img['id'] for img in coco_data['images']
        if is_low_res(img) or not (img['file_name'] in list_images(image_dir))}

    # Extract image dimensions from the COCO data
    image_info = {img["id"]: img for img in coco_data["images"] if not img['id'] in skip_img_ids}
    
    # Iterate through all annotations in the COCO file
    annotations = coco_data["annotations"]
    annotations = [
        annotation for annotation in annotations
        if annotation['category_id'] == category_id and not annotation['image_id'] in skip_img_ids
    ]
    annotations_by_image = {}
    for ann in annotations:
        image_id = ann["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Process each image
    for image_id, annotations in annotations_by_image.items():
        image_filename = image_info[image_id]["file_name"]
        image_path = os.path.join(image_dir, image_filename)
        
        # Get image dimensions
        image_width = image_info[image_id]["width"]
        image_height = image_info[image_id]["height"]
        
        # Output annotation file path
        label_filename = os.path.splitext(image_filename)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_filename)
        
        # Write YOLO-style annotations to the file
        with open(label_path, "w") as label_file:
            for ann in annotations:
                bbox = ann["bbox"]  # [x_min, y_min, width, height]
                
                # Calculate normalized center_x, center_y, width, height
                center_x = (bbox[0] + bbox[2] / 2) / image_width
                center_y = (bbox[1] + bbox[3] / 2) / image_height
                width = bbox[2] / image_width
                height = bbox[3] / image_height
                
                # Write the annotation in YOLO format (class_id always 0)
                label_file.write(f"{category_id - 1} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

    print(f"Annotation files created in: {labels_dir}")

image_dir = "dataset/images/train"
coco_annotation_file = "annotations/annotations.json"
create_annotations(image_dir, coco_annotation_file, 1)

image_dir = "dataset/images/val"
coco_annotation_file = "annotations/annotations.json"
create_annotations(image_dir, coco_annotation_file, 1)
