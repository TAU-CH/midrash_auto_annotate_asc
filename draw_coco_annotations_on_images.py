import os
import json
import argparse
import cv2

def draw_annotations(image, annotations):
    """
    Draws the COCO annotations on the image.
    """
    for ann in annotations:
        if 'bbox' in ann:
            x, y, w, h = map(int, ann['bbox'])

            # Draw the bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return image

def main(annotations_path, images_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load COCO annotations
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)

    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])

    # Create a mapping of image_id to annotations
    annotations_by_image = {}
    for ann in annotations:
        image_id = ann.get('image_id')
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    # Process each image
    for image_info in images:
        image_id = image_info.get('id')
        file_name = image_info.get('file_name')
        image_path = os.path.join(images_dir, file_name)

        if not os.path.exists(image_path):
            print(f"Image {file_name} not found in {images_dir}, skipping.")
            continue

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read {image_path}, skipping.")
            continue

        # Get annotations for this image
        image_annotations = annotations_by_image.get(image_id, [])

        # Draw annotations
        annotated_image = draw_annotations(image, image_annotations)

        # Save the annotated image
        output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_annotated.jpg")
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw COCO annotations on images and save them.")
    parser.add_argument("--annotations", required=True, help="Path to the COCO annotations JSON file.")
    parser.add_argument("--images_dir", required=True, help="Path to the directory containing the images.")
    parser.add_argument("--output_dir", required=True, help="Path to the directory to save annotated images.")
    
    args = parser.parse_args()
    main(args.annotations, args.images_dir, args.output_dir)