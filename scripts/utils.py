def to_yolo(bbox, img_width, img_height):
    """Convert bounding box from (x_min, y_min, width, height) to
    (x_center, y_center, width, height), normalized to image size."""
    x_min, y_min, width, height = bbox
    x_center = x_min + width / 2.0
    y_center = y_min + height / 2.0

    # Normalize
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height

    return [x_center, y_center, width, height]
