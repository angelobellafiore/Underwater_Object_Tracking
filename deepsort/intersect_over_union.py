def get_iou(bb1, bb2):
    """Intersection over Union between two bounding boxes

                    area of overlap
                    –––––––––––––––
                     area of union
    """
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2

    assert x1 < (x1+w1)
    assert y1 < (y1+h1)
    assert x2 < (x2+w2)
    assert y2 < (y2+h2)

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1+w1, (x2+w2))
    y_bottom = min(y1+h1, (y2+h2))

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = ((x1+w1) - x1) * ((y1+h1) - y1)
    bb2_area = ((x2+w2) - x2) * ((y2+h2) - y2)

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou