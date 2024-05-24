def xyxy_to_xywh(xyxy, w, h):
    """
    Convert XYXY format (x,y top left and x,y bottom right) to XYWH format (x,y center point and width, height).

    Args:
    xyxy (List[float]): [X1, Y1, X2, Y2]

    Returns: (List[float]) [X, Y, W, H]
    """
    x_temp = (xyxy[0] + xyxy[2]) / 2
    y_temp = (xyxy[1] + xyxy[3]) / 2
    w_temp = abs(xyxy[0] - xyxy[2])
    h_temp = abs(xyxy[1] - xyxy[3])
    return [xyxy[0], xyxy[1], w_temp, h_temp]


def xywh_to_xyxy(xywh):
    """
    Convert XYWH format (x,y center point and width, height) to XYXY format (x,y top left and x,y bottom right).

    Args:
    xywh (List[float]): [X, Y, W, H]

    Returns: [X1, Y1, X2, Y2]
    """
    x1 = xywh[0] - xywh[2] / 2
    y1 = xywh[1] - xywh[3] / 2
    x2 = xywh[0] + xywh[2] / 2
    y2 = xywh[1] + xywh[3] / 2
    return [x1, y1, x2, y2]