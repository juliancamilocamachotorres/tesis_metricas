def calculate_precision(y_true, y_pred):
    """
    Calcula la precision de dos matrices de segmentacion.

    Args:
      y_true: La matriz de segmentacion de referencia.
      y_pred: La matriz de segmentacion predicha.

    Returns:
      La precision.
    """

    intersection = np.sum((y_true & y_pred) == 1)
    union = np.sum((y_true | y_pred) == 1)

    if union == 0:
        return 0
    else:
        return intersection / union


def calculate_recall(y_true, y_pred):
    """
    Calcula el recall de dos matrices de segmentacion.

    Args:
      y_true: La matriz de segmentacion de referencia.
      y_pred: La matriz de segmentacion predicha.

    Returns:
      El recall.
    """

    intersection = np.sum((y_true & y_pred) == 1)
    ground_truth = np.sum(y_true == 1)

    if ground_truth == 0:
        return 0
    else:
        return intersection / ground_truth

def calculate_metrics(y_true, y_pred):
    """
    Calcula el miou de dos matrices de segmentación.

    Args:
      y_true: La matriz de segmentación de referencia.
      y_pred: La matriz de segmentación predicha.

    Returns:
      El miou.
    """

    intersection = np.sum((y_true & y_pred) == 1)
    union = np.sum((y_true | y_pred) == 1)

    if union == 0:
        MIOU = 0
    else:
        MIOU = intersection / union
    
    """
    Calcula el SSIM de dos matrices de segmentación.

    Args:
      y_true: La matriz de segmentación de referencia.
      y_pred: La matriz de segmentación predicha.

    Returns:
      El SSIM.
    """

    SSIM = metrics.structural_similarity(y_true, y_pred)

    """
    Calcula el F1 de dos matrices de segmentacion.

    Args:
      y_true: La matriz de segmentacion de referencia.
      y_pred: La matriz de segmentacion predicha.

    Returns:
      El F1.
    """

    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)

    if precision + recall == 0:
        F1 = 0
    else:
        F1 = 2 * (precision * recall) / (precision + recall)
    
    """
    Calcula el DICE de dos matrices de segmentacion.

    Args:
      y_true: La matriz de segmentacion de referencia.
      y_pred: La matriz de segmentacion predicha.

    Returns:
      El DICE.
    """

    intersection = np.sum(y_true * y_pred)  # Calcular la intersección
    union = np.sum(y_true) + np.sum(y_pred)  # Calcular la unión

    if union == 0:
        DICE = 0
    else:
        DICE = (2 * intersection) / (union + 1e-12)

    return [MIOU, SSIM, F1, DICE]