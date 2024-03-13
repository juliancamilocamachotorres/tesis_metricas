import sys
import os
import numpy as np
from skimage import metrics
import cv2
import pandas as pd
import argparse

def calculate_precision(y_true, y_pred):
    """
    Calcula la precisión de dos matrices de segmentación.

    Args:
      y_true: La matriz de segmentación de referencia.
      y_pred: La matriz de segmentación predicha.

    Returns:
      La precisión.
    """
    intersection = np.sum((y_true & y_pred) == 1)
    union = np.sum((y_true | y_pred) == 1)

    if union == 0:
        return 0
    else:
        return intersection / union


def calculate_recall(y_true, y_pred):
    """
    Calcula el recall de dos matrices de segmentación.

    Args:
      y_true: La matriz de segmentación de referencia.
      y_pred: La matriz de segmentación predicha.

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
    Calcula el miou, SSIM, F1 y DICE de dos matrices de segmentación.

    Args:
      y_true: La matriz de segmentación de referencia.
      y_pred: La matriz de segmentación predicha.

    Returns:
      Una lista con el MIoU, SSIM, F1 y DICE.
    """
    intersection = np.sum((y_true & y_pred) == 1)
    union = np.sum((y_true | y_pred) == 1)

    if union == 0:
        MIOU = 0
    else:
        MIOU = intersection / union

    SSIM = metrics.structural_similarity(y_true, y_pred)

    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)

    if precision + recall == 0:
        F1 = 0
    else:
        F1 = 2 * (precision * recall) / (precision + recall)

    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)

    if union == 0:
        DICE = 0
    else:
        DICE = (2 * intersection) / (union + 1e-12)

    return [MIOU, SSIM, F1, DICE]

def input():
    """
    Función para manejar los argumentos de entrada desde la línea de comandos y calcular las métricas.
    """
    parser = argparse.ArgumentParser(description="Script para calcular métricas de segmentación.")
    parser.add_argument("mascara_verdadera", help="Ruta de la máscara verdadera.")
    parser.add_argument("mascara_generada", help="Ruta de la máscara generada.")
    args = parser.parse_args()

    im = cv2.imread(args.mascara_generada)
    seg = cv2.imread(args.mascara_verdadera)
    seg = 1 - ((seg[:,:,0:1] == 0) + (seg[:,:,1:2] == 0) + (seg[:,:,2:3] == 254))
    seg = (seg * 255).astype('uint8').repeat(3,axis=2)

    metrics_0 = calculate_metrics(seg[:,:,0]/255>0.5, im[:,:,0]/255>0.5)
    metrics_1 = calculate_metrics(seg[:,:,1]/255>0.5, im[:,:,1]/255>0.5)
    metrics_2 = calculate_metrics(seg[:,:,2]/255>0.5, im[:,:,2]/255>0.5)
    print('Métricas para el canal #0:')
    print('  - MIoU:', round(metrics_0[0], 4))
    print('  - SSIM:', round(metrics_0[1], 4))
    print('  - F1:', round(metrics_0[2], 4))
    print('  - DICE:', round(metrics_0[3], 4))

    print('Métricas para el canal #1:')
    print('  - MIoU:', round(metrics_1[0], 4))
    print('  - SSIM:', round(metrics_1[1], 4))
    print('  - F1:', round(metrics_1[2], 4))
    print('  - DICE:', round(metrics_1[3], 4))

    print('Métricas para el canal #2:')
    print('  - MIoU:', round(metrics_2[0], 4))
    print('  - SSIM:', round(metrics_2[1], 4))
    print('  - F1:', round(metrics_2[2], 4))
    print('  - DICE:', round(metrics_2[3], 4))

if __name__ == "__main__":
    input()
