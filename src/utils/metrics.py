import numpy as np
from sklearn.metrics import confusion_matrix

__author__ = "Reana Naik"



def quadratic_weighted_kappa(y_true, y_pred, n_class):
    """
    Calculate the quadratic weighted kappa score.
    """
    assert len(y_true) == len(y_pred)

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    # Confusion matrix, O.
    O_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(n_class))
    O_matrix = O_matrix.astype(np.float64)

    # Linear weights matrix, w.
    w_matrix = np.zeros((n_class, n_class))
    for i in range(n_class):
        for j in range(n_class):
            w_matrix[i][j] = ((i - j)**2)/((n_class - 1)**2)

    # Expected outcomes matrix, E.
    true_hist = np.zeros([n_class])
    for i in y_true:
        true_hist[i] += 1

    pred_hist = np.zeros([n_class])
    for i in y_pred:
        pred_hist[i] += 1

    E_matrix = np.outer(true_hist, pred_hist)
    E_matrix = E_matrix.astype(np.float64)

    # Normalize O & E.
    if O_matrix.sum() == 0 or E_matrix.sum() == 0:
        return 1.0

    O_matrix /= O_matrix.sum()
    E_matrix /= E_matrix.sum()

    # Quadratic weighted kappa, k.
    if np.sum(w_matrix * E_matrix) == 0:
        return 1.0
    qwk = 1 - (np.sum(w_matrix * O_matrix) / np.sum(w_matrix * E_matrix))

    return qwk