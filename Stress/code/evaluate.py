# -*- coding: utf-8 -*-
"""Evaluation script: load the SWA test logits/labels saved under results/ and print a report.

Usage (from the repo root): python code/evaluate.py
Metrics: macro-F1, AUROC (OvR macro), AUPRC (OvR macro, average_precision_score).
"""
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize

logits = np.load('results/test_logits.npy')
labels = np.load('results/test_labels.npy')

e = np.exp(logits - logits.max(axis=1, keepdims=True))
probs = e / e.sum(axis=1, keepdims=True)
preds = logits.argmax(1)
C = probs.shape[1]
n = len(labels)

f1 = f1_score(labels, preds, average='macro')
y_bin = label_binarize(labels, classes=range(C))
auroc = roc_auc_score(y_bin, probs, multi_class='ovr', average='macro')
auprc_per = [average_precision_score(y_bin[:, c], probs[:, c]) for c in range(C)]
auprc = float(np.mean(auprc_per))

print(f'[eval] N={n}  classes={C}  class_dist={np.bincount(labels, minlength=C).tolist()}')
print(f'  macro-F1    = {f1:.4f}')
print(f'  AUROC (ovr) = {auroc:.4f}')
print(f'  AUPRC (ovr) = {auprc:.4f}')
for c in range(C):
    print(f'  class {c}: AUPRC={auprc_per[c]:.4f}')
