import os
import sys
import pandas as pd
import numpy as np
from sklearn import metrics

from lazyqsar.agnostic import LazyBinaryClassifier
from lazyqsar.utils.logging import logger

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

# You need to prepare the right descriptors and data for this script

results_foker = os.path.join(root, "results")
if not os.path.exists(results_foker):
    os.makedirs(results_foker)

descriptor_folder = os.path.join(root, "tdcomms")
if not os.path.exists(descriptor_folder):
    os.mkdir(descriptor_folder)

model_folder = os.path.join(root, "models")
if not os.path.exists(model_folder):
    os.mkdir(model_folder)

def fit_and_evaluate(benchmark_name):
    desc_file_train = os.path.join(descriptor_folder, f"train_{benchmark_name}_ok.csv")
    desc_file_test = os.path.join(descriptor_folder, f"test_{benchmark_name}_ok.csv")
    logger.info("Binary classification task")
    y_train = pd.read_csv(os.path.join(descriptor_folder, f"train_{benchmark_name}.csv"))["y_true"]
    y_test = pd.read_csv(os.path.join(descriptor_folder, f"test_{benchmark_name}.csv"))["y_true"]
    model = LazyBinaryClassifier()
    X_train = np.array(pd.read_csv(desc_file_train))
    X_test = np.array(pd.read_csv(desc_file_test))
    model.fit(X=X_train, y=y_train)
    y_prob = model.predict_proba(X=X_test)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
    auroc = metrics.auc(fpr, tpr)
    return auroc

scores = {}
datasets = ["ames", "bbb-martins", "bioavailability-ma", "dili", "herg", "hia-hou", "pgp-broccatelli"]
for b in datasets:
    output_dir = os.path.join(model_folder, b)
    results = fit_and_evaluate(b)
    scores[b] = results
    print("===")
    print(f"Dataset {b} AUROC:", results)
    print("===")
print(scores)
df = pd.DataFrame(list(scores.items()), columns=["benchmark", "score"])
df.to_csv(os.path.join(results_folder, "summary_scores.csv"), index=False)