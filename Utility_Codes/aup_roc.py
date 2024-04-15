import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE

def train_model_and_get_probabilities_smote_strat(model, X, y, smote, scaler, skf):
    # Binarize the output
    y_bin = label_binarize(y, classes=np.unique(y))
    n_classes = y_bin.shape[1]
    # lists to store the tpr, fpr, precision, recall for each class
    tpr_lists = [[] for _ in range(n_classes)]
    fpr_lists = [[] for _ in range(n_classes)]
    precision_lists = [[] for _ in range(n_classes)]
    recall_lists = [[] for _ in range(n_classes)]
    # running the stratified kfold
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # scaling the data
        X_train_scaled, y_train_resampled = smote.fit_resample(X_train, y_train)

        X_train_scaled = scaler.fit_transform(X_train_scaled)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train_resampled)

        y_prob = model.predict_proba(X_test_scaled)
        # calculating the roc and pr values
        for class_index in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[test_index][:, class_index], y_prob[:, class_index])
            tpr_lists[class_index].append(np.interp(np.linspace(0, 1, 100), fpr, tpr))
            fpr_lists[class_index].append(np.linspace(0, 1, 100))

            precision, recall, _ = precision_recall_curve(y_bin[test_index][:, class_index], y_prob[:, class_index])
            precision_lists[class_index].append(np.interp(np.linspace(0, 1, 100), recall[::-1], precision[::-1]))
            recall_lists[class_index].append(np.linspace(0, 1, 100))

    return tpr_lists, fpr_lists, precision_lists, recall_lists

def train_model_and_get_probabilities_strat(model, X, y, smote, scaler, skf):
    # binarize the output
    y_bin = label_binarize(y, classes=np.unique(y))
    n_classes = y_bin.shape[1]
    # lists to store the tpr, fpr, precision, recall for each class
    tpr_lists = [[] for _ in range(n_classes)]
    fpr_lists = [[] for _ in range(n_classes)]
    precision_lists = [[] for _ in range(n_classes)]
    recall_lists = [[] for _ in range(n_classes)]
    # running the stratified kfold
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train_scaled, y_train_resampled = (X_train, y_train)
        # scaling the data
        X_train_scaled = scaler.fit_transform(X_train_scaled)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train_resampled)

        y_prob = model.predict_proba(X_test_scaled)
        # calculating the roc and pr values
        for class_index in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[test_index][:, class_index], y_prob[:, class_index])
            tpr_lists[class_index].append(np.interp(np.linspace(0, 1, 100), fpr, tpr))
            fpr_lists[class_index].append(np.linspace(0, 1, 100))

            precision, recall, _ = precision_recall_curve(y_bin[test_index][:, class_index], y_prob[:, class_index])
            precision_lists[class_index].append(np.interp(np.linspace(0, 1, 100), recall[::-1], precision[::-1]))
            recall_lists[class_index].append(np.linspace(0, 1, 100))

    return tpr_lists, fpr_lists, precision_lists, recall_lists


def train_model_and_get_probabilities_smote(model, X, y, smote, scaler, skf):
    # binarize the output
    y_bin = label_binarize(y, classes=np.unique(y))
    n_classes = y_bin.shape[1]
    # lists to store the tpr, fpr, precision, recall for each class
    tpr_lists = [[] for _ in range(n_classes)]
    fpr_lists = [[] for _ in range(n_classes)]
    precision_lists = [[] for _ in range(n_classes)]
    recall_lists = [[] for _ in range(n_classes)]
     # running the kfold
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # scaling the data
        X_train_scaled, y_train_resampled = smote.fit_resample(X_train, y_train)

        X_train_scaled = scaler.fit_transform(X_train_scaled)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train_resampled)

        y_prob = model.predict_proba(X_test_scaled)
        # calculating the roc and pr values
        for class_index in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[test_index][:, class_index], y_prob[:, class_index])
            tpr_lists[class_index].append(np.interp(np.linspace(0, 1, 100), fpr, tpr))
            fpr_lists[class_index].append(np.linspace(0, 1, 100))

            precision, recall, _ = precision_recall_curve(y_bin[test_index][:, class_index], y_prob[:, class_index])
            precision_lists[class_index].append(np.interp(np.linspace(0, 1, 100), recall[::-1], precision[::-1]))
            recall_lists[class_index].append(np.linspace(0, 1, 100))

    return tpr_lists, fpr_lists, precision_lists, recall_lists

def plot_roc_curves_multiclass(tpr_lists, fpr_lists, save_folder=None, model_name=None):
    # custom palette, font style, size and color for the classes
    custom_palette = {1: 'green', 2: '#ff9100', 3: 'purple', 4: 'brown'}

    plt.figure(figsize=(8, 8))
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12
    # plotting the  AUC plot
    for class_index, (tpr_list, fpr_list) in enumerate(zip(tpr_lists, fpr_lists)):
        mean_tpr = np.mean(tpr_list, axis=0)
        mean_fpr = np.mean(fpr_list, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        if class_index == 9:
            label = f'Class {class_index+2} (AUC = {mean_auc:.2f})'
        elif class_index == 10:
            label = f'Class {class_index+2} (AUC = {mean_auc:.2f})'
        else:
            label = f'Class {class_index+1} (AUC = {mean_auc:.2f})'

        plt.plot(mean_fpr, mean_tpr,label=label,color=custom_palette.get(class_index + 1))

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend(loc='lower right')

    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder, f'roc_curves_{model_name}.png'))
    else:
        plt.show()

def plot_pr_curves_multiclass(precision_lists, recall_lists, save_folder=None, model_name=None):
    # custom palette, font style, size and color for the classes
    custom_palette = {1: 'green', 2: '#ff9100', 3: 'purple', 4: 'brown'}

    plt.figure(figsize=(8, 8))
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12
    # plotting the  AUPRC plot
    for class_index, (precision_list, recall_list) in enumerate(zip(precision_lists, recall_lists)):
        mean_precision = np.mean(precision_list, axis=0)
        mean_recall = np.mean(recall_list, axis=0)
        mean_aup = auc(mean_recall, mean_precision)
        
        if class_index == 9:
            label = f'Class {class_index+2} (AUP = {mean_aup:.2f})'
        elif class_index == 10:
            label = f'Class {class_index+2} (AUP = {mean_aup:.2f})'
        else:
            label = f'Class {class_index+1} (AUP = {mean_aup:.2f})'

        plt.plot(mean_recall, mean_precision, label=label, color=custom_palette.get(class_index + 1))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    plt.legend(loc='lower right')

    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder, f'pr_curves_{model_name}.png'))
    else:
        plt.show()


# SMOTE and stratified kfold
def plot_roc_and_pr_curves_multiclass_smote_strat(model, X, y, n_splits=5, save_folder=None, model_name=None):
    # function to plot the roc and pr curves for multiclass classification
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()
    smote = SMOTE(random_state=42)

    tpr_lists, fpr_lists, precision_lists, recall_lists = train_model_and_get_probabilities_smote_strat(model, X, y, smote, scaler, skf)
    
    plot_roc_curves_multiclass(tpr_lists, fpr_lists, save_folder, model_name)
    plot_pr_curves_multiclass(precision_lists, recall_lists, save_folder, model_name)

# Stratified K-fold
def plot_roc_and_pr_curves_multiclass_strat(model, X, y, n_splits=5, save_folder=None, model_name=None):
    # function to plot the roc and pr curves for multiclass classification
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()


    tpr_lists, fpr_lists, precision_lists, recall_lists = train_model_and_get_probabilities_strat(model, X, y, smote, scaler, skf)
    
    plot_roc_curves_multiclass(tpr_lists, fpr_lists, save_folder, model_name)
    plot_pr_curves_multiclass(precision_lists, recall_lists, save_folder, model_name)

# SMOTE and K-fold
def plot_roc_and_pr_curves_multiclass_smote_kfold(model, X, y, n_splits=5, save_folder=None, model_name=None):
    # function to plot the roc and pr curves for multiclass classification
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()
    smote = SMOTE(random_state=42)

    tpr_lists, fpr_lists, precision_lists, recall_lists = train_model_and_get_probabilities_smote(model, X, y, smote, scaler, skf)
    
    plot_roc_curves_multiclass(tpr_lists, fpr_lists, save_folder, model_name)
    plot_pr_curves_multiclass(precision_lists, recall_lists, save_folder, model_name)