# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 01:16:18 2020

@author: maestro
"""

import numpy as np
from time import time

from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
n_components = 100
n_top_words = 10

# functions


def print_top_words(model, feature_names, n_top_words, n):
    for topic_idx, topic in enumerate(model.components_[:n]):
        message = "Topic #%d: " % topic_idx
        message += " ".join([
            feature_names[i].replace(" ", "_")
            for i in np.abs(topic).argsort()[:-n_top_words - 1:-1]
        ])
        print(message)
    print()


def components(transformer, data_samples, y):
    print("Fitting model with features ...")
    t0 = time()
    transformer.fit(data_samples, y)
    print(("done in %0.3fs." % (time() - t0)))

    print(
        f"\nTopics in model with {transformer['decomposition'].components_.shape} features:\n"
    )
    feature_names = transformer['vectorizer'].get_feature_names()
    print_top_words(transformer['decomposition'], feature_names, n_top_words,
                    10)


# split data
def split_data(X, y, rs):
    test_portion = 0.33
    return train_test_split(X,
                            y,
                            test_size=test_portion,
                            random_state=int(rs),
                            shuffle=True,
                            stratify=y)


class Dense(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.todense()

    def fit_transform(self, X, y=None):
        return X.todense()


def checking(pipe, data_samples, y, r_seed=0, n_jobs=-1, cv=5, data_model_name = "default"):
    # split data to train/validation sets
    # X_train, X_validation, y_train, y_validation = split_data(
    #     data_samples, y, r_seed)

    # print("Fitting train ...")
    # t0 = time()
    # pipe.fit(X_train, y_train)
    # print(("done in %0.3fs." % (time() - t0)))

    # print("Predicting validation ...")
    # t0 = time()
    # probs = pipe.predict_proba(X_validation)[:, 1]
    # print(("done in %0.3fs." % (time() - t0)))

    # print(f"AUC score : {roc_auc_score(y_validation, probs).round(4)}")
    # print("Cross-validation ...")
    # t0 = time()
    if hasattr(pipe, 'named_steps'):
        pipe_name = str(pipe.named_steps['cls'])
    else:
        pipe_name = "+".join(pipe.named_estimators.keys())
    for scoring in ['f1', 'accuracy', 'roc_auc']:
        print(f"{pipe_name} with {data_model_name} model {str(scoring).upper()} score:")
        scores = cross_val_score(pipe,
                                data_samples,
                                y,
                                cv=cv,
                                n_jobs=n_jobs,
                                scoring=scoring)
        print((list(scores.round(4)), scores.mean().round(4), scores.std().round(4)))
    # print(("done in %0.3fs." % (time() - t0)))
    print()