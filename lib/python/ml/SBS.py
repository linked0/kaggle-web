from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):

        X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=self.test_size,
                                 random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

def example():
    print("k_feat = [len(k) for k in sbs.subsets_]\n\
    plt.plot(k_feat, sbs.scores_, marker='o')\n\
    plt.ylim([0.7, 1.1])\n\
    plt.ylabel('Accuracy')\n\
    plt.xlabel('Number of features')\n\
    plt.grid()\n\
    plt.show()\n\
    k5 = list(sbs.subsets_[8])\n\
    print(df_wine.columns[1:][k5])\n\
    Index(['Alcohol', 'Malic acid', 'Alcalinity of ash', 'Hue', 'Proline'], dtype='object')\n\
    knn.fit(X_train_std, y_train)\n\
    print('Training accuracy:', knn.score(X_train_std, y_train))\n\
    Training accuracy: 0.983870967742\n\
    print('Test accuracy:', knn.score(X_test_std, y_test))\n\
    Test accuracy: 0.944444444444")