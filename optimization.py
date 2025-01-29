from oracles import BinaryLogistic
import numpy as np
import time
from scipy.special import expit
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
            self, loss_function='binary_logistic', step_alpha=1, step_beta=0,
            tolerance=1e-5, max_iter=1000, **kwargs
    ):
        if loss_function == 'binary_logistic':
            self.loss_func = BinaryLogistic(**kwargs)
            self.step_alpha = step_alpha
            self.step_beta = step_beta
            self.tolerance = tolerance
            self.max_iter = max_iter
        else:
            raise TypeError

    def fit(self, X, y, w_0=None, trace=False, X_test=None, y_test=None):
        iter = 0
        self.weights = w_0
        history = {}
        history['time'] = []
        history['func'] = []
        history['accuracy'] = []
        prev_time = time.time()
        history['func'].append(self.loss_func.func(X, y, self.weights))
        if X_test is not None:
            history['accuracy'].append(accuracy_score(y_test, self.predict(X_test)))
        now_time = time.time()
        history['time'].append(now_time - prev_time)
        prev_time = now_time
        while iter < self.max_iter:
            iter += 1
            now_time = time.time()
            history['time'].append(now_time - prev_time)
            prev_time = now_time
            grad = self.loss_func.grad(X, y, self.weights)
            learn_rate = self.step_alpha / (iter) ** self.step_beta
            self.weights = self.weights - learn_rate * grad
            history['func'].append(self.loss_func.func(X, y, self.weights))
            if X_test is not None:
                history['accuracy'].append(accuracy_score(y_test, self.predict(X_test)))
            if abs(history['func'][-1] - history['func'][-2]) < self.tolerance:
                break
        if trace:
            return history

    def predict(self, X):
        return np.where((X @ self.weights) > 0, 1, -1)

    def predict_proba(self, X):
        predict = np.array(expit(X @ self.weights))
        ans = np.zeros((X.shape[0], 2))
        predict = predict.reshape(X.shape[0], 1)
        ans[:, 0] = predict
        ans[:, 1] = 1 - predict
        return ans

    def get_objective(self, X, y):
        return self.loss_func.func(X, y, self.weights)

    def get_gradient(self, X, y):
        return self.loss_func.grad(X, y, self.weights)

    def get_weights(self):
        return self.weights


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
            self, loss_function='binary_logistic', batch_size=500, step_alpha=1, step_beta=0,
            tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs
    ):
        if loss_function == 'binary_logistic':
            self.loss_func = BinaryLogistic(**kwargs)
            self.batch_size = batch_size
            self.step_alpha = step_alpha
            self.step_beta = step_beta
            self.tolerance = tolerance
            self.max_iter = max_iter
            self.random_seed = random_seed
        else:
            raise TypeError

    def fit(self, X, y, w_0=None, trace=False, log_freq=1, X_test=None, y_test=None):
        np.random.seed(self.random_seed)
        iter = 0
        self.weights = w_0
        n = len(y)
        history = {}
        history['time'] = []
        history['func'] = []
        history['weights_diff'] = []
        history['epoch_num'] = []
        history['accuracy'] = []
        prev_time = time.time()
        history['func'].append(self.loss_func.func(X, y, self.weights))
        if X_test is not None:
            history['accuracy'].append(accuracy_score(y_test, self.predict(X_test)))
        history['epoch_num'].append(0)
        history['weights_diff'].append(0)
        now_time = time.time()
        history['time'].append(now_time - prev_time)
        prev_time = now_time
        prev_weights = self.weights
        epoch_num = iter * self.batch_size / n
        while epoch_num < self.max_iter:
            iter += 1

            rand_indexes = np.random.randint(0, X.shape[0], self.batch_size)
            grad = self.loss_func.grad(X[rand_indexes], y[rand_indexes], self.weights)
            learn_rate = self.step_alpha / ((iter) ** self.step_beta)
            self.weights = self.weights - learn_rate * grad

            epoch_num = iter * self.batch_size / n
            if (epoch_num - history['epoch_num'][-1]) > log_freq:
                history['epoch_num'].append(epoch_num)
                now_time = time.time()
                history['time'].append(now_time - prev_time)
                prev_time = now_time
                history['func'].append(self.loss_func.func(X, y, self.weights))
                history['weights_diff'].append(np.sum((self.weights - prev_weights) ** 2))
                if X_test is not None:
                    history['accuracy'].append(accuracy_score(y_test, self.predict(X_test)))
                prev_weights = self.weights
                if abs(history['func'][-1] - history['func'][-2]) < self.tolerance:
                    break
        if trace:
            return history
