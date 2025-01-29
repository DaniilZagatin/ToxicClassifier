from scipy.special import expit
import numpy as np
from scipy.sparse import csr_matrix


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """

    def func(self, w):
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.

    Оракул должен поддерживать l2 регуляризацию.
    """

    def __init__(self, l2_coef=0):
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        """
        n = len(y)
        M = -y * (X @ w)
        res = np.sum(np.logaddexp(0, M)) / n + self.l2_coef / 2 * np.sum(w * w)
        return res

    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.
        """
        n = len(y)
        M = -y * (X @ w)
        sigma = expit(M)
        grad_loss = - (X.T @ (y * sigma)) / n
        grad_reg = self.l2_coef * w
        grad = grad_loss + grad_reg
        return grad
