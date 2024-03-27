import numpy as np
from numpy import array
from scipy.spatial.distance import cdist

from tqdm import tqdm


class AffinityPropagation:
    def __init__(self, vectors: array):
        self.matrixSize = vectors.shape[0]

        self.S = cdist(vectors.tolist(), vectors.tolist())
        self.R = np.zeros([self.matrixSize, self.matrixSize])
        self.A = np.zeros([self.matrixSize, self.matrixSize])

    def affinity_propagation(self, max_iter=200, conv_iter=15, damping=0.5):
        """
        Реализация алгоритма Affinity Propagation.

        Параметры:
        - S: матрица сходства (размером NxN, где N - количество элементов).
        - max_iter: максимальное количество итераций.
        - conv_iter: количество итераций для сходимости.
        - damping: коэффициент затухания (между 0.5 и 1).

        Возвращает:
        - labels: метки кластеров для каждого элемента.
        """
        self.__damping = damping
        # Основной цикл
        for m in tqdm(range(max_iter)):
            # Обновление матрицы ответственности
            self.__Responsibility_Calc()

            # Обновление матрицы доступности
            self.__Availability_Calc()

            # Проверка на сходимость
            if m > conv_iter and np.allclose(self.R, self.__R_old) and np.allclose(self.A, self.__A_old):
                break

        # Вычисление итоговых меток кластеров
        E = self.R + self.A
        listIndexMaxItem = np.argmax(E, axis=1)
        labels = np.unique(listIndexMaxItem, return_inverse=True)[1]

        return labels

    def __Responsibility_Calc(self):
        self.__R_old = self.R.copy()
        AS = self.A + self.S

        listIndexMaxItem = np.argmax(AS, axis=1)
        listMaxItem = np.max(AS, axis=1)

        AS[range(self.matrixSize), listIndexMaxItem] = -np.inf

        listSecondMaxItem = np.max(AS, axis=1)

        self.R = self.S - listMaxItem
        self.R[range(self.matrixSize), listIndexMaxItem
        ] = self.S[range(self.matrixSize), listIndexMaxItem] - listSecondMaxItem

        self.R = (1 - self.__damping) * self.R + self.__damping * self.__R_old

    def __Availability_Calc(self):
        self.__A_old = self.A.copy()
        Rp = np.maximum(self.R, 0)
        np.fill_diagonal(Rp, self.R.diagonal())
        self.A = np.sum(Rp, axis=0) - Rp
        dA = np.diag(self.A)
        self.A = np.minimum(self.A, 0)
        self.A[range(self.matrixSize), range(self.matrixSize)] = dA
        self.A = (1 - self.__damping) * self.A + self.__damping * self.__A_old


if __name__ == '__main__':
    x = np.random.rand(500, 40)
    AP = AffinityPropagation(x)
    print(AP.affinity_propagation())
