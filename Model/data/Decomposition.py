# Decomposition
from sklearn.decomposition import (
    PCA,
    KernelPCA,
    DictionaryLearning,
    FastICA,
    IncrementalPCA,
    MiniBatchDictionaryLearning,
    MiniBatchSparsePCA,
    NMF,
    SparseCoder,
    SparsePCA,
    dict_learning,
    dict_learning_online,
    fastica,
    non_negative_factorization,
    randomized_svd,
    sparse_encode,
    FactorAnalysis,
    TruncatedSVD,
    LatentDirichletAllocation,
)

from scipy.linalg import svd


class Decomposition:
    def __init__(self, X_train, X_test, y_train, y_test, n_components=2):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def PCA(self):
        pca = PCA(n_components=self.n_components)
        pca.fit(self.X_train)
        X_train_pca = pca.transform(self.X_train)
        X_test_pca = pca.transform(self.X_test)
        return X_train_pca, X_test_pca

    def KernelPCA(self):
        kpca = KernelPCA(n_components=self.n_components)
        kpca.fit(self.X_train)
        X_train_kpca = kpca.transform(self.X_train)
        X_test_kpca = kpca.transform(self.X_test)
        return X_train_kpca, X_test_kpca

    def SVD(self):
        X_train_U, X_train_s, X_train_VT = svd(self.X_train)
        X_test_U, X_test_s, X_test_VT = svd(self.X_test)
        y_train_U, y_train_s, y_train_VT = svd(self.y_train)
        y_test_U, y_test_s, y_test_VT = svd(self.y_test)
        return [
            (X_train_U, X_train_s, X_train_VT),
            (X_test_U, X_test_s, X_test_VT),
            (y_train_U, y_train_s, y_train_VT),
            (y_test_U, y_test_s, y_test_VT),
        ]

    def DictionaryLearning(self):
        dl = DictionaryLearning(n_components=self.n_components)
        dl.fit(self.X_train)
        X_train_dl = dl.transform(self.X_train)
        X_test_dl = dl.transform(self.X_test)
        return X_train_dl, X_test_dl

    def FastICA(self):
        fica = FastICA(n_components=self.n_components)
        fica.fit(self.X_train)
        X_train_fica = fica.transform(self.X_train)
        X_test_fica = fica.transform(self.X_test)
        return X_train_fica, X_test_fica

    def IncrementalPCA(self):
        ipca = IncrementalPCA(n_components=self.n_components)
        ipca.fit(self.X_train)
        X_train_ipca = ipca.transform(self.X_train)
        X_test_ipca = ipca.transform(self.X_test)
        return X_train_ipca, X_test_ipca

    def MiniBatchDictionaryLearning(self):
        mbdl = MiniBatchDictionaryLearning(n_components=self.n_components)
        mbdl.fit(self.X_train)
        X_train_mbdl = mbdl.transform(self.X_train)
        X_test_mbdl = mbdl.transform(self.X_test)
        return X_train_mbdl, X_test_mbdl

    def NMF(self):
        nmf = NMF(n_components=self.n_components)
        nmf.fit(self.X_train)
        X_train = nmf.transform(self.X_train)
        X_test = nmf.transform(self.X_test)
        return X_train, X_test

    def SparseCoder(self):
        sc = SparseCoder(n_components=self.n_components)
        sc.fit(self.X_train)
        X_train = sc.transform(self.X_train)
        X_test = sc.transform(self.X_test)
        return X_train, X_test

    def SparsePCA(self):
        spca = SparsePCA(n_components=self.n_components)
        spca.fit(self.X_train)
        X_train = spca.transform(self.X_train)
        X_test = spca.transform(self.X_test)
        return X_train, X_test

    def dict_learning(self):
        dl = dict_learning(n_components=self.n_components)
        dl.fit(self.X_train)
        X_train = dl.transform(self.X_train)
        X_test = dl.transform(self.X_test)
        return X_train, X_test

    def dict_learning_online(self):
        dto = dict_learning_online(n_components=self.n_components)
        dto.fit(self.X_train)
        X_train = dto.transform(self.X_train)
        X_test = dto.transform(self.X_test)
        return X_train, X_test

    def fastica(self):
        fastica = fastica(n_components=self.n_components)
        fastica.fit(self.X_train)
        X_train = fastica.transform(self.X_train)
        X_test = fastica.transform(self.X_test)
        return X_train, X_test

    def non_negative_factorization(self):
        nnf = non_negative_factorization(n_components=self.n_components)
        nnf.fit(self.X_train)
        X_train = nnf.transform(self.X_train)
        X_test = nnf.transform(self.X_test)
        return X_train, X_test

    def randomized_svd(self):
        rsvd = randomized_svd(n_components=self.n_components)
        rsvd.fit(self.X_train)
        X_train = rsvd.transform(self.X_train)
        X_test = rsvd.transform(self.X_test)
        return X_train, X_test

    def sparse_encode(self):
        se = sparse_encode(n_components=self.n_components)
        se.fit(self.X_train)
        X_train = se.transform(self.X_train)
        X_test = se.transform(self.X_test)
        return X_train, X_test

    def FactorAnalysis(self):
        fa = FactorAnalysis(n_components=self.n_components)
        fa.fit(self.X_train)
        X_train = fa.transform(self.X_train)
        X_test = fa.transform(self.X_test)
        return X_train, X_test

    def TruncatedSVD(self):
        tsvd = TruncatedSVD(n_components=self.n_components)
        tsvd.fit(self.X_train)
        X_train = tsvd.transform(self.X_train)
        X_test = tsvd.transform(self.X_test)
        return X_train, X_test

    def LatentDirichletAllocation(self):
        lda = LatentDirichletAllocation(n_components=self.n_components)
        lda.fit(self.X_train)
        X_train = lda.transform(self.X_train)
        X_test = lda.transform(self.X_test)
        return X_train, X_test
