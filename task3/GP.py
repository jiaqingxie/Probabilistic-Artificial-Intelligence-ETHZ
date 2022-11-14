import numpy as np

class GP:
    def __init__(self, ker, noise):
        self.ker = ker
        self.noise = noise

    def k(self, x1, x2):
        return self.ker([x1],[x2])
    def mu(self, x):
        return 0
        #return self.h*x1.dot(x2)

    def mu_post(self, x):
        n = self.A.shape[0]
        k_xA = np.zeros(n)
        for i in range(0, n):
            k_xA[i] = self.k(x, self.A[i, :])

        return self.mu(x) + k_xA.dot(self.B_dot_delta)

    def k_post(self, x1, x2):
        n = self.A.shape[0]
        k_x1A = np.zeros(n)
        for i in range(0, n):
            k_x1A[i] = self.k(x1, self.A[i, :])
        k_x2A = np.zeros(n)
        for i in range(0, n):
            k_x2A[i] = self.k(x2, self.A[i, :])

        return self.k(x1, x2) - (k_x1A.transpose()).dot(self.B.dot(k_x2A))

    def compute_B(self, train_x):
        n = train_x.shape[0]
        K = np.zeros((n, n))

        for col in range(0, n):
            for row in range(0, n):
                K[row, col] = self.k(train_x[row], train_x[col])

        self.B = np.linalg.inv(K + self.noise*np.eye(n))

    def fit(self, train_x, train_y):
        n = train_x.shape[0]
        self.compute_B(train_x)
        # mu_A calculation
        mu_A = np.zeros((n,1))
        for row in range(0, n):
            mu_A[row,0] = self.mu(train_x[row, :])
        # Store A and delta
        self.A = train_x
        self.B_dot_delta = self.B.dot(train_y - mu_A)