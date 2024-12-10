import numpy as np

class KalmanFilter():
    def __init__(self, x0, sig_Q, sig_R, sig_0, A, C):
        self.x = x0
        self.sig_x = sig_0
        self.sig_Q = sig_Q
        self.sig_R = sig_R

        self.A = A
        self.C = C

    # source: https://en.wikipedia.org/wiki/Kalman_filter
    def predict(self):
        self.x_ = self.A @ self.x 
        self.sig_x_ = self.A  @ self.sig_x @ self.A.T + self.sig_Q

    def update(self, y):
        # Implicitly does the predict step
        self.predict()

        # Then do update rule
        y_ = y - self.C @ self.x_

        S = self.C @ self.sig_x_ @ self.C.T + self.sig_R
        K = self.sig_x_ @ self.C.T @ np.linalg.inv(S)
        
        self.x = self.x_ + K @ y_
        self.sig_x = self.sig_x_ - K @ self.C @ self.sig_x_

    def get_estimate(self):
    # returns mean and covariance
        return self.x, self.sig_x