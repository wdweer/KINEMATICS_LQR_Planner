"""

LQR local path planning

author: Atsushi Sakai (@Atsushi_twi)

"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

SHOW_ANIMATION = True


class LQRPlanner:

    def __init__(self):
        self.MAX_TIME = 100.0  # Maximum simulation time
        self.DT = 0.1  # Time tick
        self.GOAL_DIST = 0.1
        self.MAX_ITER = 150
        self.EPS = 0.01

    def lqr_planning(self, sx, sy, gx, gy, show_animation=True):

        rx, ry = [sx], [sy]

        x = np.array([sx - gx, sy - gy, 0, 0]).reshape(4, 1)  # State vector

        # Linear system model
        A, B = self.get_system_model()

        found_path = False

        time = 0.0
        while time <= self.MAX_TIME:
            time += self.DT

            u = self.lqr_control(A, B, x)

            x = A @ x + B @ u

            rx.append(x[0, 0] + gx)
            ry.append(x[1, 0] + gy)

            d = math.hypot(gx - rx[-1], gy - ry[-1])
            if d <= self.GOAL_DIST:
                found_path = True
                break

            # animation
            if show_animation:  # pragma: no cover
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                        lambda event: [exit(0) if event.key == 'escape' else None])
                plt.plot(sx, sy, "or")
                plt.plot(gx, gy, "ob")
                plt.plot(rx, ry, "-r")
                plt.axis("equal")
                plt.pause(1.0)

        if not found_path:
            print("Cannot found path")
            return [], []

        return rx, ry

    def solve_dare(self, A, B, Q, R):
        """
        solve a discrete time_Algebraic Riccati equation (DARE)
        """
        X, Xn = Q, Q

        for i in range(self.MAX_ITER):
            Xn = A.T @ X @ A - A.T @ X @ B @ np.linalg.inv(R + B.T @ X @ B) @ B.T @ X @ A + Q
            # if np.allclose(X, Xn, atol=tolerance):
            #     break
            X = Xn

        return Xn

    def dlqr(self, A, B, Q, R):
        """Solve the discrete time lqr controller.
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        # ref Bertsekas, p.151
        """

        # first, try to solve the ricatti equation
        X = self.solve_dare(A, B, Q, R)

        # compute the LQR gain
        K = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

        eigValues = la.eigvals(A - B @ K)

        return K, X, eigValues

    def get_system_model(self):
        self.v=10
        self.theta=0
        self.delta=0
    # Constants (you can adjust these for your vehicle's characteristics)
        Lf = 1.5  # Distance from center of mass to front axle [m]
        Lr = 1.5  # Distance from center of mass to rear axle [m]
        wheelbase = Lf + Lr  # Total wheelbase [m]

        # State Transition Matrix (A)
        A = np.array([[1.0, 0.0, -self.DT * self.v * np.sin(self.theta), self.DT * np.cos(self.theta)],
                    [0.0, 1.0,  self.DT * self.v * np.cos(self.theta), self.DT * np.sin(self.theta)],
                    [0.0, 0.0, 1.0, self.DT * np.tan(self.delta) / wheelbase],
                    [0.0, 0.0, 0.0, 1.0]])

        # Input Matrix (B)
        B = np.array([[0.0, 0.0],
                    [0.0, 0.0],
                    [self.DT / wheelbase, 0.0],
                    [0.0, self.DT]]).reshape(4, 2)

        # State: [x, y, theta, v]
        # Inputs: [steering angle (delta), longitudinal acceleration (a)]
        return A, B

    def lqr_control(self, A, B, x):

        Kopt, X, ev = self.dlqr(A, B, np.eye(4), np.eye(2))
        print(Kopt)
        u = -Kopt @ x

        return u


def main():
    print(__file__ + " start!!")

    ntest = 10  # number of goal
    area = 100.0  # sampling area

    lqr_planner = LQRPlanner()

    for i in range(ntest):
        sx = 6.0
        sy = 6.0
        gx = random.uniform(-area, area)
        gy = random.uniform(-area, area)

        rx, ry = lqr_planner.lqr_planning(sx, sy, gx, gy, show_animation=SHOW_ANIMATION)

        if SHOW_ANIMATION:  # pragma: no cover
            plt.plot(sx, sy, "or")
            plt.plot(gx, gy, "ob")
            plt.plot(rx, ry, "-r")
            plt.axis("equal")
            plt.pause(1.0)


if __name__ == '__main__':
    main()
