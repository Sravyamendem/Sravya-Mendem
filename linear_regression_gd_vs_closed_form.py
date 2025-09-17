import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

n = 200
x = rng.uniform(0, 5, size=n)
epsilon = rng.normal(0, 1.0, size=n)
y = 3 + 4 * x + epsilon

X = np.column_stack([x, np.ones_like(x)])
theta_cf = np.linalg.inv(X.T @ X) @ (X.T @ y)
slope_cf, intercept_cf = theta_cf[0], theta_cf[1]

theta_gd = np.array([0.0, 0.0])
eta = 0.05
iters = 1000
loss_history = []

def loss_half(theta):
    y_hat = theta[0] * x + theta[1]
    r = y - y_hat
    return 0.5 / len(x) * np.sum(r**2)

for t in range(iters):
    y_hat = theta_gd[0] * x + theta_gd[1]
    r = y - y_hat
    grad_slope = -(1/len(x)) * np.sum(x * r)
    grad_intercept = -(1/len(x)) * np.sum(r)
    theta_gd = theta_gd - eta * np.array([grad_slope, grad_intercept])
    loss_history.append(loss_half(theta_gd))

slope_gd, intercept_gd = theta_gd[0], theta_gd[1]

plt.figure()
plt.scatter(x, y, s=12, label="Data")
xs = np.linspace(0, 5, 200)
ys_cf = slope_cf * xs + intercept_cf
ys_gd = slope_gd * xs + intercept_gd
plt.plot(xs, ys_cf, label="Closed-form fit")
plt.plot(xs, ys_gd, label="Gradient Descent fit")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression: Data and Fitted Lines")
plt.legend()
plt.tight_layout()
plt.savefig("linear_regression_fits.png", dpi=160)

plt.figure()
plt.plot(np.arange(1, iters + 1), loss_history, label="MSE (1/2n * sum r^2)")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Gradient Descent Loss Curve")
plt.legend()
plt.tight_layout()
plt.savefig("gd_loss_curve.png", dpi=160)

print(f"Closed-form: intercept={intercept_cf:.4f}, slope={slope_cf:.4f}")
print(f"Gradient Descent: intercept={intercept_gd:.4f}, slope={slope_gd:.4f}")
