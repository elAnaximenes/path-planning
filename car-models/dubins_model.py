# See planning.cs.uiuc.edu chapter 13 for reference material
import math
import matplotlib.pyplot as plt

def simpson(f, a, b, epsilon = 0.001, level=0, level_max = 10):
    # Recursive real function integrating from a to b given a real function f
    level += 1
    h = b[-1] - a[-1]
    c = (a[0], ((a[-1] + b[-1]) / 2.0))
    d = (a[0], ((a[-1] + c[-1]) / 2.0))
    e = (a[0], ((c[-1] + b[-1]) / 2.0))

    oneSimpson = h * (f(a) + (4*f(c)) + f(b)) / 6.0
    twoSimpson = h * (f(a) + (4 * f(d)) + (2 * f(c)) + (4 * f(e)) + f(b)) / 12.0

    if level >= level_max:
        simpsonResult = twoSimpson
    else:
        if (twoSimpson - oneSimpson) < (15 * epsilon):
            simpsonResult = twoSimpson + ((twoSimpson - oneSimpson) / 15.0)
        else:
            leftSimpson = simpson(f, a, c, epsilon/2.0, level, level_max)
            rightSimpson = simpson(f, c, b, epsilon/2.0, level, level_max)
            simpsonResult = rightSimpson + leftSimpson

    return simpsonResult

# Velocity equations for cart state (equation 13.15)
def theta_velocity(t):
    u_s, u_phi, L = t
    return (u_s / L) * math.tan(u_phi)

def y_velocity(t):
    u_s, theta = t
    return u_s * math.sin(theta)

def x_velocity(t):
    u_s, theta = t
    return u_s * math.cos(theta)

class DubinsCar:
    def __init__(self, v, input_range, dt=1/30):
        self.state = np.zeros(3)
        # Set pointers to state indices
        self.x = self.state[:1]
        self.y = self.state[1:2]
        self.psi = self.state[2:3]
        self.v = v
        self.dt = dt
        self.umin = input_range[0]
        self.umax = input_range[1]

    def step(self, u):
        u = np.clip(u, self.umin, self.umax)
        self.x += self.v * np.cos(self.psi) * self.dt
        self.y += self.v * np.sin(self.psi) * self.dt
        self.psi += u*self.dt


def dubins_car_simulation(wheelBase = 1.0):
    dubinsCarModel = DubinsCar(wheelBase)
    u_s = 1.0
    u_phi = 0.5

    #simpleCar.print_current_state()

    for i in range(100):
        dubinsCarModel.step(u_s, u_phi)
        #simpleCar.print_current_state()

    plt.plot(dubinsCarModel.x, dubinsCarModel.y, 'o', color = 'black')
    plt.show()



if __name__ == "__main__":
    simple_car_simulation()
