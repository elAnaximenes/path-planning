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

# a simple car model with three degrees of freedom. Turning front axle and fixed rear axle
class Car:

    def __init__(self, wheelBase):
        self.x = [0.0]
        self.y = [0.0]
        self.theta = [0.0]
        # wheel base describes the distance between center of front and rear axles(L in figure 13.1)
        self.wheelBase = wheelBase

    # integrate over time step and update car with new states
    def step(self, u_s, u_phi, stepSize = 0.5):
        # current state parameters
        xStartConfig = (u_s, self.theta[-1])
        yStartConfig = (u_s, self.theta[-1])
        thetaStartConfig = (u_s, u_phi, self.wheelBase)

        # get angle between center of car's rear axle and x-axis
        # in order to compute next state
        thetaEnd = self.theta[-1] + stepSize

        # next state parameters
        xEndConfig = (u_s, thetaEnd)
        yEndConfig = (u_s, thetaEnd)
        thetaEndConfig = thetaStartConfig

        # store new state
        self.x.append(simpson(x_velocity, xStartConfig, xEndConfig))
        self.y.append(simpson(y_velocity, yStartConfig, yEndConfig))
        self.theta.append(thetaEnd)

    # print state info to standard out
    def print_current_state(self):
        print("x: " + str(self.x[-1]))
        print("y: " + str(self.y[-1]))
        print("theta: " + str(self.theta[-1]) + "\n")

def simple_car_simulation(wheelBase = 1.0):
    simpleCar = Car(wheelBase)
    u_s = 1.0
    u_phi = 0.5

    #simpleCar.print_current_state()

    for i in range(100):
        simpleCar.step(u_s, u_phi)
        #simpleCar.print_current_state()

    plt.plot(simpleCar.x, simpleCar.y, 'o', color = 'black')
    plt.show()

if __name__ == "__main__":
    simple_car_simulation()
