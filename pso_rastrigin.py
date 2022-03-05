import numpy as np
def rastrigin(x, y):
    return (x-3.14)**2 + (y-2.72)**2 + np.sin(3*x+1.41) + np.sin(4*y-1.73)

import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation

from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi

global_best = [0, 0]
global_best_z = 0
c = 0.6
W = 0.8  # 0<W<1




ax = plt.axes(projection='3d')


def equation(x, y):
        return (x-3.14)**2 + (y-2.72)**2 + np.sin(3*x+1.41) + np.sin(4*y-1.73)


class Particle:
    def __init__(self, position):
        self.position = position
        self.velocity = np.array([random.uniform(-1, 1) for i in range(len(position))])
        global global_best_z
        global global_best
        x, y = position
        self.pbest = position
        if global_best_z > equation(x,y):
            global_best_z = equation(x,y)
            global_best = position


    def setVelocity(self,vel):
        self.velocity = vel

    def z_pos(self):
        x, y = self.position
        return equation(x, y)
    def pb_pc_difference(self):
        return np.sum([self.pbest, -self.position], axis=0)
    def gb_pc_difference(self):
        return np.sum([global_best,-self.position], axis=0)

    def setVelocity(self, vel):
        self.velocity = vel

    def updateVelocity(self):
        self.velocity = W*self.velocity + c * random.uniform(0,0.5) * self.pb_pc_difference() + c * random.uniform(0,0.5) * self.gb_pc_difference()
    def move(self):

        self.position = np.sum([self.position,self.velocity],axis=0)
        x,y = self.position
        global global_best_z
        global global_best
        if global_best_z > equation(x,y):
            global_best_z = equation(x,y)
            global_best = self.position
        x,y = self.pbest
        if  equation(x,y) > self.z_pos():
            self.pbest = self.position
        self.updateVelocity()

def show_points(particles):
    scat = []
    for i in particles:
        scat.append(ax.scatter(i.position[0], i.position[1], i.z_pos(), s=30, c='black'))


if __name__ == "__main__":
    n_particles = 30
    particles = []
    for i in range(n_particles):
        particles.append(Particle( [random.uniform(-5.12,5.12),random.uniform(-5.12,5.12)] ))

    for x in range(30):

        for particle in particles:
            particle.move()
    show_points(particles)
    x_data = np.arange(-5, 5, 0.5)
    y_data = np.arange(-5, 5, 0.5)
    X, Y = np.meshgrid(x_data, y_data)
    Z = equation(X,Y)
    ax.plot_surface(X, Y, Z, alpha=0.5)
    plt.show()
