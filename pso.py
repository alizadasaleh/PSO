import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation

from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi

global_best = [0,0]
global_best_z = 0
c = 0.25
W = 0.01



ax = plt.axes(projection='3d')


def equation(x, y):
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2)))-exp(0.5 * (cos(2 *
                                                                                pi * x)+cos(2 * pi * y))) + e + 20



class Particle:
    def __init__(self, position):
        self.position = position
        self.velocity = [random.uniform(-1,1) for i in range(len(position))]
        global global_best_z
        global global_best
        x,y = position
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
        return np.sum([self.pbest, - self.position], axis=0)
    def gb_pc_difference(self):
        return np.sum([global_best,-self.position], axis=0)

    def setVelocity(self, vel):
        self.velocity = vel
    
    def updateVelocity(self):
        self.velocity = W*self.velocity + c * random.uniform(-1,1) * self.pb_pc_difference() + c * random.uniform(-1,1) * self.gb_pc_difference()
    
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

def show_points(particles):
    scat = []
    for i in particles:
        scat.append(ax.scatter(i.position[0], i.position[1], i.z_pos()))
    plt.pause(0.1)


if __name__ == "__main__":
    n_particles = 5
    particles = []
    for i in range(n_particles):
        particles.append(Particle( [random.uniform(-32.768,32.768),random.uniform(-32.768,32.768)] ))





    for x in range(30):
        show_points(particles)

        for particle in particles:
            particle.move()


    x_data = np.arange(-32,32,0.5)
    y_data = np.arange(-32,32,0.5)
    X, Y = np.meshgrid(x_data, y_data)
    Z = equation(X,Y)
    ax.plot_surface(X, Y, Z)
    plt.show()
