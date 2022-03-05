from numpy import sin
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy import exp,pi,sqrt,cos,e,pi

global_best = [0, 0]
global_best_z = 0
c = 0.9
W = 0.8

ax = plt.axes(projection='3d')


def equation(x, y):
    return 418.9829*2 - x * sin( sqrt( abs( x )))-y*sin(sqrt(abs(y)))


class Particle:
    def __init__(self, position):
        self.position = position
        self.velocity = np.array([random.uniform(-1,1) for i in range(len(position))])
        global global_best_z
        global global_best
        x,y = position

        self.current_z = equation(x,y)
        self.pbest = position
        self.pbest_z = self.current_z
        if global_best_z > self.current_z:
            global_best_z = self.current_z
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
        self.velocity = W*self.velocity + c * random.uniform(0, 0.5) * self.pb_pc_difference() + c * random.uniform(0, 0.5) * self.gb_pc_difference()
    def move(self):

        self.position = np.sum([self.position,self.velocity],axis=0)
        x,y = self.position
        global global_best_z
        global global_best
        current_z = equation(x,y)
        if global_best_z > current_z:
            global_best_z = current_z
            global_best = self.position
        if  self.pbest_z > self.z_pos():
            self.pbest = self.position
            self.pbest_z = current_z
        self.updateVelocity()

def show_points(particles):
    scat = []
    for i in particles:
        scat.append(ax.scatter(i.position[0], i.position[1], i.z_pos(), s=5, c='black'))

if __name__ == "__main__":
    n_particles = 30
    particles = []
    for i in range(n_particles):
        particles.append(Particle( [random.uniform(-32.768,32.768),random.uniform(-32.768,32.768)] ))

    for x in range(30):
        for particle in particles:
            particle.move()
    show_points(particles)

    x_data = np.arange(-500, 500, 0.5)
    y_data = np.arange(-500, 500, 0.5)
    X, Y = np.meshgrid(x_data, y_data)
    Z = equation(X, Y)
    ax.plot_surface(X, Y, Z, alpha=0.5)
    plt.show()