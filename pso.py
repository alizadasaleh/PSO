import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation

ax = plt.axes(projection='3d')


def equation(x, y):
    return (x-3.14)**2 + (y-2.72)**2 + np.sin(3*x+1.41) + np.sin(4*y-1.73)



global_best = None


c = 0.5
W = 0.5
class Particle:
    def __init__(self, position):
        self.position = position
        self.velocity = [random.random() for i in range(len(position))]
        self.pbest = position

    def setVelocity(self,vel):
        self.velocity = vel

    def z_pos(self, position):
        x, y = position
        return equation(x, y)
    def pb_pc_difference(self):
        return np.sum([self.pbest, - self.position], axis=0)
    def gb_pc_difference(self):
        return np.sum([global_best,-self.position], axis=0)

    def setVelocity(self, vel):
        self.velocity = vel
    
    def updateVelocity(self):
        self.velocity = W*self.velocity + c * random.random() * self.pb_pc_difference() + c * random.random() * self.gb_pc_difference()
    
    def move(self):
        self.position = np.sum([self.position,self.velocity],axis=0)



if __name__ == "__main__":
    n_particles = 5
    particles = []
    for i in range(n_particles):
        particles.append(Particle( [random.randint(0,15),random.randint(0,15)] ))

    particles = np.array(particles)
    scat = []

    for i in particles:
        scat.append(ax.scatter(i.position[0], i.position[1], i.z_pos(i.position)))
    plt.pause(5)

    for particle in particles:
        particle.move()

    for i in particles:
        ax.scatter(i.position[0], i.position[1], i.z_pos(i.position))

    x_data = np.arange(0,15,0.1)
    y_data = np.arange(0,15,0.1)
    X, Y = np.meshgrid(x_data, y_data)
    Z = equation(X,Y)
    # ax.plot_surface(X, Y, Z)
    plt.show()



