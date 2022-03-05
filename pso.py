import random
import numpy as np


def equation(x, y):
    return (x - 3.14) ** 2 + (y - 2.72) ** 2 + np.sin(3 * x + 1.41) + np.sin(4 * y - 1.73)


global_best = None
c = 0.5
W = 0.5
class Particle:
    def __init__(self, position):
        self.position = position
        self.velocity = [random.random() for i in range(len(position))]
        self.pbest = position
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
        particles.append(Particle([random.randint(0, 15), random.randint(0, 15)]))
        print(particles[i].velocity)
    particles = np.array(particles)
    for particle in particles:
        print(particle.position)
        particle.move()
        print(particle.position)
