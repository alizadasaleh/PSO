import numpy as np
import matplotlib.pyplot as plt
import random
from numpy import exp,pi,sqrt,cos,e,pi,sin

from pso_rastrigin import rastrigin

global global_best
global global_best_z
global c 
global W 
global equation
ax = plt.axes(projection='3d')

def schwefel(args):
    answer = 418.9829*len(args)
    for arg in args:

        answer += arg * sin(sqrt(abs(arg)))
    return answer

def ackley(args):
    first_part, second_part = 0, 0
    for arg in args:
        first_part += arg**2
        second_part += cos(2*pi*arg)
    answer = -20*exp(-0.2*sqrt(1/len(args)*first_part))-exp(1/len(args)*second_part)+20+exp(1)
    return answer

def rosenbrock(args):
    answer = 0
    for i in range(len(args)-1):
        answer += 100* (args[i+1]- args[i]**2 )**2 +(args[i]-1)**2
    return answer


def rastrigin(args):
    answer = 10*len(args)
    for arg in args:
        answer += arg**2 - 10* cos(2*pi*arg)
    return answer

class Particle:
    def __init__(self, position):
        self.position = position
        self.velocity = np.array([random.uniform(-1,1) for i in range(len(position))])
        global global_best_z
        global global_best
        self.current_z = equation(position)
        self.pbest = position
        self.pbest_z = self.current_z
        if global_best_z > self.current_z:
            global_best_z = self.current_z
            global_best = position
        
    def setVelocity(self,vel):
        self.velocity = vel

    def z_pos(self):
        return equation(self.position)
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
        global global_best_z
        global global_best
        current_z = equation(self.position)
        if global_best_z > current_z:
            global_best_z = current_z
            global_best = self.position
        if  self.pbest_z > self.z_pos():
            self.pbest = self.position
            self.pbest_z = current_z
        self.updateVelocity()

def standart_deviation(*args):
    z = rosenbrock(args)
    std_deviation = 0
    for arg in args:
        std_deviation += (arg-z)**2
    answer = sqrt(std_deviation/len(args)-1)
    return answer


def show_points(particles):
    scat = []
    if len(particles[0].position) ==3:
        for i in particles:
            scat.append(ax.scatter(i.position[0], i.position[1], i.z_pos(), s=5, c='black'))
    elif len(particles[0].position) == 2:
        for i in particles:
            scat.append(ax.scatter(i.position[0], i.position[1], 0, s=5, c='black'))
    return scat


def ackley_run(dimension, sec):
    global global_best
    global global_best_z
    global c
    global W
    global equation
    global d
    equation = ackley
    global_best_z = 0
    c = 0.9
    W = 0.8
    d = dimension
    global_best = d*[0]
    n_particles = 30
    particles = []
    for i in range(n_particles):
        particles.append(Particle( d*[random.uniform(-32.768,32.768)] ))


    if d == 3:
        x_data = np.arange(-32,32,0.5)
        y_data = np.arange(-32,32,0.5)
        X, Y = np.meshgrid(x_data, y_data)
        Z = equation([X,Y])
        ax.plot_surface(X, Y, Z, alpha=0.5)
        for x in range(30):
            scat = show_points(particles)
            plt.pause(sec)

            if x != 29:
                for i in scat:
                    i.remove()
                for particle in particles:
                    particle.move()
        plt.show()
    elif d == 2:
        for x in range(30):
            scat = show_points(particles)
            plt.pause(sec)

            if x != 29:
                for i in scat:
                    i.remove()
                for particle in particles:
                    particle.move()

        x_data = np.arange(-32, 32, 0.5)
        X = np.meshgrid(x_data)
        Y = []
        for x in X:
            Y.append(equation([x]))
        plt.plot(X[0], Y[0], 0)
        plt.show()

    if d > 3:
        print([particle.current_z for particle in particles])

def rosenbrock_run(dimension, sec):
    global equation
    global global_best
    global global_best_z
    global c
    global W
    global d
    d = dimension
    c = 0.1
    W = 0.8
    
    equation = rosenbrock
    global_best = d*[1]
    global_best_z = 0
    n_particles = 30
    particles = []
    for i in range(n_particles):
        particles.append(Particle( d*[random.uniform(-5,10)] ))

    # for x in range(30):
    #     for particle in particles:
    #         particle.move()
    if d == 3:
        x_data = np.arange(-5,10,0.2)
        y_data = np.arange(-5,10,0.2)
        X, Y = np.meshgrid(x_data, y_data)
        Z = equation([X,Y])
        ax.plot_surface(X, Y, Z, alpha=0.5)
        for x in range(30):
            scat = show_points(particles)

            plt.pause(sec)

            if x != 29:
                for i in scat:
                    i.remove()
                for particle in particles:
                    particle.move()
        plt.show()
    if d > 3:
        print([particle.position for particle in particles])


def schewefel_run(dimension, sec):
    global equation
    global global_best
    global global_best_z
    global c
    global W
    global d
    c = 0.1
    W = 0.8
    d = dimension
    equation = schwefel
    global_best = d * [420.9687]
    global_best_z = 0
    n_particles = 30
    particles = []
    for i in range(n_particles):
        particles.append(Particle( d*[random.uniform(-500,500)]))
    #
    # for x in range(30):
    #     for particle in particles:
    #         particle.move()
    if d == 3:
        # show_points(particles)
        x_data = np.arange(-500,500,20)
        y_data = np.arange(-500,500,20)
        X, Y = np.meshgrid(x_data, y_data)
        Z = equation([X,Y])
        ax.plot_surface(X, Y, Z, alpha=0.5)
        for x in range(30):
            scat = show_points(particles)
            plt.pause(sec)

            if x != 29:
                for i in scat:
                    i.remove()
                for particle in particles:
                    particle.move()
        plt.show()
    elif d == 2:
        for x in range(30):
            scat = show_points(particles)
            plt.pause(sec)

            if x != 29:
                for i in scat:
                    i.remove()
                for particle in particles:
                    particle.move()

        x_data = np.arange(-500,500,20)
        X = np.meshgrid(x_data)
        Y = []
        for x in X:
            Y.append(equation([x]))
        plt.plot(X[0], Y[0], 0)

        plt.show()
    if d > 3:
        print([particle.current_z for particle in particles])
    
def rastrigin_run(dimension, sec):
    global equation
    global global_best
    global global_best_z
    global c
    global W
    global d
    c = 0.1
    W = 0.8
    d = dimension
    equation = rastrigin
    global_best = d*[0]
    global_best_z = 0
    n_particles = 30
    particles = []
    for i in range(n_particles):
        particles.append(Particle( d*[random.uniform(-5.12, 5.12)] ))

    # for x in range(30):
        # for particle in particles:
            # particle.move()
    if d == 3:
        # show_points(particles)
        x_data = np.arange(-5.12, 5.12, 0.1)
        y_data = np.arange(-5.12, 5.12, 0.1)
        X, Y = np.meshgrid(x_data, y_data)
        Z = equation([X,Y])
        ax.plot_surface(X, Y, Z, alpha=0.5)
        for x in range(30):
            scat = show_points(particles)
            plt.pause(sec)
            if x != 29:
                for i in scat:
                    i.remove()
                for particle in particles:
                    particle.move()

        plt.show()
    elif d == 2:
        for x in range(30):
            scat = show_points(particles)
            plt.pause(sec)
            if x != 29:
                for i in scat:
                    i.remove()
                for particle in particles:
                    particle.move()

        x_data = np.arange(-5.12, 5.12, 0.1)
        X = np.meshgrid(x_data)
        Y = []
        for x in X:
            Y.append(equation([x]))
        plt.plot(X[0], Y[0], 0)
        plt.show()
    if d == 5:
        print([particle.position for particle in particles])

def mean(particles):
    answer = np.zeros(len(particles[0].position))
    for particle in particles:
        answer = np.add(particle.position,answer)
    return answer

if __name__ == "__main__": 
    schewefel_run(3, 0.1)

