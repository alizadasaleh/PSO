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

def schwefel(*args):
    answer = 418.9829*len(args)
    for arg in args:

        answer += arg * sin(sqrt(abs(arg)))
    return answer
    # - x * sin( sqrt( abs( x )))-y*sin(sqrt(abs(y)))

def ackley(*args):
    first_part, second_part = 0, 0
    for arg in args:
        first_part += arg**2
        second_part += cos(2*pi*arg)
    answer = -20*exp(-0.2*sqrt(1/len(args)*first_part))-exp(1/len(args)*second_part)+20+exp(1)
    return answer

def rosenbrock(*args):
    answer = 0
    for i in range(len(args)-1):
        answer += 100* (args[i+1]- args[i]**2 )**2 +(args[i]-1)**2
    return answer


def rastrigin(*args):
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
        self.velocity = W*self.velocity + c * random.uniform(0,0.5) * self.pb_pc_difference() + c * random.uniform(0,0.5) * self.gb_pc_difference()
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

def standart_deviation(*args):
    sum = 0
    for arg in args:
        sum += (arg-mean())**2
    answer = sqrt(sum/len(args))
    return answer
def show_points(particles):
    scat = []
    for i in particles:
        scat.append(ax.scatter(i.position[0], i.position[1], i.z_pos(), s=5, c='black'))


def ackley_run():
    global global_best
    global global_best_z
    global c
    global W
    global equation
    equation = ackley
    global_best = [0,0]
    global_best_z = 0
    c = 0.9
    W = 0.8
    n_particles = 30
    particles = []
    for i in range(n_particles):
        particles.append(Particle( [random.uniform(-32.768,32.768),random.uniform(-32.768,32.768)] ))

    for x in range(30):
        for particle in particles:
            particle.move()
    show_points(particles)

    x_data = np.arange(-32,32,0.5)
    y_data = np.arange(-32,32,0.5)
    X, Y = np.meshgrid(x_data, y_data)
    Z = equation(X,Y)
    ax.plot_surface(X, Y, Z, alpha=0.5)
    plt.show()

def rosenbrock_run():
    global equation
    global global_best
    global global_best_z
    global c
    global W
    c = 0.1
    W = 0.8
    equation = rosenbrock
    global_best = [1,1]
    global_best_z = 0
    n_particles = 30
    particles = []
    for i in range(n_particles):
        particles.append(Particle( [random.uniform(-5,10),random.uniform(-5,10)] ))

    for x in range(30):
        for particle in particles:
            particle.move()
    show_points(particles)

    x_data = np.arange(-5,10,0.2)
    y_data = np.arange(-5,10,0.2)
    X, Y = np.meshgrid(x_data, y_data)
    Z = equation(X,Y)
    ax.plot_surface(X, Y, Z, alpha=0.5)
    plt.show()

def schewefel_run():
    global equation
    global global_best
    global global_best_z
    global c
    global W
    c = 0.1
    W = 0.8
    equation = schwefel
    global_best = [420.9687,420.9687]
    global_best_z = 0
    n_particles = 30
    particles = []
    for i in range(n_particles):
        particles.append(Particle( [random.uniform(-500,500),random.uniform(-500,500)]))

    for x in range(30):
        for particle in particles:
            particle.move()
    show_points(particles)

    x_data = np.arange(-500,500,20)
    y_data = np.arange(-500,500,20)
    X, Y = np.meshgrid(x_data, y_data)
    Z = equation(X,Y)
    ax.plot_surface(X, Y, Z, alpha=0.5)
    plt.show()
    
def rastrigin_run():
    global equation
    global global_best
    global global_best_z
    global c
    global W
    c = 0.1
    W = 0.8
    equation = rastrigin
    global_best = [0,0]
    global_best_z = 0
    n_particles = 30
    particles = []
    for i in range(n_particles):
        particles.append(Particle( [random.uniform(-5.12, 5.12),random.uniform(-5.12, 5.12)]))

    for x in range(30):
        for particle in particles:
            particle.move()
    show_points(particles)

    x_data = np.arange(-5.12, 5.12,1)
    y_data = np.arange(-5.12, 5.12,1)
    X, Y = np.meshgrid(x_data, y_data)
    Z = equation(X,Y)
    ax.plot_surface(X, Y, Z, alpha=0.5)
    plt.show()
    
if __name__ == "__main__": 
    rosenbrock_run()



