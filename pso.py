import numpy as np
import matplotlib.pyplot as plt
import random
from numpy import exp,pi,sqrt,cos,e,pi,sin
from matplotlib import pyplot as plt


global global_best
global global_best_z
global c1 
global c2
global W 
global equation
ax = plt.axes(projection='3d')

def schwefel(args):
    answer = +418.9829*len(args)
    for arg in args:

        answer -= arg * sin(sqrt(abs(arg)))
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
        self.velocity = W*self.velocity + c1 * random.uniform(0,0.5) * self.pb_pc_difference() + c2 * random.uniform(0,0.5) * self.gb_pc_difference()
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

def mean(particles):
    answer = np.zeros(len(particles[0].position))
    for particle in particles:
        answer = np.add(particle.position,answer)

    answer = np.multiply(answer,1/len(particles))
    return answer

def standart_deviation(particles):
    mean_var = mean(particles)
    answer = 0
    for particle in particles:
        test =  np.add(-mean_var,particle.position)
        answer += np.multiply(test,test)

    answer = np.multiply(answer,len(particles)-1)
    answer = np.array([sqrt(x) for x in  answer])
    return answer
def show_points(particles):
    scat = []
    for i in particles:
        scat.append(ax.scatter(i.position[0], i.position[1], i.z_pos(), s=5, c='black'))


def ackley_run(dimension):
    global global_best
    global global_best_z
    global c1
    global c2
    global W
    global equation
    global d
    equation = ackley
    global_best_z = 0
    c1 = 0.1
    c2 = 0.1

    W = 0.8
    d = dimension - 1
    global_best = d*[0]
    n_particles = 30
    particles = []
    for i in range(n_particles):
        particles.append(Particle( d*[random.uniform(-32.768,32.768)] ))

    for x in range(30):
        for particle in particles:
            particle.move()
    show_points(particles)
    if dimension == 3:
        x_data = np.arange(-32,32,0.5)
        y_data = np.arange(-32,32,0.5)
        X, Y = np.meshgrid(x_data, y_data)
        Z = equation([X,Y])
        ax.plot_surface(X, Y, Z, alpha=0.5)
        print([particle.position for particle in particles[:2]],standart_deviation(particles[:2]))
        ax.text2D(-0.30, 0.95, f"Mean: {mean(particles)}   Standart Deviation : {standart_deviation(particles)}", transform=ax.transAxes)

        plt.show()
        plt.savefig(f"figures/ackley/fig{random.randint(0,1000)}.png")

    if d > 3:
        print([particle.position for particle in particles])

def rosenbrock_run(dimension):
    global equation
    global global_best
    global global_best_z
    global c1
    global c2
    global W
    global d
    d = dimension - 1
    c1 = 0.1
    c2 = 0.1
    W = 0.8
    
    equation = rosenbrock
    global_best = d*[1]
    global_best_z = 0
    n_particles = 30
    particles = []
    for i in range(n_particles):
        particles.append(Particle( d*[random.uniform(-5,10)] ))

    for x in range(30):
        for particle in particles:
            particle.move()
    if dimension == 3:
        show_points(particles)

        x_data = np.arange(-5,10,0.2)
        y_data = np.arange(-5,10,0.2)
        X, Y = np.meshgrid(x_data, y_data)
        Z = equation([X,Y])
        ax.plot_surface(X, Y, Z, alpha=0.5)
        ax.text2D(-0.30, 0.95, f"Mean: {mean(particles)}   Standart Deviation : {standart_deviation(particles)}", transform=ax.transAxes)
        plt.savefig(f"figures/rosenbrock/fig{random.randint(0,1000)}.png")

        plt.show()
    if d > 3:
        print([particle.position for particle in particles])

def schewefel_run(dimension):
    global equation
    global global_best
    global global_best_z
    global c1
    global c2
    global W
    global d
    c1 = 0.1
    c2 = 0.2
    W = 0.8
    d = dimension - 1
    equation = schwefel
    global_best = d * [420.9687]
    global_best_z = 0
    n_particles = 30
    particles = []
    for i in range(n_particles):
        particles.append(Particle( d*[random.uniform(-500,500)]))

    for x in range(30):
        for particle in particles:
            particle.move()
    if dimension == 3:
        show_points(particles)
        x_data = np.arange(-500,500,20)
        y_data = np.arange(-500,500,20)
        X, Y = np.meshgrid(x_data, y_data)
        Z = equation([X,Y])
        ax.plot_surface(X, Y, Z, alpha=0.5)
        ax.text2D(-0.30, 0.95, f"Mean: {mean(particles)}   Standart Deviation : {standart_deviation(particles)}", transform=ax.transAxes)
        plt.savefig(f"figures/schewefel/fig{random.randint(0,1000)}.png")

        plt.show()
    if d > 3:
        print([particle.current_z for particle in particles])
    
def rastrigin_run(dimension):
    global equation
    global global_best
    global global_best_z
    global c1
    global c2
    global W
    global d
    c1 = 0.1
    c2 = 0.2
    W = 0.8
    d = dimension - 1
    equation = rastrigin
    global_best = d*[0]
    global_best_z = 0
    n_particles = 30
    particles = []
    for i in range(n_particles):
        particles.append(Particle( d*[random.uniform(-5.12, 5.12)] ))

    for x in range(30):
        for particle in particles:
            particle.move()
    if dimension == 3:
        show_points(particles)
        x_data = np.arange(-5.12, 5.12,0.1)
        y_data = np.arange(-5.12, 5.12,0.1)
        X, Y = np.meshgrid(x_data, y_data)
        Z = equation([X,Y])
        ax.plot_surface(X, Y, Z, alpha=0.5)
        ax.text2D(-0.30, 0.95, f"Mean: {mean(particles)}   Standart Deviation : {standart_deviation(particles)}", transform=ax.transAxes)
        plt.savefig(f"figures/rastrigin/fig{random.randint(0,1000)}.png")

        plt.show()
        
    if d > 3:
        print([particle.position for particle in particles])



if __name__ == "__main__": 
    schewefel_run(3)




