from numpy import sqrt
from numpy import sin
def schwefel( x,y):
    return 418.9829*2 - x * sin( sqrt( abs( x )))-y*sin(sqrt(abs(y)))