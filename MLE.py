import numpy as np
import matplotlib.pyplot as plt
#part 7a 1
theta=1
x = np.linspace(0, 10, 100)
y = theta*np.exp(-theta*x)
plt.plot(x, y,'-b')
plt.title("P(X|theta) vx X")
plt.legend(["theta=1"])
plt.xlabel("X")
plt.ylabel("P(X|theta)")
#plt.savefig('p_vs_x.png')
plt.show()


#part 7a 2
x=2
theta = np.linspace(0, 5, 100)
y = theta*np.exp(-x*theta)
plt.plot(theta, y,'-b')
plt.title("P(X|theta) vx theta")
plt.legend(["X=2"])
plt.xlabel("theta")
plt.ylabel("P(X|theta)")
#plt.savefig('p_vs_theta.png')
plt.show()

#part 7c
theta=1
x = np.linspace(0, 10, 100)
y = theta*np.exp(-theta*x)
plt.plot(x, y,'-b')
plt.plot(np.ones((100,1)), y,'-r')
plt.title("P(X|theta) vx X")
plt.legend(["theta=1","MLE"])
plt.xlabel("X")
plt.ylabel("P(X|theta)")
#plt.savefig('p_vs_x_MLe.png')
plt.show()