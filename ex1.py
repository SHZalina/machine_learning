import numpy as np
import matplotlib.pyplot as plt

def plan_matrix(M0, N0, x0):
  F0 = np.ones((N0,1))
  x0 = np.array((x0))[:, np.newaxis]
  for i in range(M0):
    F0 = np.concatenate((F0, x0), axis=1)
  powers = range(M0 + 1)
  F0 = np.power(F0, np.array([powers]))
  return F0

def func_w(F0, t0):
  t0 = np.array((t0))[:, np.newaxis]
  F0_t = np.transpose(F0)
  #W = np.dot(F0_t, F0)
  #W = np.dot(W, F0_t)
  #W = np.dot(W, t)
  #print(W)
  W = np.dot(np.dot(np.linalg.inv(np.dot(F0_t, F0)), F0_t), t0) #формулы все с файла с теорией
  return W

N = 1000
x = np.linspace(0, 1, N)
print(x)
z = 20*np.sin(2*np.pi * 3 * x) + 100*np.exp(x)
error = 10 * np.random.randn(N)
t = z + error
#print(x)

plt.figure() # новая фигура/окно
plt.plot(x, z, label="z=f(x)")
plt.legend(loc="upper left") # где выводить label="z=f(x)"

plt.scatter(x, t, s=1, color=(1, 0, 0, 0.5), label=(' t=f(x) scatter'))
plt.legend(loc="upper left")

plt.show()

M1 = 1
M2 = 8
M3 = 100

F1 = plan_matrix(M1, N, x)
w1 = func_w(F1,t)
F2 = plan_matrix(M2, N, x)
w2 = func_w(F2,t)
F3 = plan_matrix(M3, N, x)
w3 = func_w(F3,t)

y1 = np.dot(F1, w1)
y2 = np.dot(F2, w2)
y3 = np.dot(F3, w3)
 
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Horizontally stacked subplots')

ax1.plot(x, y1, x, z,'g') 
ax1.scatter(x, t, s=1, color=(1, 0, 0, 0.5), label=(' t=f(x) scatter'))
ax1.set_title(f"M = {M1}")
ax2.plot(x, y2, x, z, 'g') 
ax2.scatter(x, t, s=1, color=(1, 0, 0, 0.5), label=(' t=f(x) scatter'))
ax2.set_title(f"M = {M2}")
ax3.plot(x, y3, x, z, 'g') 
ax3.scatter(x, t, s=1, color=(1, 0, 0, 0.5), label=(' t=f(x) scatter'))
ax3.set_title(f"M = {M3}")
plt.show()

plt.plot(x, y3, label="y3=f(x)")
plt.show()


err = []
Mn = 100
for i in range (1, Mn+1):
  F = plan_matrix(i, N, x)
  w = func_w(F, t)
  y = np.dot(F, w)
  
  err0 = 0
  for i in range(N):
    err0 += ((t[i] - np.transpose(w) @ F[i]) ** 2)
  err.append(err0/2)

plt.xlabel("M")
plt.ylabel("Error")
plt.title("E(w)")
plt.plot(range(1, Mn+1), err) 
plt.show()


