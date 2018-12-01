import numpy as np

a = -0.3
thresh = 0.000001
old_a = 0
while True:
    old_a = a
    a = 0.99*a + 0.01
    if abs(old_a - a) < thresh:
        break

print (np.round(a, decimals=3))

a = -0.5
thresh = 0.001
old_a = 0
while True:
    old_a = a
    a = 0.99*a - 0.01
    if abs(old_a - a) < thresh:
        break

# print (np.round(a, decimals=3))

a = 0.6
b = 0.8
old_a = 0
old_b = 0
while True:
    old_a = a
    old_b = b
    a = 0.99*a + 0.01*b
    b = 0.99*b
    # print (a,b)
    if abs(old_a - a) < thresh and abs(old_b -b) < thresh:
        break

# print (np.round(a, decimals=3), np.round(b, decimals=3))

q_vals = np.array([0.6, -0.3, -0.5, 0.8])
itr = 0
while True:
    old_q = q_vals.copy()
    i, j = np.argmax(q_vals), np.max(q_vals)
    if i == 3:
        q_vals[i] = 0.99*q_vals[i]
    if i == 2:
        q_vals[i] = 0.99*q_vals[i] - 0.01
    if i == 1:
        q_vals[i] = 0.99*q_vals[i] - 0.01
    if i == 0:
        q_vals[i] = 0.99*q_vals[i] + 0.01*q_vals[3]

    print (itr, q_vals, old_q)
    itr += 1
    delta = [False, False, False, False]
    for l, (x,y) in enumerate(zip(q_vals, old_q)):
        print (l, x, y)
        if abs(x-y) < thresh:
            delta[l] = True

    print (delta)
    if all(delta):
        break

print (np.round(q_vals, decimals=3))