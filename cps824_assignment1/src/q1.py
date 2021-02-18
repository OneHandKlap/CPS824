import numpy as np

# state probabilities

p=[[.75, .25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[.25, .5, .25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, .25, .5, .25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],
[0, 0, .25, .5, 0, 0, 0, .25, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, .25, .25, .25, 0, 0, .25, 0, 0, 0, 0, 0,0],
[0, 0, 0, 0, 0, 0, .25, .25, .25, 0, 0, .25, 0, 0, 0, 0],
[0, 0, 0, .25, 0, 0, .25,.25, 0, 0, 0, .25, 0, 0, 0, 0],
[0, 0, 0, 0, .25, 0, 0, 0, .25, .25, 0, 0, .25, 0, 0, 0],
[0, 0, 0, 0, 0, .25, 0, 0, .25, .25, .25, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, .25, 0, 0, .25, .25, .25, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, .25, 0, 0, 0, .5, .25, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .25, .5, .25, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .25, .5, .25],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .25, .75]]


# reward values
r1 = np.array([-1,-1,-1,-1,-5,-1,-1,-1,-1,-1,-1,5,-1,-1,-1,-1]).astype(float)
r2 = np.array([0,0,0,0,-5,0,0,0,0,0,0,5,0,0,0,0]).astype(float)
r3 = np.array([1,1,1,1,-5,1,1,1,1,1,1,5,1,1,1,1]).astype(float)
r4 =  r1+2
r5= r1+3
r6=np.array([-3,-3,-3,-3,-5,-3,-3,-3,-3,-3,-3,5,-3,-3,-3,-3]).astype(float)

# discount
y = 1

#Question E new_gamma
new_y=.7

def calcValues(p, r, y):
    return np.linalg.inv(np.identity(r.shape[0])-np.multiply(y,p)).dot(r)

print("\nr="+str(r1))
print(calcValues(p, r1, y).reshape((4,4)))
print("\nr="+str(r2))
print(calcValues(p, r2, y).reshape((4,4)))
print("\nr="+str(r3))
print(calcValues(p, r3, y).reshape((4,4)))
print("Question D")
print("\nr="+str(r5))
print(calcValues(p, r5, y).reshape((4,4)))
print("Question E")
print("\nr="+str(r5))
print(calcValues(p, r5, new_y).reshape((4,4)))

print("Question F")
print("\nr="+str(r6))
print(calcValues(p, r6, y).reshape((4,4)))
