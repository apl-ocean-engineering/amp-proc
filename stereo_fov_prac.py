import numpy as np

y = 10
t = 0.25
blind_spot = 0.07
theta1 = np.radians(64.4)
theta2 = np.radians(64.4)

x1 = -y/np.tan(np.radians(90) - theta1/2)
x2 = y/np.tan(np.radians(90) - theta2/2) + t

v1 = np.array([x1,y])
v2 = np.array([x2, y])

theta = np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*(np.linalg.norm(v2))))

print(np.degrees(theta))

# SV = 2*np.tan(theta1/2)*y - t
# theta = 2*np.arctan(SV/(2*(y - blind_spot)))
#
# print(SV, np.degrees(theta))
