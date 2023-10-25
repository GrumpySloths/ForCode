import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import struct

width = 5
height = 0.05

nrows = 100
ncols = 100
terrain = np.zeros((nrows, ncols))
start = 6
end = 12

for i in range(start, end):
    terrain[width * i:width * (i + 1), :50] = height
    height += 0.05

terrain_data = terrain.ravel()

with open("stairs.bin", "wb") as file:
    file.write(struct.pack("i", nrows))
    file.write(struct.pack("i", ncols))
    for value in terrain_data:
        file.write(struct.pack("f", value))

# terrain[:10, :10] = 2
# plt.contour(terrain, levels=24)
# plt.contourf(terrain,
#              levels=end - start,
#              cmap='gray',
#              )
# plt.colorbar()
# plt.savefig("stairs.png")
