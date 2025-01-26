import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def create_cube(center, size):
    """
    Generate vertices for a cube.
    Args:
        center: Center of the cube (x, y, z).
        size: Length of the cube's sides.
    Returns:
        Array of vertices for the cube.
    """
    half_size = size / 2
    offsets = [-half_size, half_size]
    vertices = np.array([[center[0] + x, center[1] + y, center[2] + z]
                         for x in offsets for y in offsets for z in offsets])
    return vertices

def add_faces(ax, vertices, color, alpha=0.5):
    """
    Add faces to a cube for visualization.
    """
    faces = [
        [vertices[0], vertices[1], vertices[3], vertices[2]],  # Front
        [vertices[4], vertices[5], vertices[7], vertices[6]],  # Back
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Bottom
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Top
        [vertices[0], vertices[2], vertices[6], vertices[4]],  # Left
        [vertices[1], vertices[3], vertices[7], vertices[5]]   # Right
    ]
    for face in faces:
        face_poly = Poly3DCollection([face], color=color, alpha=alpha)
        ax.add_collection3d(face_poly)

# Outer cubes
outer1_vertices = create_cube(center=(0, 0, 0), size=2)
outer2_vertices = create_cube(center=(2, 2, 2), size=2)

# Inner shared cube (1/4 volume)
inner_vertices = create_cube(center=(1, 1, 1), size=1)

# Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Add cubes
add_faces(ax, outer1_vertices, color='blue', alpha=0.3)
add_faces(ax, outer2_vertices, color='green', alpha=0.3)
add_faces(ax, inner_vertices, color='red', alpha=0.6)

# Configure plot
ax.set_xlim(-1, 3)
ax.set_ylim(-1, 3)
ax.set_zlim(-1, 3)
ax.set_title("Holographic Projection: 9-Faced Inner Cube and Transparent Outer Cube")
plt.show()
