import trimesh
import numpy as np
import vedo
from vedo import Plotter

# Load F1 car mesh
mesh = trimesh.load('f1_2026_v58.stl')
vmesh = vedo.Mesh(mesh).c('silver').alpha(0.8).lighting('glossy')

# Number of flow lines
n = 60  # Reduced for better performance

# Create starting points in front of the car
y = np.linspace(-50, 50, n)
z = np.linspace(-15, 25, n)
starts0 = np.column_stack([
    np.full(n, -200),  # x-coordinate
    y,
    z
])

# Create Plotter
plt = Plotter(bg='black', title='F1 Aerodynamic Flow', size=(1200, 800), interactive=False)

# Store visualization objects
flow_objects = []

def calculate_flow(points, time):
    """Calculate smooth aerodynamic flow"""
    flow = np.zeros_like(points)
    
    # Main flow in x-direction
    flow[:,0] = 30  # Base speed
    
    # Side spread effect
    flow[:,1] = np.sin(points[:,1]/50 * np.pi) * 8
    
    # Vertical flow based on height
    flow[:,2] = np.where(points[:,2] > 0, 
                        -points[:,2]*0.2,  # Down for upper flow
                        -points[:,2]*0.4)   # Up for lower flow
    
    # Gentle time-based variation
    flow[:,1] += np.sin(time * 0.2) * 3
    flow[:,2] += np.cos(time * 0.15) * 2
    
    return flow

for frame in range(100):  # Reduced number of frames
    time = frame * 0.1
    
    # Clear previous objects
    for obj in flow_objects:
        plt.remove(obj)
    flow_objects.clear()
    
    # Calculate flow
    flow = calculate_flow(starts0, time)
    starts = starts0.copy()
    ends = starts + flow
    
    # Create flow lines
    for i in range(n):
        points = np.array([starts[i], ends[i]])
        line = vedo.Line(points, lw=3, c='coolwarm')
        line.cmap('coolwarm', points[:,1])  # Color by y-position
        flow_objects.append(line)
    
    # Add arrows
    arrows = vedo.Arrows(
        ends - flow*0.2,
        ends,
        s=0.05,
        c='cyan',
        alpha=0.8
    )
    flow_objects.append(arrows)
    
    # Show everything
    plt.show(
        vmesh, 
        *flow_objects,
        resetcam=frame==0,
    )
    plt.render()

plt.interactive().close()