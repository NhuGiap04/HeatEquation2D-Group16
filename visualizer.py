from numpy import dtype, float64, ndarray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
# from mpl_toolkits.mplot3d import Axes3D  # No longer necessary for 3D plotting

# Load the data
data = np.loadtxt('heat_output.txt')

# Validate that each row has the same number of elements
if not np.all([len(row) == len(data[0]) for row in data]):
    raise ValueError("Inconsistent number of elements in rows.")

# Determine N
total_points = data.shape[1]
N = int(np.sqrt(total_points))
if N * N != total_points:
    raise ValueError("Each row must contain N^2 elements for some integer N.")

# Reshape data into (time_steps, N, N)
time_steps = data.shape[0]
heat_data: ndarray[tuple[int, int, int], dtype[float64]] = data.reshape((time_steps, N, N))

# Create a progress bar with the total number of frames
progress_bar = tqdm(total=time_steps, desc="Animating Heat Distribution", unit="frame")

# Define the progress callback function
def update_progress(current_frame, total_frames): # pyright: ignore[reportUnusedParameter, reportMissingParameterType]
    progress_bar.update(1)
    

x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots(figsize=(6, 6))

cax = ax.imshow(heat_data[0], origin='lower', extent=(0, 1, 0, 1),
                cmap='jet', vmin=0, vmax=100)
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('°C', rotation=0, labelpad=15)

title = ax.set_title("Heat Distribution at t=0")

# Optional: Add boundary condition annotations
# ax.text(-0.5, 2, r"$u(0,y,t)=100\sqrt{y/4}$", color='navy', fontsize=12, rotation=90, va='center')
# ax.text(2, 4.2, r"$u(x,4,t)=100(0.7+0.3\sin{\frac{5\pi x}{4}})$", color='navy', fontsize=12, ha='center')
# ax.text(4.1, 2, r"$u(4,y,t)=0$", color='navy', fontsize=12, rotation=270, va='center')
# ax.text(2, -0.5, r"$u(x,0,t)=100\sqrt[3]{x/4}$", color='navy', fontsize=12, ha='center')
ax.set_xlabel('x')
ax.set_ylabel('y')

def update(frame:int):
    cax.set_data(heat_data[frame])
    title.set_text(f"Heat Distribution at t={frame}")
    return cax, title

ani = animation.FuncAnimation(fig, update, frames=time_steps, interval=200)

# Save animation to MP4
ani.save('heat_distribution.mp4', writer='ffmpeg', fps=120, progress_callback=update_progress)

progress_bar.close()
plt.close()  # Optional: Prevents final static display when running in notebooks