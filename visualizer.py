from doctest import debug
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
assert time_steps == 20000, "Expected 25000 time steps in the data."
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
# ax.text(0.5, 1.02, 
#          (r"$u(0,y,t)=100\sqrt{y}$\n"
#           r"$u(x,1,t)=100(0.7+0.3\sin{5\pi x})$\n"
#           r"$u(1,y,t)=0$\n"
#           r"$u(x,0,t)=100\sqrt[3]{x}$"),
#          ha='center', va='bottom', fontsize=11, color='navy')

ax.set_xlabel('x')
ax.set_ylabel('y')

debug_steps = list(range(1, time_steps, 500))
debug_steps.append(time_steps - 1)  # Ensure the last frame is included in debug steps

def update(frame:int):
    cax.set_data(heat_data[frame])
    title.set_text(f"Heat Distribution at t={frame}")
    if frame in debug_steps:
        plt.savefig(f"output_frames/frame_{frame}.png", dpi=300)
    return cax, title

ani = animation.FuncAnimation(fig, update, frames=time_steps, interval=200)

# Save animation to MP4
ani.save('heat_distribution.mp4', writer='ffmpeg', fps=300, progress_callback=update_progress)

progress_bar.close()
plt.close()  # Optional: Prevents final static display when running in notebooks