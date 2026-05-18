import numpy as np
import matplotlib.pyplot as plt

# Load the compressed archive
# (Replace with your actual filename if it's different)
filename = 'dataset_chunk_0000.npz'
archive = np.load(filename)

# See what arrays are stored inside
print(f"--- Inspecting {filename} ---")
print("Keys in this file:", archive.files)
# Expected output: ['images', 'meta']

# Extract the arrays
images = archive['images']
meta_data = archive['meta']

# Check if it is a perfect square
print("\n--- Shape Verification ---")
print(f"Images tensor shape: {images.shape}")
print(f"Meta tensor shape:   {meta_data.shape}")

if images.shape[1] == 113 and images.shape[2] == 113:
    print("SUCCESS: The images are perfectly 113x113 square!")
else:
    print(f"ERROR: Expected 113x113, but got {images.shape[1]}x{images.shape[2]}")

# Outputs the values (manually look if they're not absurd)
print("\n--- Data Value Checks ---")
print(f"First event Run/Event ID:  {meta_data[0][0]:.0f}, {meta_data[0][1]:.0f}")
print(f"First event Total Energy:  {meta_data[0][2]:.2f} MeV")
print(f"Max pixel value in chunk:  {images.max():.4f}")
print(f"Min pixel value in chunk:  {images.min():.4f}")

# Visual proof
print("\n--- Generating Visual Proof ---")
print("Close the plot window to exit the script.")

# Grab the first event's image
first_image = images[0]

# Plot it
fig, ax = plt.subplots(figsize=(8, 8))
# We use origin='lower' so the CAL stays at the bottom and TKR is at the top
mesh = ax.imshow(first_image, cmap='plasma', origin='lower', vmin=0, vmax=1)
fig.colorbar(mesh, ax=ax, label='Normalized Log10(Energy)', shrink=0.8)

ax.set_title(f'Unified LAT Tensor (Run {meta_data[0][0]:.0f}, Event {meta_data[0][1]:.0f})')
ax.set_xlabel('LAT Width (113 Pixels)')
ax.set_ylabel('LAT Height (113 Pixels)')

plt.tight_layout()
plt.show()

archive.close()