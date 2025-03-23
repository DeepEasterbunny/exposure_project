import numpy as np

def generate_euler_angles(n):
    """
    Generates `n` evenly spaced Euler angles within a reasonable range.
    """
    phi1_values = np.linspace(0, 360, 100, endpoint=False)  # First Euler angle (0 to 360)
    phi_values = np.linspace(0, 180, 100, endpoint=False)   # Second Euler angle (0 to 180)
    phi2_values = np.linspace(0, 360, 100, endpoint=False)  # Third Euler angle (0 to 360)

    comb_array = np.array( 
    np.meshgrid(phi1_values, phi_values, phi2_values)).T.reshape(-1, 3) 
    
    # Stack into (n, 3) array
    return comb_array#np.column_stack((phi1_values, phi_values, phi2_values))

def save_euler_angles(filename, n):
    """
    Saves the generated Euler angles in the required file format.
    """
    euler_angles = generate_euler_angles(n)

    # Prepare header
    header = f"eu\n{n}\n"

    # Format the angles as comma-separated values without spaces
    data_lines = "\n".join([",".join(f"{angle:.6f}" for angle in angles) for angles in euler_angles])

    # Combine everything into one output string
    output_text = header + data_lines

    # Write to file
    with open(filename, "w") as f:
        f.write(output_text)

    print(f"File saved as {filename}")

N = 100000
save_euler_angles("/work3/s203768/EMSoftData/angle/evenly_spaced_angles.txt", N)