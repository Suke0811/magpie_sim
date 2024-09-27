import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import magpylib as magpy
from stl import Mesh  # Corrected import
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import pandas as pd


class MagnetSensorSimulation:
    """
    Class to simulate the interaction between magnets and sensors using Magpylib.
    """

    def __init__(self):
        """
        Initializes the MagnetSensorSimulation class.

        Attributes:
            sensors (list): List of sensors in the simulation.
            magnets (list): List of magnets in the simulation.
        """
        self.sensors = []
        self.magnets = []

    def add_sensor(self, sensor_pixels, sensor_position, sensor_orientation=None, style_label="Sensor", stl_file=None, stl_offset=None):
        """
        Adds a sensor to the simulation, with an optional STL file for 3D model representation.

        Args:
            sensor_pixels (list): List of pixel coordinates for the sensor.
            sensor_position (list): 3D position of the sensor [x, y, z].
            sensor_orientation (scipy.spatial.transform.Rotation, optional): Orientation of the sensor as Euler angles.
                Defaults to no rotation.
            style_label (str, optional): Label for the sensor. Defaults to "Sensor".
            stl_file (str, optional): Path to the STL file representing the sensor. Defaults to None.
            stl_offset (list, optional): Offset to apply to the STL model. Defaults to None.
        """
        sensor_orientation = sensor_orientation or R.from_euler('xyz', [0, 0, 0])
        sensor = magpy.Sensor(
            pixel=sensor_pixels,
            position=sensor_position,
            orientation=sensor_orientation,
            style_label=style_label
        )

        if stl_file:
            trace = self.trace_from_stl(stl_file, stl_offset)
            sensor.style.model3d.add_trace(trace)

        self.sensors.append(sensor)

    def add_magnet(self, shape, polarization, dimension, position, orientation=None, style_magnetization=None):
        """
        Adds a magnet of a specific shape to the simulation.

        Args:
            shape (str): Shape of the magnet ("cylinder", "cube", "sphere", or "cylinder_segment").
            polarization (tuple): Polarization vector of the magnet.
            dimension (tuple or float): Dimensions of the magnet (e.g., diameter, length).
            position (list): 3D position of the magnet [x, y, z].
            orientation (scipy.spatial.transform.Rotation, optional): Orientation of the magnet. Defaults to no rotation.
            style_magnetization (dict, optional): Style parameters for the magnet visualization. Defaults to None.
        """
        if shape == "cylinder":
            magnet = magpy.magnet.Cylinder(
                polarization=polarization,
                dimension=dimension,
                position=position,
                orientation=orientation or R.from_euler('xyz', [0, 0, 0]),
                style_magnetization=style_magnetization,
                style_label=f"magnet_{len(self.magnets) + 1}",
            )
        elif shape == "cube":
            magnet = magpy.magnet.Cuboid(
                polarization=polarization,
                dimension=dimension,
                position=position,
                orientation=orientation or R.from_euler('xyz', [0, 0, 0]),
                style_magnetization=style_magnetization,
                style_label=f"magnet_{len(self.magnets) + 1}",
            )
        elif shape == 'sphere':
            magnet = magpy.magnet.Sphere(
                polarization=polarization,
                diameter=dimension,
                position=position,
                orientation=orientation or R.from_euler('xyz', [0, 0, 0]),
                style_magnetization=style_magnetization,
                style_label=f"magnet_{len(self.magnets) + 1}",
            )
        elif shape == 'cylinder_segment':
            magnet = magpy.magnet.CylinderSegment(
                polarization=polarization,
                dimension=dimension,
                position=position,
                orientation=orientation or R.from_euler('xyz', [0, 0, 0]),
                style_magnetization=style_magnetization,
                style_label=f"magnet_{len(self.magnets) + 1}",
            )
        else:
            raise ValueError(f"Unsupported magnet shape: {shape}")

        self.magnets.append(magnet)

    def set_magnet_path(self, path_positions, path_orientations, magnet_index=0):
        """
        Set the movement and rotation path for a specific magnet in the simulation.

        Args:
            path_positions (list): List of positions for the magnet path.
            path_orientations (list): List of orientations for the magnet path.
            magnet_index (int, optional): Index of the magnet to apply the path to. Defaults to 0.

        Raises:
            IndexError: If the magnet_index is out of range.
        """
        if magnet_index >= len(self.magnets):
            raise IndexError(f"Magnet index {magnet_index} out of range.")

        magnet = self.magnets[magnet_index]
        magnet.move(path_positions, start=0)
        magnet.rotate_from_euler(path_orientations, seq='xyz', start=0)

    def set_sensor_path(self, path_positions, path_orientations, sensor_index=0):
        """
        Set the movement and rotation path for a specific sensor in the simulation.

        Args:
            path_positions (list): List of positions for the sensor path.
            path_orientations (list): List of orientations for the sensor path.
            sensor_index (int, optional): Index of the sensor to apply the path to. Defaults to 0.

        Raises:
            IndexError: If the sensor_index is out of range.
        """
        if sensor_index >= len(self.sensors):
            raise IndexError(f"Sensor index {sensor_index} out of range.")

        sensor = self.sensors[sensor_index]
        sensor.move(path_positions, start=0)
        sensor.rotate_from_euler(path_orientations, seq='xyz', start=0)

    def get_magnetic_field_at_sensors(self, sensor_index=0):
        """
        Compute the magnetic field at the position of a specific sensor for each timestep.

        Args:
            sensor_index (int, optional): Index of the sensor. Defaults to 0.

        Returns:
            np.ndarray: Magnetic field components (Bx, By, Bz) at each timestep.

        Raises:
            IndexError: If the sensor_index is out of range.
        """
        if sensor_index >= len(self.sensors):
            raise IndexError(f"Sensor index {sensor_index} out of range.")

        sensor = self.sensors[sensor_index]
        B_field = magpy.getB(self.magnets, sensor)
        return B_field

    def display_simulation(self, animation=True, backend="plotly"):
        """
        Display the magnet and sensor simulation with optional animation.

        Args:
            animation (bool, optional): Whether to animate the simulation. Defaults to True.
            backend (str, optional): The backend for rendering the simulation. Defaults to "plotly".
        """
        fig = magpy.show(*self.sensors, *self.magnets, animation=animation, backend=backend, return_fig=True)

        fig.update_layout(
            scene=dict(
                xaxis=dict(backgroundcolor="white"),
                yaxis=dict(backgroundcolor="white"),
                zaxis=dict(backgroundcolor="white"),
            ),
            paper_bgcolor="white",
            plot_bgcolor="white"
        )
        fig.show()

    @staticmethod
    def _bin_color_to_hex(x):
        """
        Converts binary RGB value to hexadecimal color.

        Args:
            x (int): Binary RGB value.

        Returns:
            str: Hexadecimal color code.
        """
        sb = f"{x:015b}"[::-1]
        r = int(sb[:5], base=2) / 31
        g = int(sb[5:10], base=2) / 31
        b = int(sb[10:15], base=2) / 31
        return to_hex((r, g, b))

    def trace_from_stl(self, stl_file, stl_offset=None):
        """
        Generate a Magpylib 3D model trace from an STL file, applying an optional offset.

        Args:
            stl_file (str): Path to the STL file.
            stl_offset (list, optional): 3D offset to apply to the STL vertices. Defaults to None.

        Returns:
            dict: Trace dictionary for rendering the STL model in Magpylib.
        """
        stl_mesh = Mesh.from_file(stl_file)
        vertices, ixr = np.unique(stl_mesh.vectors.reshape(-1, 3), return_inverse=True, axis=0)

        if stl_offset is not None:
            vertices += np.array(stl_offset) * 1e3  # Apply offset in mm

        i = np.take(ixr, range(0, len(ixr), 3))
        j = np.take(ixr, range(1, len(ixr), 3))
        k = np.take(ixr, range(2, len(ixr), 3))
        x, y, z = vertices.T / 1000  # Convert from mm to meters

        colors = stl_mesh.attr.flatten()
        facecolor = np.array([self._bin_color_to_hex(c) for c in colors]).T

        return {"backend": "generic", "constructor": "mesh3d",
                "kwargs": dict(x=x, y=y, z=z, i=i, j=j, k=k, facecolor=facecolor)}

    def display_with_outputs(self, backend='plotly', animation=True, show=True):
        """
        Display the magnetic field components (Bx, By, Bz) and the 3D model in a multi-column layout.

        Args:
            backend (str, optional): The backend for rendering the simulation. Defaults to "plotly".
            animation (bool, optional): Whether to animate the simulation. Defaults to True.
            show (bool, optional): Whether to display the plot immediately. Defaults to True.

        Returns:
            plotly.graph_objects.Figure: The generated figure.
        """
        if not self.sensors:
            raise ValueError("No sensors added to the simulation.")
        fig = magpy.show(
            dict(objects=[*self.magnets, *self.sensors], output=["Bx", "By", "Bz"], col=1),
            dict(objects=[*self.magnets, *self.sensors], output="model3d", col=2),
            backend=backend, animation=animation, return_fig=True,
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    backgroundcolor="white",
                    color="black",
                    gridcolor="lightgray",
                    range=[-0.005, 0.005],
                ),
                yaxis=dict(
                    backgroundcolor="white",
                    color="black",
                    gridcolor="lightgray",
        range = [-0.005, 0.005],
        ),
                zaxis=dict(
                    backgroundcolor="white",
                    color="black",
                    gridcolor="lightgray",
                    range=[-0.001, 0.009],

                ),
                aspectmode='cube',
            ),
            paper_bgcolor="white",
            plot_bgcolor="white"
        )

        if show:
            fig.show()
        return fig


    def stream_plot(self, plane="XY", plane_limits=(-0.05, 0.05), resolution=100, timestep=0, save_path=None, show=True):
        """
        Generate and optionally save a magnetic field stream plot for a specified plane at a specific timestep.

        Args:
            plane (str, optional): Plane to generate the plot ('XY', 'XZ', or 'YZ'). Defaults to 'XY'.
            plane_limits (tuple, optional): Range of the plane axes. Defaults to (-0.05, 0.05).
            resolution (int, optional): Grid resolution for the plot. Defaults to 100.
            timestep (int, optional): Timestep to plot. Set to -1 for the last timestep. Defaults to 0.
            save_path (str, optional): Path to save the plot. Defaults to None.
            show (bool, optional): Whether to display the plot. Defaults to True.

        Raises:
            ValueError: If the timestep or plane is out of range.
        """
        fig, ax = plt.subplots()
        num_timesteps = len(self.magnets[0].position)

        if timestep == -1:
            timestep = num_timesteps - 1

        if timestep < 0 or timestep >= num_timesteps:
            raise ValueError(f"Timestep {timestep} is out of range (0 to {num_timesteps - 1}).")

        for magnet in self.magnets:
            if isinstance(magnet.position[0], (list, tuple, np.ndarray)):
                magnet.position = magnet.position[timestep]
            if isinstance(magnet.orientation, list) or isinstance(magnet.orientation, np.ndarray):
                magnet.orientation = magnet.orientation[timestep]

        if plane == "XY":
            X, Y = np.meshgrid(
                np.linspace(plane_limits[0], plane_limits[1], resolution),
                np.linspace(plane_limits[0], plane_limits[1], resolution)
            )
            Z = np.zeros_like(X)
            grid = np.stack([X, Y, Z], axis=-1)
        elif plane == "XZ":
            X, Z = np.meshgrid(
                np.linspace(plane_limits[0], plane_limits[1], resolution),
                np.linspace(plane_limits[0], plane_limits[1], resolution)
            )
            Y = np.zeros_like(X)
            grid = np.stack([X, Y, Z], axis=-1)
        elif plane == "YZ":
            Y, Z = np.meshgrid(
                np.linspace(plane_limits[0], plane_limits[1], resolution),
                np.linspace(plane_limits[0], plane_limits[1], resolution)
            )
            X = np.zeros_like(Y)
            grid = np.stack([X, Y, Z], axis=-1)
        else:
            raise ValueError(f"Unsupported plane: {plane}. Choose from 'XY', 'XZ', or 'YZ'.")

        B = magpy.getB(self.magnets, observers=grid.reshape(-1, 3)).reshape(grid.shape)

        if plane == "XY":
            Bx, By = B[:, :, 0], B[:, :, 1]
            normB = np.linalg.norm(B[:, :, :2], axis=-1)
            cp = ax.contourf(X, Y, normB, cmap="GnBu", levels=100)
            ax.streamplot(X, Y, Bx, By, color="k", density=1.5, linewidth=1)
        elif plane == "XZ":
            Bx, Bz = B[:, :, 0], B[:, :, 2]
            normB = np.linalg.norm(B[:, :, [0, 2]], axis=-1)
            cp = ax.contourf(X, Z, normB, cmap="GnBu", levels=100)
            ax.streamplot(X, Z, Bx, Bz, color="k", density=1.5, linewidth=1)
        elif plane == "YZ":
            By, Bz = B[:, :, 1], B[:, :, 2]
            normB = np.linalg.norm(B[:, :, [1, 2]], axis=-1)
            cp = ax.contourf(Y, Z, normB, cmap="GnBu", levels=100)
            ax.streamplot(Y, Z, By, Bz, color="k", density=1.5, linewidth=1)

        fig.colorbar(cp, ax=ax, label="|B| (T)")
        ax.set_xlabel(f'{plane[0]}-position (m)')
        ax.set_ylabel(f'{plane[1]}-position (m)')
        ax.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            output_dir = os.path.dirname(save_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            fig.savefig(save_path)
            print(f"Stream plot saved to {save_path}")
        if show:
            plt.show()

        plt.close(fig)

    def save_b_field(self, filepath, sensor_index=0):
        """
        Save the magnetic field (B_field) data to a CSV file with x path positions in the first column.

        Args:
            filepath (str): Path to save the CSV file.
            sensor_index (int, optional): Index of the sensor to get magnetic field data for. Defaults to 0.

        Raises:
            IndexError: If the sensor_index is out of range.
        """
        if sensor_index >= len(self.sensors):
            raise IndexError(f"Sensor index {sensor_index} out of range.")

        B_field = self.get_magnetic_field_at_sensors(sensor_index)
        x_positions = [pos[0] for pos in self.magnets[0].position]

        data = {
            "x_position": x_positions,
            "Bx": B_field[:, 0],  # Magnetic field component in the x-direction
            "By": B_field[:, 1],  # Magnetic field component in the y-direction
            "Bz": B_field[:, 2],  # Magnetic field component in the z-direction
        }

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"B_field data saved to {filepath}")


# Example usage
if __name__ == "__main__":
    simulation = MagnetSensorSimulation()
    steps=200

    simulation.add_sensor(
        sensor_pixels=[(0, 0, 0)],
        sensor_position=[0, 0, 0],
        stl_file='model/A31301EEJASR-XYZ-IC-20.stl',
        stl_offset=[0,0, -0.0008]
    )

    simulation.add_magnet(
        shape="cylinder",
        polarization=(0, 0, -1),
        dimension=(0.003, 0.005),
        position=[0, 0, 0.001 + 0.005 / 2],
        style_magnetization = {
        'color': {
                'north': '#00FFFF',
                'south': '#00008B',
                'middle': '#FFFFFF',
                'mode': 'tricolor'
        }},
    )

    simulation.add_magnet(
        shape="cylinder",
        polarization=(0, 0, -1),
        dimension=(0.003, 0.005),
        position=[0.014, 0, 0.001 + 0.005 / 2]
    )

    path_positions = [(x, 0, 0) for x in np.linspace(-0.005, 0.005, steps)]
    path_orientations = [(0, 0, 0)] * steps

    simulation.set_magnet_path(path_positions, path_orientations)
    sensor_path_positions = [[0, 0, 0]] * steps
    sensor_path_orientations = [(0, 0, 0)] * steps
    simulation.set_sensor_path(sensor_path_positions, sensor_path_orientations)

    simulation.display_simulation(animation=True, backend="plotly")
    simulation.display_with_outputs()
    simulation.save_b_field("out/b_field_cylinder_diametric.csv")
    simulation.stream_plot(plane="XY", plane_limits=(-0.005, 0.005), resolution=100, timestep=-1)

