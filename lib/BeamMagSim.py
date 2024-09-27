import numpy as np
from scipy.spatial.transform import Rotation as R
from BeamModel import VonKarmanBeamSolver
from MagSim import MagnetSensorSimulation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os


class BeamSensorTrajectory:
    """Handles the generation of sensor and magnet trajectories for a beam-based magnetic sensor simulation."""

    def __init__(self, sensor_beam_config, magnet_beam_config, simulation, unit_conversion=0.001,
                 frame_translation=None, sensor_rotation_angles=None, magnet_rotation_angles=None):
        """
        Initializes the BeamSensorTrajectory class.

        Args:
            sensor_beam_config (dict or str): Beam configuration for the sensor, as a dict or YAML file path.
            magnet_beam_config (dict or str): Beam configuration for the magnet, as a dict or YAML file path.
            simulation (MagnetSensorSimulation): Simulation object for running the sensor and magnet simulation.
            unit_conversion (float, optional): Unit conversion factor from mm to meters. Defaults to 0.001.
            frame_translation (list of float, optional): 3D translation to apply between beam and magnet simulation.
                Defaults to None.
            sensor_rotation_angles (list of float, optional): Rotation angles (Euler angles) for the sensor in degrees.
                Defaults to [0, 90, 0].
            magnet_rotation_angles (list of float, optional): Rotation angles (Euler angles) for the magnet in degrees.
                Defaults to [0, 90, 90].
        """

        # Check if sensor_beam_config is a dictionary or a YAML file path
        if isinstance(sensor_beam_config, dict):
            # Initialize using dictionaries
            self.sensor_beam_solver = VonKarmanBeamSolver(**sensor_beam_config)
        else:
            self.sensor_beam_solver = VonKarmanBeamSolver.from_yaml(sensor_beam_config)

        # Check if magnet_beam_config is a dictionary or a YAML file path
        if isinstance(magnet_beam_config, dict):
            # Initialize using dictionaries
            self.magnet_beam_solver = VonKarmanBeamSolver(**magnet_beam_config)
        else:
            self.magnet_beam_solver = VonKarmanBeamSolver.from_yaml(magnet_beam_config)

        # Initialize and solve the beam models for the sensor and magnet
        self.sensor_beam_solver.solve()
        self.magnet_beam_solver.solve()

        self.simulation = simulation
        self.unit_conversion = unit_conversion
        self.frame_translation = frame_translation if frame_translation is not None else [0, 0, 0]

        sensor_rotation_angles = sensor_rotation_angles if sensor_rotation_angles is not None else [0, 90, 0]
        self.sensor_rotation_matrix = R.from_euler('xyz', sensor_rotation_angles, degrees=True).as_matrix()

        magnet_rotation_angles = magnet_rotation_angles if magnet_rotation_angles is not None else [0, 90, 90]
        self.magnet_rotation_matrix = R.from_euler('xyz', magnet_rotation_angles, degrees=True).as_matrix()


    def apply_translation(self, position):
        """
        Applies frame translation to a given position.

        Args:
            position (list of float): Position vector.

        Returns:
            list of float: Translated position.
        """
        return [position[0] + self.frame_translation[0], position[1] + self.frame_translation[1],
                position[2] + self.frame_translation[2]]

    def apply_rotation(self, position, rotation_matrix):
        """
        Applies a given rotation matrix to the position.

        Args:
            position (list of float): Position vector.
            rotation_matrix (np.ndarray): 3x3 rotation matrix.

        Returns:
            np.ndarray: Rotated position vector.
        """
        return np.dot(rotation_matrix, position)

    def create_sensor_trajectory(self):
        """
        Generates sensor trajectory based on the last elements of each solution from the sensor beam deformation.

        Returns:
            tuple: (sensor_positions, sensor_orientations) - Lists of sensor positions and orientations.
        """
        sensor_positions = []
        sensor_orientations = []

        for solution in self.sensor_beam_solver.solutions:
            x_final = solution.t[-1]  # Last value of x
            w_final = solution.y[0][-1]  # Last value of vertical deflection (w)
            u_final = solution.y[2][-1]  # Last value of horizontal displacement (u)
            slope_final = solution.y[1][-1]  # Last value of slope (dw/dx)
            angle_final = np.degrees(slope_final)

            x_meters = x_final * self.unit_conversion
            u_meters = u_final * self.unit_conversion
            w_meters = w_final * self.unit_conversion

            position = [x_meters, u_meters, w_meters]
            rotated_position = self.apply_rotation(position, self.sensor_rotation_matrix)
            translated_position = self.apply_translation(rotated_position)
            sensor_positions.append(translated_position)

            orientation = [0, 0, np.radians(angle_final)]
            sensor_orientations.append(orientation)

        return sensor_positions, sensor_orientations

    def create_magnet_trajectory(self):
        """
        Generates magnet trajectory based on the magnet beam solver.

        Returns:
            tuple: (magnet_positions, magnet_orientations) - Lists of magnet positions and orientations.
        """
        magnet_positions = []
        magnet_orientations = []

        for solution in self.magnet_beam_solver.solutions:
            x_final = solution.t[-1]  # Last value of x
            w_final = solution.y[0][-1]  # Last value of vertical deflection (w)
            u_final = solution.y[2][-1]  # Last value of horizontal displacement (u)
            slope_final = solution.y[1][-1]  # Last value of slope (dw/dx)

            x_meters = x_final * self.unit_conversion
            u_meters = u_final * self.unit_conversion
            w_meters = w_final * self.unit_conversion

            position = [x_meters, u_meters, w_meters]
            rotated_position = self.apply_rotation(position, self.magnet_rotation_matrix)
            translated_position = self.apply_translation(rotated_position)
            magnet_positions.append(translated_position)

            orientation = [0, 0, np.radians(np.degrees(slope_final))]
            magnet_orientations.append(orientation)

        return magnet_positions, magnet_orientations

    def apply_sensor_trajectory(self):
        """Applies the generated sensor trajectory to the simulation."""
        sensor_positions, sensor_orientations = self.create_sensor_trajectory()
        self.simulation.set_sensor_path(sensor_positions, sensor_orientations)

    def apply_magnet_trajectory(self):
        """Applies the generated magnet trajectory to the simulation."""
        magnet_positions, magnet_orientations = self.create_magnet_trajectory()
        self.simulation.set_magnet_path(magnet_positions, magnet_orientations)

    def plot_trajectories_2d(self, sensor_positions, magnet_positions, sensor_orientations, magnet_orientations,
                             file_path=None, show=True):
        """
        Creates a Plotly 2D subplot showing movement (x, y, z) and angle for both the sensor and magnet.

        Args:
            sensor_positions (list): List of sensor positions.
            magnet_positions (list): List of magnet positions.
            sensor_orientations (list): List of sensor orientations.
            magnet_orientations (list): List of magnet orientations.
            file_path (str, optional): Path to save the plot. Defaults to None.
            show (bool, optional): Whether to display the plot. Defaults to True.

        Returns:
            plotly.graph_objects.Figure: The generated figure.
        """
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=("Sensor Position", "Sensor Angle", "Magnet Position", "Magnet Angle"))

        time = list(range(len(sensor_positions)))

        # Sensor positions (x, y, z)
        sensor_x, sensor_y, sensor_z = zip(
            *[(p[0] * 1e3, p[1] * 1e3, p[2] * 1e3) for p in sensor_positions])  # Convert to mm
        fig.add_trace(go.Scatter(x=time, y=sensor_x, mode='lines', name='Sensor X (mm)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=time, y=sensor_y, mode='lines', name='Sensor Y (mm)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=time, y=sensor_z, mode='lines', name='Sensor Z (mm)'), row=1, col=1)

        # Sensor angles
        sensor_angles = [np.degrees(o[2]) for o in sensor_orientations]
        fig.add_trace(go.Scatter(x=time, y=sensor_angles, mode='lines', name='Sensor Angle (degrees)'), row=1, col=2)

        # Magnet positions
        magnet_x, magnet_y, magnet_z = zip(*[(p[0] * 1e3, p[1] * 1e3, p[2] * 1e3) for p in magnet_positions])
        fig.add_trace(go.Scatter(x=time, y=magnet_x, mode='lines', name='Magnet X (mm)'), row=2, col=1)
        fig.add_trace(go.Scatter(x=time, y=magnet_y, mode='lines', name='Magnet Y (mm)'), row=2, col=1)
        fig.add_trace(go.Scatter(x=time, y=magnet_z, mode='lines', name='Magnet Z (mm)'), row=2, col=1)

        # Magnet angles
        magnet_angles = [np.degrees(o[2]) for o in magnet_orientations]  # Assuming z-rotation
        fig.add_trace(go.Scatter(x=time, y=magnet_angles, mode='lines', name='Magnet Angle (degrees)'), row=2, col=2)

        # Layout settings
        fig.update_layout(height=800, width=1000, title_text="Sensor and Magnet Movement and Angles Over Time")
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=2)
        fig.update_yaxes(title_text="Position (mm)", row=1, col=1)
        fig.update_yaxes(title_text="Angle (degrees)", row=1, col=2)
        fig.update_yaxes(title_text="Position (mm)", row=2, col=1)
        fig.update_yaxes(title_text="Angle (degrees)", row=2, col=2)
        if file_path is not None:
            fig.write_html(file_path)
        if show:
            fig.show()
        return fig

    def save_results_to_csv(self, sensor_positions, sensor_orientations, magnet_positions, magnet_orientations,
                            file_name="combined_data.csv", return_data=False):
        """
        Saves sensor and magnet positions, orientations, and magnetic field data to a CSV file or returns a DataFrame.

        Args:
            sensor_positions (list): List of sensor positions.
            sensor_orientations (list): List of sensor orientations.
            magnet_positions (list): List of magnet positions.
            magnet_orientations (list): List of magnet orientations.
            file_name (str, optional): Name of the CSV file. Defaults to "combined_data.csv".
            return_data (bool, optional): Whether to return the data as a DataFrame instead of saving it. Defaults to False.

        Returns:
            pandas.DataFrame: DataFrame with combined data if return_data is True, otherwise None.
        """
        sensor_data = {'Time': list(range(len(sensor_positions))), 'Sensor_X (m)': [p[0] for p in sensor_positions],
            'Sensor_Y (m)': [p[1] for p in sensor_positions], 'Sensor_Z (m)': [p[2] for p in sensor_positions],
            'Sensor_Orientation_Z (rad)': [o[2] for o in sensor_orientations]}

        magnet_data = {'Magnet_X (m)': [p[0] for p in magnet_positions],
            'Magnet_Y (m)': [p[1] for p in magnet_positions], 'Magnet_Z (m)': [p[2] for p in magnet_positions],
            'Magnet_Orientation_Z (rad)': [o[2] for o in magnet_orientations]}

        B_values = self.simulation.get_magnetic_field_at_sensors(0)
        magnetic_field_data = {'Bx (T)': [B[0] for B in B_values], 'By (T)': [B[1] for B in B_values],
            'Bz (T)': [B[2] for B in B_values]}

        combined_data = {**sensor_data, **magnet_data, **magnetic_field_data}

        combined_df = pd.DataFrame(combined_data)

        if return_data:
            return combined_df

        output_dir = os.path.dirname(file_name)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not file_name.endswith(".csv"):
            file_name += ".csv"

            combined_df.to_csv(file_name, index=False)
            print(f"Combined data saved to {file_name}")
        return None

    def compute_sensitivity(self, applied_forces, file_path=None, show=True):
        """
        Computes the sensitivity (gradient of B-field components with respect to force) and plots sensitivity vs. force.

        Args:
            applied_forces (list): List of applied forces.
            file_path (str, optional): Path to save the plot. Defaults to None.
            show (bool, optional): Whether to display the plot. Defaults to True.

        Returns:
            tuple: Sensitivity of Bx, By, Bz (in G/N), and the force corresponding to max(abs(Bx)).
        """
        B_values = self.simulation.get_magnetic_field_at_sensors()
        Bx_values = np.array([B[0] for B in B_values]) * 1e4
        By_values = np.array([B[1] for B in B_values]) * 1e4
        Bz_values = np.array([B[2] for B in B_values]) * 1e4

        applied_forces_N = np.insert(np.array(applied_forces), 0, 0)

        # Compute the gradient of B-field components with respect to the applied force
        sensitivity_Bx = np.gradient(Bx_values, applied_forces_N)  # dBx/dF in G/N
        sensitivity_By = np.gradient(By_values, applied_forces_N)  # dBy/dF in G/N
        sensitivity_Bz = np.gradient(Bz_values, applied_forces_N)  # dBz/dF in G/N

        max_abs_Bx_index = np.argmax(np.abs(Bx_values))
        max_force = applied_forces_N[max_abs_Bx_index]
        print(f"The maximum applied force corresponding to max(abs(Bx)) is: {max_force} N")

        if file_path is not None:
            with open('max_force.txt', 'w') as f:
                f.write(f"The maximum applied force corresponding to max(abs(Bx)) is: {max_force} N\n")

        fig_sensitivity = go.Figure()
        fig_sensitivity.add_trace(go.Scatter(x=applied_forces_N, y=sensitivity_Bx, mode='lines', name='dBx/dF (G/N)'))
        fig_sensitivity.add_trace(go.Scatter(x=applied_forces_N, y=sensitivity_By, mode='lines', name='dBy/dF (G/N)'))
        fig_sensitivity.add_trace(go.Scatter(x=applied_forces_N, y=sensitivity_Bz, mode='lines', name='dBz/dF (G/N)'))
        fig_sensitivity.update_layout(title='Sensitivity (dB/dF) over Applied Force', xaxis_title='Applied Force (N)',
                                      yaxis_title='Sensitivity (G/N)')
        if file_path is not None:
            fig_sensitivity.write_html(file_path)
        if show:
            fig_sensitivity.show()

        # Return the sensitivity data and max force
        return sensitivity_Bx, sensitivity_By, sensitivity_Bz, max_force


    def run_simulation(self, save_as=None, return_data=False):
        """
        Runs the full sensor and magnet simulation and optionally returns the data.

        Args:
            save_as (str, optional): File path to save the results. Defaults to None.
            return_data (bool, optional): Whether to return the data as a DataFrame. Defaults to False.

        Returns:
            pandas.DataFrame: Data with sensitivity and safety factors, if return_data is True.
        """
        sensor_positions, sensor_orientations = self.create_sensor_trajectory()
        magnet_positions, magnet_orientations = self.create_magnet_trajectory()


        self.simulation.set_sensor_path(sensor_positions, sensor_orientations)
        self.simulation.set_magnet_path(magnet_positions, magnet_orientations)

        # Animate and display the sensor and magnet movement
        fig = self.simulation.display_with_outputs(animation=True, backend="plotly", show=False)

        # Compute sensitivity
        sensitivity_Bx, sensitivity_By, sensitivity_Bz, max_force = self.compute_sensitivity(self.sensor_beam_solver.applied_force, show=False)
        yield_safety_factor = self.sensor_beam_solver.safety_factors_yield[-1]
        fatigue_safety_factor = self.sensor_beam_solver.safety_factors_fatigue[-1]
        force_below_yield_sf = self.sensor_beam_solver.force_below_yield_sf
        force_below_fatigue_sf = self.sensor_beam_solver.force_below_fatigue_sf

        if return_data:
            data = self.save_results_to_csv(sensor_positions, sensor_orientations, magnet_positions, magnet_orientations,
                                        return_data=True)
            data['sensitivity_Bx (G/N)'] = sensitivity_Bx
            data['sensitivity_By (G/N)'] = sensitivity_By
            data['sensitivity_Bz (G/N)'] = sensitivity_Bz
            data['max_force (N)'] = max_force
            data['yield_safety_factor'] = yield_safety_factor
            data['fatigue_safety_factor'] = fatigue_safety_factor
            data['force_below_yield_sf (N)'] = force_below_yield_sf
            data['force_below_fatigue_sf (N)'] = force_below_fatigue_sf

            return data

        if save_as:
            sensitivity_plot_file = save_as if save_as.endswith('.html') else save_as + '_sensitivity.html'
            self.compute_sensitivity(self.sensor_beam_solver.applied_force, file_path=sensitivity_plot_file)
            svg_file = save_as if save_as.endswith('.svg') else save_as + '.svg'
            self.simulation.stream_plot(plane="XZ", plane_limits=(-0.01, 0.01), resolution=100, timestep=-1,
                                        save_path=svg_file)
        else:
            self.plot_trajectories_2d(sensor_positions, magnet_positions, sensor_orientations, magnet_orientations)
            self.compute_sensitivity(self.sensor_beam_solver.applied_force)

        return None


# Example usage
# sensor_beam_config_path = 'config/sensor_beam_parameters.yaml'
# magnet_beam_config_path = 'config/magnet_beam_parameters.yaml'

if '__main__' == __name__:
    sensor_beam_config = {'P': 350, 'L': 30, 'E': 72000, 'b': 11.5, 'h': 4.5, 'point_offset': 5.5, 'kappa': 0.833,
        'poisson_ratio': 0.33, 'time_steps': 10}

    magnet_beam_config = {'P': 350, 'L': 30, 'E': 72000, 'b': 11.5, 'h': 4.5, 'point_offset': 5.5, 'kappa': 0.833,
        'poisson_ratio': 0.33, 'time_steps': 10}

    # Initialize the simulation
    simulation = MagnetSensorSimulation()
    simulation.add_sensor(sensor_pixels=[(0, 0, 0)], sensor_position=[0, 0, 0],
                          stl_file='model/A31301EEJASR-XYZ-IC-20.stl', stl_offset=[0, 0, -0.0008])
    simulation.add_magnet(shape="cylinder", polarization=(0, 0, -1), dimension=(0.003, 0.002),
        position=[0, 0, 0.0015 + 0.001], style_magnetization={'color': {'north': '#00FFFF',  # Cyan for the north pole
            'south': '#00008B',  # Dark blue for the south pole
            'middle': '#FFFFFF',  # White for the middle
            'mode': 'tricolor'  # Use tricolor mode
        }})

    # Create the trajectory generator with a frame translation
    trajectory_generator = BeamSensorTrajectory(sensor_beam_config, magnet_beam_config, simulation,
                                                frame_translation=[0, 0, 0.03])

    # Run the full simulation internally
    trajectory_generator.run_simulation(save_as='out1/mag1')
