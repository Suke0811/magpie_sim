# MAGPIE Sensor Mechanisms Simulation APIs

??? warning 

    All api docs are auto-generated and refernce only


## How to Use the `BeamSensorTrajectory` and `MagnetSensorSimulation` Classes

This guide explains how to use the `BeamSensorTrajectory` class in conjunction with the `MagnetSensorSimulation` class for simulating beam and magnetic sensor interactions. 
The example provided runs a full simulation of sensor and magnet trajectories using beam deformation models and magnet configurations.

For the APIs, please see the following links. 

::cards::  image-tags cols="3" class_name="bigger"
- title: Beam Deformation
  content: • Von Karaman Beam Solver <br> • Beam deformation in $x$, $y$ and $\theta$
  url: ../beam
- title: Magnetic Simulation
  content: • Magnetic field simulation <br> • Sensor plots <br> • stream plots
  url: ../mag
- title: Trajectory Generator
  content: • handle sequential trajectory generation
  url: ../traj
::/cards::

---

## Prerequisites
Before running the code, make sure you have the following packages installed:
```bash
pip install -r requirements.txt
```

## Example Usage

1. **Define Sensor and Magnet Beam Configurations**  
   To begin, define the configurations for both the sensor beam and magnet beam using dictionaries. You can also load these from YAML files if desired. In this example, both configurations are passed directly as dictionaries.

    ```python
    sensor_beam_config = {
        'P': 350,  # Maximum load in Newtons
        'L': 30,  # Length of the beam in mm
        'E': 72000,  # Young's Modulus in MPa
        'b': 11.5,  # Width of the beam in mm
        'h': 4.5,  # Thickness of the beam in mm
        'point_offset': 5.5,  # Offset from the center of the beam in mm
        'kappa': 0.833,  # Shear correction factor
        'poisson_ratio': 0.33,  # Poisson's ratio of the material
        'time_steps': 10  # Number of time steps for force increment
    }

    magnet_beam_config = {
        'P': 350,
        'L': 30,
        'E': 72000,
        'b': 11.5,
        'h': 4.5,
        'point_offset': 5.5,
        'kappa': 0.833,
        'poisson_ratio': 0.33,
        'time_steps': 10
    }
    ```

2. **Initialize the Simulation**  
   Create an instance of the `MagnetSensorSimulation` class and add the sensor and magnet to the simulation. You can specify sensor pixel coordinates, positions, orientations, and load STL files to represent the sensor and magnet geometries.

    ```python
    simulation = MagnetSensorSimulation()
    
    # Add a sensor to the simulation
    simulation.add_sensor(
        sensor_pixels=[(0, 0, 0)],  # Sensor pixel configuration
        sensor_position=[0, 0, 0],  # Position of the sensor
        stl_file='model/A31301EEJASR-XYZ-IC-20.stl',  # STL model for the sensor
        stl_offset=[0, 0, -0.0008]  # Offset for the sensor STL model
    )
    
    # Add a magnet to the simulation
    simulation.add_magnet(
        shape="cylinder",  # Magnet shape (cylinder)
        polarization=(0, 0, -1),  # Polarization direction of the magnet
        dimension=(0.003, 0.002),  # Dimensions of the magnet (in meters)
        position=[0, 0, 0.0015 + 0.001],  # Position of the magnet
        style_magnetization={
            'color': {'north': '#00FFFF', 'south': '#00008B', 'middle': '#FFFFFF', 'mode': 'tricolor'}
        }
    )
    ```

3. **Create the Beam and Magnet Trajectories**  
   Create an instance of the `BeamSensorTrajectory` class, which generates sensor and magnet trajectories based on the provided beam configurations.

    ```python
    trajectory_generator = BeamSensorTrajectory(
        sensor_beam_config,  # Beam configuration for the sensor
        magnet_beam_config,  # Beam configuration for the magnet
        simulation,  # Simulation instance
        frame_translation=[0, 0, 0.03]  # Translation of the frame
    )
    ```

4. **Run the Simulation**  
   Call the `run_simulation` method of the `BeamSensorTrajectory` instance to run the full simulation. You can also specify a path to save the results of the simulation.

    ```python
    trajectory_generator.run_simulation(save_as='out1/mag1')
    ```

## Methods
- `add_sensor()`: Adds a sensor to the simulation with the option to load an STL file for 3D visualization.
- `add_magnet()`: Adds a magnet to the simulation. You can specify the magnet's shape (e.g., cylinder, cube), polarization, and position.
- `create_sensor_trajectory()`: Generates the sensor trajectory based on beam deformation, which includes sensor positions and orientations.
- `create_magnet_trajectory()`: Similar to `create_sensor_trajectory`, this generates the magnet's trajectory based on the magnet beam deformation model.
- `run_simulation()`: Runs the full simulation, applying sensor and magnet trajectories, displaying the results, and computing sensitivity data. You can optionally save the results to files.

## Output and Visualization

### Plotting Sensor and Magnet Trajectories
The `plot_trajectories_2d()` method creates a 2D plot of sensor and magnet positions and orientations over time. You can save the plot as an HTML file or display it directly in the browser.

```python
trajectory_generator.plot_trajectories_2d(sensor_positions, magnet_positions, sensor_orientations, magnet_orientations)
```

## Saving Results to CSV
You can save the computed sensor and magnet positions, orientations, and magnetic field data to a CSV file using the save_results_to_csv() method.
```python
trajectory_generator.save_results_to_csv(sensor_positions, sensor_orientations, magnet_positions, magnet_orientations, file_name="output.csv")
```

## Sensitivity Computation
The compute_sensitivity() method calculates the sensitivity (rate of change of the magnetic field with respect to applied forces) and plots the results. You can also save the plot to an HTML file.

```python
trajectory_generator.compute_sensitivity(applied_forces=[0, 100, 200], file_path="sensitivity_plot.html")
```

