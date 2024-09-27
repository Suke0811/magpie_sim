import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
import pandas as pd
import yaml

class VonKarmanBeamSolver:
    """
    Class to solve the von Karman beam bending problem using numerical methods.
    """

    def __init__(self, P, L, E, b, h, point_offset, time_steps=100, kappa=5/6, poisson_ratio=0.33, yield_strength=250, fatigue_strength=160):
        """
        Initializes the Von Karman Beam Solver.

        Args:
            P (float): Maximum load in Newtons.
            L (float): Length of the beam in mm.
            E (float): Young's Modulus in MPa (N/mm^2).
            b (float): Width of the beam in mm.
            h (float): Thickness of the beam in mm.
            point_offset (float): Offset from the center of the beam in mm.
            time_steps (int, optional): Number of time steps for force increase. Defaults to 100.
            kappa (float, optional): Shear correction factor. Defaults to 5/6.
            poisson_ratio (float, optional): Poisson's ratio of the material. Defaults to 0.33.
            yield_strength (float, optional): Yield strength of the material in MPa. Defaults to 250 MPa.
            fatigue_strength (float, optional): Fatigue strength of the material at 1e7 cycles in MPa. Defaults to 160 MPa.
        """
        self.P_max = P
        self.L = L
        self.E = E
        self.G = E / (2 * (1 + poisson_ratio))  # Shear modulus
        self.b = b
        self.h = h
        self.point_offset = point_offset
        self.kappa = kappa
        self.time_steps = time_steps
        self.yield_strength = yield_strength
        self.fatigue_strength = fatigue_strength

        # Cross-sectional properties
        self.I = (b * h ** 3) / 12  # Second moment of area for rectangular cross-section
        self.A = b * h  # Cross-sectional area

        # Initialize safety factor lists
        self.safety_factors_yield = []  # List to store safety factors for yield strength
        self.safety_factors_fatigue = []  # List to store safety factors for fatigue strength

        # Variables to store forces at which safety factors fall below 2
        self.force_below_yield_sf = None
        self.force_below_fatigue_sf = None

    @classmethod
    def from_yaml(cls, yaml_file):
        """
        Load parameters from a YAML file and initialize the class.

        Args:
            yaml_file (str): Path to the YAML file containing beam configuration parameters.

        Returns:
            VonKarmanBeamSolver: Instance of the VonKarmanBeamSolver class.
        """
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
        return cls(
            P=data['P'],
            L=data['L'],
            E=data['E'],
            b=data['b'],
            h=data['h'],
            point_offset=data['point_offset'],
            time_steps=data.get('time_steps', 100),
            kappa=data.get('kappa', 5 / 6),
            poisson_ratio=data.get('poisson_ratio', 0.33)
        )

    def von_karman_beam(self, t, y, P):
        """
        Differential equation for Von Karman beam bending.

        Args:
            t (float): Position along the beam.
            y (list of float): State vector [deflection, slope, horizontal displacement].
            P (float): Applied load at the current time step.

        Returns:
            list of float: List of derivatives [dw/dx, d²w/dx², du/dx].
        """
        w, dw_dx, u = y  # Deflection, slope, and horizontal displacement
        d2w_dx2 = (P * (self.L - t)) / (self.E * self.I)  # Curvature
        du_dx = 0.5 * (dw_dx ** 2)  # Horizontal stretch
        return [dw_dx, d2w_dx2, du_dx]

    def solve(self, save_path=None):
        """
        Solve the beam equations using numerical integration.

        Args:
            save_path (str, optional): Path to save the results as a CSV file. Defaults to None.

        Returns:
            None
        """
        y0 = [0, 0, 0]  # Initial conditions: w(0) = 0, slope = 0, u(0) = 0
        t_eval = np.linspace(0, self.L, 100)  # Evaluation points along the beam

        # Store solutions for each time step where the force P increases linearly
        self.solutions = []
        self.applied_force = []

        # Solve for t=0 with no force
        sol_initial = solve_ivp(self.von_karman_beam, [0, self.L], y0, t_eval=t_eval, args=(0,))
        self.solutions.append(sol_initial)

        # Solve for subsequent timesteps where force increases linearly
        for t_step in range(1, self.time_steps + 1):
            P_current = self.P_max * (t_step / self.time_steps)  # Linearly increase force
            self.applied_force.append(P_current)
            sol = solve_ivp(self.von_karman_beam, [0, self.L], y0, t_eval=t_eval, args=(P_current,))
            self.solutions.append(sol)

            # Calculate yield and fatigue safety factors
            yield_sf = self.calculate_safety_factor(P_current)
            fatigue_sf = self.calculate_fatigue_safety_factor(P_current)
            self.safety_factors_yield.append(yield_sf)
            self.safety_factors_fatigue.append(fatigue_sf)

            # Record the force when safety factors fall below 2
            if yield_sf < 2 and self.force_below_yield_sf is None:
                self.force_below_yield_sf = P_current

            if fatigue_sf < 2 and self.force_below_fatigue_sf is None:
                self.force_below_fatigue_sf = P_current

        # Extract the final solution for plotting and saving
        final_solution = self.solutions[-1]
        self.x = final_solution.t  # Position along the beam
        self.w = final_solution.y[0]  # Vertical deflection
        self.u = final_solution.y[2]  # Horizontal displacement
        self.slope = final_solution.y[1]  # Slope (dw/dx)
        self.angle = np.degrees(self.slope)  # Slope in degrees

        # Calculate total displacements
        self.u_total = self.u + self.point_offset * np.sin(self.slope)
        self.w_total = self.w + self.point_offset * (1 - np.cos(self.slope))

        if save_path:
            # Save data to a CSV file for the final time step
            data = {
                'Position (x)': self.x,
                'Vertical Deflection (w)': self.w_total,
                'Horizontal Displacement (u)': self.u_total,
                'Angle of Slope (degrees)': self.angle,
                'Safety Factor (Yield)': self.safety_factors_yield[-1],
                'Safety Factor (Fatigue)': self.safety_factors_fatigue[-1]
            }
            df = pd.DataFrame(data)
            df.to_csv(save_path, index=False)
            print(f"Data saved to {save_path}")

    def calculate_safety_factor(self, P):
        """
        Calculate the safety factor for yielding in bending.

        Args:
            P (float): Current applied load in Newtons.

        Returns:
            float: Safety factor for bending.
        """
        M_max = P * self.L / 4  # Maximum bending moment
        c = self.h / 2  # Distance from neutral axis to outermost fiber
        sigma_max = (M_max * c) / self.I  # Maximum stress in bending
        safety_factor = self.yield_strength / sigma_max
        return safety_factor

    def calculate_fatigue_safety_factor(self, P):
        """
        Calculate the safety factor for fatigue at 1e7 cycles.

        Args:
            P (float): Current applied load in Newtons.

        Returns:
            float: Safety factor for fatigue.
        """
        M_max = P * self.L / 4  # Maximum bending moment
        c = self.h / 2  # Distance from neutral axis to outermost fiber
        sigma_max = (M_max * c) / self.I  # Maximum bending stress
        safety_factor_fatigue = self.fatigue_strength / sigma_max
        return safety_factor_fatigue

    def plot(self, save_path=None):
        """
        Generate an interactive plot using Plotly to visualize beam deflection and slope.

        Args:
            save_path (str, optional): Path to save the plot as an HTML file. Defaults to None.

        Returns:
            None
        """
        fig = go.Figure()

        # Horizontal movement trace
        fig.add_trace(go.Scatter(x=self.x, y=self.u_total, mode='lines', name='Horizontal Movement (u)',
                                 line=dict(color='blue')))

        # Vertical movement trace
        fig.add_trace(go.Scatter(x=self.x, y=self.w_total, mode='lines', name='Vertical Movement (w)',
                                 line=dict(color='red', dash='dash')))

        # Angle trace
        fig.add_trace(go.Scatter(x=self.x, y=self.angle, mode='lines', name='Angle of Slope (degrees)',
                                 line=dict(color='green', dash='dot')))

        # Layout settings
        fig.update_layout(title='Movement of the Point at the Bending Edge',
                          xaxis_title='Position along the beam (mm)',
                          yaxis_title='Displacement (mm) / Angle (degrees)',
                          legend=dict(x=0.05, y=0.95),
                          template='plotly_white')

        if save_path:
            # Save plot as an HTML file
            fig.write_html(save_path)
            print(f"Plot saved to {save_path}")
        # Show the interactive plot
        fig.show()

# Example usage
if __name__ == "__main__":
    # Load from YAML file
    beam_solver = VonKarmanBeamSolver.from_yaml('config/beam_parameters.yaml')

    # Solve for the beam displacements and save data to CSV
    beam_solver.solve(save_path="beam_data.csv")

    # Plot the results and save plot as HTML
    beam_solver.plot(save_path="beam_plot.html")
