# Mechanisms and Designs

## Hardware Design

MAGPIE is designed with a parallel gripper that can transition between flat and line foot configurations, serving dual roles in grasping and locomotion. The design incorporates 8-axis force sensing using four 3-axis Hall effect sensors, each installed on compliant flexure mechanisms.

### Foot and Finger Design

The fingers of MAGPIE are designed to measure both ground reaction forces and grasping forces. The flexure mechanism allows for multi-axis force measurement by isolating different axes. The Hall effect sensors are positioned laterally for grasping force measurement, while the magnets are placed longitudinally to measure ground contact forces. This placement helps protect the sensors from high impact forces during landing, ensuring durability.

### Actuation Mechanisms

MAGPIE's actuation mechanism uses a crossed-roller linear rail system, actuated by a brushless DC (BLDC) motor. The gripper is driven by lead screws connected to the motor, distributing torque evenly. The lateral flexure mechanism is designed to be sensitive, while the longitudinal mechanism is protected once the gripper is closed.

### Circuitry and Sensors

The electronics in MAGPIE are modular and integrate the sensors, actuators, and control systems. The system includes Hall effect sensors mounted on custom flexible PCBs, a BLDC motor, an IMU, and other components like a camera and display for future enhancements. This modular design minimizes wiring complexity and improves reliability.

## Force Sensing Mechanism

The Hall effect sensors detect magnetic field changes caused by deflections in the flexure mechanisms. The magnet-sensor configuration allows for multi-axis force measurements, with the flexure beam isolating forces along different axes. By using cylindrical magnets, MAGPIE ensures consistent signal detection across both axes.

The computational framework simulates the magnet’s behavior under different deflections, optimizing the sensor placement and flexure design for accurate force sensing while considering potential interference.

## Gripper and Foot Capabilities

MAGPIE has a custom-designed direct-drive BLDC motor capable of achieving a nominal grasping force of 350 N and opening/closing speeds of 58.3 mm/sec. The design is robust enough to handle ground impact forces while providing precise grasping capabilities. The force-sensing mechanism allows MAGPIE to function as a stable foot while also providing tactile sensing for grasping.
