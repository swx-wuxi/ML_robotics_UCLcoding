# Simulation and Control

This Python package provides a simple interface to the PyBullet simulator and Pinocchio for robotic simulation and control, designed specifically for educational purposes.

## Installation

### Prerequisites

To set up the necessary environment with all dependencies, use the provided `environment.yaml` file which will create a Mamba environment named `robo_env`.

1. If Mamba is not installed, first install it via [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

2. Clone the repository and install the package:
```bash
   git clone https://github.com/VModugno/simulation_and_control.git
   cd simulation_and_control
   mamba env create -f environment.yaml
   conda activate robo_env
   pip install . 
```


## Configuration Files

Place your configuration files in a directory named `config` within your project. Below is an example of a configuration file:
```json
{
    "sim": {
        "time_step": 0.001, // The time step for the simulation, in seconds
        "experiment_duration": 5, // Total duration of the experiment in seconds
        "FL": [["FL_foot", "FL_foot_fixed"]], // Fixations for the front-left foot
        "FR": [["FR_foot", "FR_foot_fixed"]], // Fixations for the front-right foot
        "RL": [["RL_foot", "RL_foot_fixed"]], // Fixations for the rear-left foot
        "RR": [["RR_foot", "RR_foot_fixed"]], // Fixations for the rear-right foot
        "feet_contact_names": [["FR_foot", "FL_foot", "RL_foot", "RR_foot"]] // Names of contact points for the feet
    },
    "robot_pybullet": {
        "base_type": ["fixed"], // The base type of the robot, fixed in this case
        "collision_enable": [0], // Collision detection flag (0 for off)
        "robot_description_model": [""], // Placeholder for model descriptions
        "urdf_path": ["panda_description/panda.urdf"], // Path to the URDF file for the robot
        "ros_urdf_path": "", // Optional path for ROS-compatible URDFs
        "init_link_base_position": [[0, 0, 0]], // Initial position of the robot base
        "init_link_base_vel": [[0.0, 0.0, 0.0]], // Initial velocity of the robot base
        "init_link_base_orientation": [[0.0, 0.0, 0.0, 1]], // Initial orientation quaternion of the robot base
        "init_link_base_ang_vel": [[0.0, 0.0, 0.0]], // Initial angular velocity of the robot base
        "init_motor_angles": [[0.0, 1.5, -1.0, -0.54, 0.0, 0.0, 0.0]], // Initial angles for the motors
        "init_motor_vel": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], // Initial velocities for the motors
        "motor_offset": [[]], // Motor offset settings
        "motor_direction": [[]], // Direction settings for each motor
        "motor_friction": [0], // Friction settings for the motors
        "motor_friction_coeff": [0], // Friction coefficients for the motors
        "foot_friction": [2], // Friction settings for the feet
        "foot_restitution": [0], // Elasticity settings for the feet upon impact
        "enable_feet_joint_force_sensors": [["RL_foot_fixed", "RR_foot_fixed", "FL_foot_fixed", "FR_foot_fixed"]], // Force sensors at the feet joints
        "floating_base_name": ["floating_base"], // Name of the floating base
        "servo_pos_gains": [400], // Positional gain for servo control
        "servo_vel_gains": [3] // Velocity gain for servo control
    },
    "robot_pin": {
        "base_type": ["fixed"], // Indicates that the base of the robot is fixed
        "robot_description_model": [""], // Model description placeholder
        "urdf_path": ["panda_description/panda.urdf"], // Path to the URDF file
        "ros_urdf_path": [""], // Optional ROS URDF path
        "floating_base_name": ["floating_base"], // Name of the floating base
        "joint_state_conversion_active": [1] // Flag for active joint state conversion
    }
}
```

## Usage Example

Here is a full example of how to use the package in your projects:


```python
import numpy as np
import time
import os
import simulation_and_control as sac
from simulation_and_control.sim import pybullet_robot_interface as pb
from simulation_and_control.controllers.servo_motor import MotorCommands
from simulation_and_control.controllers.pin_wrapper import PinWrapper

def main():
    # Configuration for the simulation
    conf_file_name = "elephantconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext = cur_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False,0,cur_dir)

    # Display dynamics information
    print("Joint info simulator:")
    sim.GetBotJointsInfo()

    print("Link info simulator:")
    sim.GetBotDynamicsInfo()

    print("Link info pinocchio:")
    dyn_model.getDynamicsInfo()

    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    while True:
        cmd.tau_cmd = np.zeros((dyn_model.getNumberofActuatedJoints(),))  # Zero torque command
        sim.Step(cmd, "torque")  # Simulation step with torque command

        if dyn_model.visualizer: 
            for index in range(len(sim.bot)): # Conditionally display the robot model
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)  # Update the display of the robot model

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        time.sleep(0.1)  # Slow down the loop for better visualization

if __name__ == '__main__':
    main()
```

## TODO LIST

- [ ] adding in pin_wrapper an internal structure to manage continuos joints. remember that in pinocchio a continuos joint has 2 degrees of freedom


