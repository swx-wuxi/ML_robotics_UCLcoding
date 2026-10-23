import numpy as np
import os
from matplotlib import pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
from skopt import gp_minimize
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import RBF
from gp_functions import fit_gp_model_1d, fit_gp_model_full, plot_gp_results_1d, plot_gp_slices, make_gp_regressor

# Enable/disable the white-noise term in the GP kernel (used by both gp_minimize and the final fit)
USE_WHITE_KERNEL = True

# Search-space bounds for [kp (7), kd (7)]
GAIN_LOWER = np.array([0.1] * 7 + [0.0] * 7)
GAIN_UPPER = np.array([1000.0] * 7 + [100.0] * 7)

# Configuration for the simulation
conf_file_name = "pandaconfig.json"  # Configuration file for the robot
cur_dir = os.path.dirname(os.path.abspath(__file__))
sim = pb.SimInterface(conf_file_name, conf_file_path_ext = cur_dir)  # Initialize simulation interface

# Get active joint names from the simulation
ext_names = sim.getNameActiveJoints()
ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

source_names = ["pybullet"]  # Define the source for dynamic modeling

# Create a dynamic model of the robot
dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False,0,cur_dir)
num_joints = dyn_model.getNumberofActuatedJoints()

init_joint_angles = sim.GetInitMotorAngles()

print(f"Initial joint angles: {init_joint_angles}")

# Sinusoidal reference
# Specify different amplitude values for each joint
amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
# Specify different frequency values for each joint
frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

# Convert lists to NumPy arrays for easier manipulation in computations
amplitude = np.array(amplitudes)
frequency = np.array(frequencies)
ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference

# Global lists to store data
kp0_values = []
kd0_values = []
gains_stacked = []  # each entry: [kp (7), kd (7)]
tracking_errors = []

def simulate_with_given_pid_values(sim_, kp, kd, episode_duration=10):
    
    # here we reset the simulator each time we start a new test
    sim_.ResetPose()
    
    # IMPORTANT: to ensure that no side effect happens, we need to copy the initial joint angles
    q_des = init_joint_angles.copy()
    qd_des = np.array([0]*dyn_model.getNumberofActuatedJoints())

    time_step = sim_.GetTimeStep()
    current_time = 0
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors

    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all,  = [], [], [], []
    
    steps = int(episode_duration/time_step)
    # testing loop
    for i in range(steps):
        # measure current state
        q_mes = sim_.GetMotorAngles(0)
        qd_mes = sim_.GetMotorVelocities(0)
    
        # Compute sinusoidal reference trajectory
        q_des, qd_des = ref.get_values(current_time)  # Desired position and velocity
        # Control command
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)  
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)  # Set the torque com
        sim_.Step(cmd, "torque")  # Simulation step with torque command

        # Exit logic with 'q' key
        keys = sim_.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim_.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        
        # simulation_time = sim.GetTimeSinceReset()

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(q_des)
        qd_d_all.append(qd_des)
        
        #time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
       
    # Calculate tracking error
    q_mes_all = np.array(q_mes_all)
    q_des_all = np.array(q_d_all)

    tracking_error = np.sum((q_mes_all - q_des_all)**2)  # Sum of squared error as the objective
    # print tracking error
    print("Tracking error: ", tracking_error)
    # print PD gains
    print("kp: ", kp)
    print("kd: ", kd)
    
    return tracking_error


# Objective function for optimization
def objective(params):
    kp = np.array(params[:7])  # First 7 elements correspond to kp
    kd = np.array(params[7:])  # Last 7 elements correspond to kd
    episode_duration = 5
    
    # Call the simulation with given kp and kd values
    # TODO to remove for final version
    tracking_error = simulate_with_given_pid_values(sim, kp, kd, episode_duration)

    # Collect data for the first kp and kd TODO to remove for final version
    gains_stacked.append(np.concatenate([kp, kd]))
    kp0_values.append(kp[0])
    kd0_values.append(kd[0])
    tracking_errors.append(tracking_error)
    
    return tracking_error


def main():
    # Define the search space for Kp and Kd
    # Define the search space as before
    space = [
        Real(GAIN_LOWER[i], GAIN_UPPER[i], name=f'kp{i}') for i in range(7)
    ] + [
        Real(GAIN_LOWER[7 + i], GAIN_UPPER[7 + i], name=f'kd{i}') for i in range(7)
    ]

    # Same kernel/noise settings as the GP used for the slice plots
    gp = make_gp_regressor(len(space), USE_WHITE_KERNEL)

    # Perform Bayesian optimization
    result = gp_minimize(
    objective,
    space,
    n_calls=50,
    n_initial_points=25,
    base_estimator=gp,  # Use the custom Gaussian Process Regressor
    acq_func='EI',      # LCB': Lower Confidence Bound 'EI': Expected Improvement 'PI': Probability of Improvement
    random_state=42)
    
    # Extract the optimal values
    best_kp = result.x[:7]  # Optimal kp vector
    best_kd = result.x[7:]  # Optimal kd vector

    # Prepare data
    # kp0_values_array = np.array(kp0_values).reshape(-1, 1)
    # kd0_values_array = np.array(kd0_values).reshape(-1, 1)
    tracking_errors_array = np.array(tracking_errors)

    # Fit GP models
    gains_array = np.array(gains_stacked)  # shape (n_samples, 14)
    gp_full = fit_gp_model_full(gains_array, tracking_errors_array, GAIN_LOWER, GAIN_UPPER, USE_WHITE_KERNEL)
    print("Fitted full GP kernel:", gp_full.kernel_)
    plot_gp_slices(gp_full, best_kp, best_kd, gains_array, tracking_errors_array, GAIN_LOWER, GAIN_UPPER)

    # gp_kp0 = fit_gp_model_1d(kp0_values_array, tracking_errors_array)
    # gp_kd0 = fit_gp_model_1d(kd0_values_array, tracking_errors_array)

    # # Plot the results
    # plot_gp_results_1d(kp0_values_array, kd0_values_array, tracking_errors_array, gp_kp0, gp_kd0)


    print(f"Optimal Kp: {best_kp}, Optimal Kd: {best_kd}")

if __name__ == "__main__":
    main()