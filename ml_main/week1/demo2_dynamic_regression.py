import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference 

def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    # Print initial joint angles
    print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    # Sinusoidal reference
    # Specify different amplitude values for each joint
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
    # Specify different frequency values for each joint
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference
    
    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 10 # seconds
    
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    # PD controller gains
    kp = 1000
    kd = 100

    # Initialize data storage
    tau_mes_all = []
    regressor_all = []

    # Data collection loop
    while current_time < max_time:
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)
        
        # Compute sinusoidal reference trajectory
        q_d, qd_d = ref.get_values(current_time)  # Desired position and velocity
        
        # Control command
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)  # Zero torque command
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)  # Set the torque com
        sim.Step(cmd, "torque")

        # Get measured torque
        tau_mes = sim.GetMotorTorques(0)

        if dyn_model.visualizer: 
            for index in range(len(sim.bot)):  # Conditionally display the robot model
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)  # Update the display of the robot model

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # TODO (1)Compute regressor and store it
        regressor = dyn_model.ComputeDynamicRegressor(q_mes,qd_mes,qdd_mes)
        regressor_all.append(regressor.copy())
        tau_mes_all.append(tau_mes.copy())
       
        current_time += time_step
        # Optional: print current time
        # print(f"Current time in seconds: {current_time:.2f}")
   
    # TODO (1) Stack the regressors and measured torques.
    # Each sample: regressor shape (7, 70), torque shape (7,).
    W = np.vstack(regressor_all)

    # Match the torque order to the Pinocchio regressor order.
    T_pin = np.concatenate([
        np.asarray(
            dyn_model.ReoderJoints2PinVec(tau, "vel")
        ).reshape(-1)
        for tau in tau_mes_all
    ])
    T = T_pin

    if W.shape[1] != 70:
        raise ValueError("Expected 70 inertial parameters for seven links.")

    # Estimate all 70 parameters together.
    # No true parameters are used in this estimation.
    W_pinv = np.linalg.pinv(W)
    a_est = W_pinv @ T

    # TODO (2) Organize the estimated parameters by link.
    # Each row contains the 10 estimated parameters of one link.
    a_est_links = a_est.reshape(7, 10)

    names = [
        "m", "mc_x", "mc_y", "mc_z", "I_xx",
        "I_xy", "I_yy", "I_xz", "I_yz", "I_zz"
    ]

    # Model truth is used ONLY for comparison after estimation.
    a_true = np.concatenate([
        np.asarray(inertia.toDynamicParameters()).reshape(-1)
        for inertia in dyn_model.pin_model.inertias[1:]
    ])

    if a_true.size != 70:
        raise ValueError("Expected seven links with 10 parameters each.")

    a_true_links = a_true.reshape(7, 10)

    for link in range(7):
        print(f"\nLink {link + 1}")
        print("Parameter       True       Estimate     Abs. error")
        for name, truth, estimate in zip(
                names, a_true_links[link], a_est_links[link]):
            print(
                f"{name:8s} {truth:12.6g} {estimate:12.6g} "
                f"{abs(estimate - truth):12.6g}"
            )

    # Check how many parameter directions the data can identify.
    model_rank = np.linalg.matrix_rank(W)
    print(f"\nRegressor rank: {model_rank} / {W.shape[1]}")

    if model_rank < W.shape[1]:
        print(
            "Rank deficient: the 70 individual parameters cannot "
            "all be uniquely identified. The pseudoinverse returns "
            "a minimum-norm least-squares solution."
        )

    # TODO (3) Compute training-fit metrics for all joint torques.
    torque_pred_vector = W @ a_est
    residual = T - torque_pred_vector

    n = T.size
    dof = n - model_rank
    sse = float(residual @ residual)
    sst = float(T @ T)

    if dof > 0 and sst > 0:
        r2 = 1 - sse / sst
        adj_r2 = 1 - (sse / dof) / (sst / n)

        f_stat = (
            ((sst - sse) / model_rank) / (sse / dof)
            if model_rank > 0 and sse > 0 else np.nan
        )

        print(f"\nUncentered R-squared: {r2:.6g}")
        print(f"Adjusted uncentered R-squared: {adj_r2:.6g}")
        print(f"F-statistic (no intercept): {f_stat:.6g}")

        # Classical OLS intervals assume independent,
        # constant-variance errors.
        if model_rank == W.shape[1]:
            try:
                from scipy.stats import t as student_t
            except ImportError:
                print("Confidence intervals require scipy.")
            else:
                covariance = (sse / dof) * (W_pinv @ W_pinv.T)
                margin = student_t.ppf(0.975, dof) * np.sqrt(
                    np.maximum(np.diag(covariance), 0)
                )

                print("\nParameter     95% confidence interval")
                for index, (estimate, delta) in enumerate(
                        zip(a_est, margin)):
                    link = index // 10 + 1
                    name = names[index % 10]
                    print(
                        f"L{link} {name:8s} "
                        f"[{estimate - delta:.6g}, "
                        f"{estimate + delta:.6g}]"
                    )
        else:
            print(
                "Individual parameter confidence intervals skipped: "
                "rank deficient."
            )
    else:
        print("Insufficient data or zero response energy for fit statistics.")

    # Reconstruct all joint torques in Pinocchio order.
    # These variables also feed the existing plotting code below.
    torque_pred = torque_pred_vector.reshape(-1, 7)
    torque_true = T_pin.reshape(-1, 7)
    torque_error = torque_pred - torque_true

    print(
        "\nTorque RMSE per joint:",
        np.sqrt(np.mean(torque_error**2, axis=0))
    )

    # TODO （4 opt)plot the torque prediction error for each joint (optional)
    # Plot the full error and a view after startup.
    sample_time = np.arange(torque_error.shape[0]) * time_step
    startup_time = 0.5
    mask = sample_time >= startup_time

    fig, axes = plt.subplots(
        7, 2, figsize=(12, 12), sharex="col"
    )

    for j in range(7):
        axes[j, 0].plot(sample_time, torque_error[:, j])
        axes[j, 1].plot(sample_time[mask], torque_error[mask, j])

        axes[j, 0].set_ylabel(f"J{j + 1} (Nm)")
        axes[j, 0].grid(True)
        axes[j, 1].grid(True)

    axes[0, 0].set_title("Full torque error")
    axes[0, 1].set_title("Torque error after 0.5 s")
    axes[-1, 0].set_xlabel("Simulation time (s)")
    axes[-1, 1].set_xlabel("Simulation time (s)")

    fig.tight_layout()
    plt.show()

    # Report startup and later errors separately.
    if np.any(mask):
        print(
            "Torque RMSE after 0.5 s:",
            np.sqrt(np.mean(torque_error[mask]**2, axis=0))
        )

if __name__ == '__main__':
    main()
