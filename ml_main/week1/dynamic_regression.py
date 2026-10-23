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
        # ******************
        # 用本次 Step 的速度增量匹配本次实际施加的力矩。（重点）
        # ***************************
        qdd_mes = (sim.GetMotorVelocities(0) - qd_mes) / time_step
        
        regressor = dyn_model.ComputeDynamicRegressor(q_mes,qd_mes,qdd_mes)
        regressor_all.append(regressor.copy())
        tau_mes_all.append(tau_mes.copy())
       
        current_time += time_step
        # Optional: print current time
        # print(f"Current time in seconds: {current_time:.2f}")

    # TODO （1）After data collection, staGetMotorVelocitiesck all the regressor and all the torque and compute the parameters 'a'  using pseudoinverse for all the joint
    W = np.vstack(regressor_all)
    # 参考 week2：启动后 0.1 s 再辨识；后 30% 留作预测验证。
    fit_start = int(np.ceil(0.1 / time_step))
    sample_count = len(regressor_all)
    fit_end = fit_start + int(0.7 * (sample_count - fit_start))
    if not fit_start < fit_end < sample_count:
        raise ValueError("采样不足，请至少运行到启动阶段之后再退出")
    fit_rows = slice(fit_start * num_joints, fit_end * num_joints)

    # TODO （2）reshape the regressor and the torque vector to isolate the last joint and find the its dynamical parameters
    # 获取模型中七个运动连杆的惯性参数，每个连杆 10 个
    # 固定子连杆的惯性,会合并到对应运动连杆中
    a_true = np.concatenate([
        np.asarray(inertia.toDynamicParameters()).reshape(-1)
        for inertia in dyn_model.pin_model.inertias[1:]
    ])

    if W.shape[1] != 70 or a_true.size != 70:
        raise ValueError("预期模型包含七个运动连杆，每个连杆十个参数")

    # 回归矩阵使用 Pinocchio 关节顺序，力矩也转换为相同顺序
    T_pin = np.concatenate([
        dyn_model.ReoderJoints2PinVec(tau, "vel")
        for tau in tau_mes_all
    ])

    # 前六个连杆：60 个已知参数;第七个连杆：10 个未知参数
    a_known = a_true[:60]
    W_known = W[:, :60]
    W_last = W[:, 60:70]

    # 扣除前六个连杆的已知力矩贡献
    T_last = T_pin - W_known @ a_known

    # 保留全时段矩阵供绘图；估计和统计只使用训练时段。
    W_last_all = W_last
    W_last = W_last_all[fit_rows]
    T_last = T_last[fit_rows]

    # 用伪逆求末端连杆的 10 个参数(Core)
    a_last_est = np.linalg.pinv(W_last) @ T_last

    # 检查是否能够独立辨识全部 10 个参数
    last_rank = np.linalg.matrix_rank(W_last)

    # TODO （3）compute the metrics (R-squared adjusted etc...) for the linear model on a different file 
    # Compare estimated parameters with model truth.
    names = ["m", "mc_x", "mc_y", "mc_z", "I_xx", "I_xy",
                 "I_yy", "I_xz", "I_yz", "I_zz"]
    a_last_true = a_true[60:70]
    param_error = a_last_est - a_last_true
    print("\nParameter       True       Estimate     Abs. error")
    for name, truth, estimate, error in zip(
            names, a_last_true, a_last_est, param_error):
        print(f"{name:8s} {truth:12.6g} {estimate:12.6g} {abs(error):12.6g}")

    # Training-fit statistics for the no-intercept regression.
    residual = T_last - W_last @ a_last_est
    n = T_last.size
    dof = n - last_rank
    sse = residual @ residual
    sst = T_last @ T_last
    if dof > 0 and sst > 0:
        r2 = 1 - sse / sst
        adj_r2 = 1 - (sse / dof) / (sst / n)
        f_stat = ((sst - sse) / last_rank) / (sse / dof) if (
            last_rank > 0 and sse > 0) else np.nan
        print(f"\nUncentered R-squared: {r2:.6g}")
        print(f"Adjusted uncentered R-squared: {adj_r2:.6g}")
        print(f"F-statistic (no intercept): {f_stat:.6g}")

        # Classical OLS intervals assume independent, constant-variance errors.
        if last_rank == 10:
            try:
                from scipy.stats import t as student_t
            except ImportError:
                print("Confidence intervals require scipy.")
            else:
                W_pinv = np.linalg.pinv(W_last)
                covariance = (sse / dof) * (W_pinv @ W_pinv.T)
                margin = student_t.ppf(0.975, dof) * np.sqrt(
                    np.maximum(np.diag(covariance), 0))
                print("\nParameter     95% confidence interval")
                for name, estimate, delta in zip(names, a_last_est, margin):
                    print(f"{name:8s} [{estimate - delta:.6g}, {estimate + delta:.6g}]")
        else:
            print("Parameter confidence intervals skipped: rank deficient.")
    else:
        print("Insufficient data or zero response energy for fit statistics.")

    # Reconstruct all joint torques in Pinocchio order.
    torque_pred = (W_known @ a_known + W_last_all @ a_last_est).reshape(-1, 7)
    torque_true = T_pin.reshape(-1, 7)
    torque_error = torque_pred - torque_true
    print("\nTorque RMSE per joint:", np.sqrt(np.mean(torque_error**2, axis=0)))

    # TODO plot the torque prediction error for each joint (optional)
    # 时间留出验证：这些样本没有用于拟合，不等同于独立第二次仿真。
    validation_error = torque_error[fit_end:]
    validation_true = torque_true[fit_end:]
    rmse = np.sqrt(np.mean(validation_error**2, axis=0))
    torque_rms = np.sqrt(np.mean(validation_true**2, axis=0))
    relative_rmse = np.divide(rmse, torque_rms,
                              out=np.full_like(rmse, np.nan), where=torque_rms > 0)
    overall_relative = np.linalg.norm(validation_error) / np.linalg.norm(validation_true)
    print(f"Fit interval: {fit_start * time_step:.3f}–{fit_end * time_step:.3f} s")
    print("Held-out torque RMSE per joint (Nm):", rmse)
    print("Held-out relative RMSE per joint (%):", 100 * relative_rmse)
    print(f"Held-out overall relative RMSE: {100 * overall_relative:.6g}%")
    print("Mass relative error (%):", 100 * abs(param_error[0]) / a_last_true[0])
    print("Startup command is unchanged; excluded from fitting, retained in full-error plot.")
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
