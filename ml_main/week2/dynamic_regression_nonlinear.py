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
    a_GT = np.zeros((10 * num_joints, 1))
    for joint_index in range(num_joints):
        a_joint = np.asarray(
            dyn_model.pin_model.inertias[joint_index + 1].toDynamicParameters()
        ).reshape(-1, 1)
        a_GT[10 * joint_index : 10 * (joint_index + 1)] = a_joint

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
    max_time = 2  # seconds
    
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    # PD controller gains
    kp = 1000
    kd = 100

    
    # Initialize data storage
    tau_mes_all = []
    regressor_all = []

    # Initialize data storage
    q_mes_all, qd_mes_all, qdd_mes_all, q_d_all, qd_d_all = [], [], [], [], []
    
    # regressor all is a list of matrices
    regressor_all = np.empty((0, 70))
    tau_all = np.empty((0, 1))    

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

        
        # Compute regressor and store it
        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        qdd_mes_all.append(qdd_mes)
        q_d_all.append(q_d)
        qd_d_all.append(qd_d)
        cur_regressor = dyn_model.ComputeDynamicRegressor(q_mes,qd_mes, qdd_mes)
        regressor_all = np.vstack((regressor_all, cur_regressor))
        tau_all = np.vstack((tau_all, np.asarray(tau_cmd).reshape(-1, 1)))
        
        
        current_time += time_step
        # Optional: print current time
        print(f"Current time in seconds: {current_time:.2f}")

    # Remove transient regime from fitting data
    # Can you understand why we do this?
    fit_start     = 100
    regressor_fit = regressor_all[fit_start*7:, :]
    tau_fit       = tau_all[fit_start*7:]

    # Linear regression of the last joint parameters (Activity 1)
    a1      = np.linalg.pinv(regressor_fit[:, 60:]) @ (tau_fit - regressor_fit[:, :60] @ a_GT[:60])
    
    # Linear fitting of joint torques
    tau_est = regressor_all @ np.vstack((a_GT[:60], a1))
    error1  = tau_est - tau_all

    last_joint = num_joints - 1
    last_joint_torque = tau_all[last_joint::num_joints, 0]

    # TODO: Implement decision tree regression fitting and add results to plots below
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_squared_error

    # 每一行对应一个时间点：
    # 7 个关节角度 + 7 个角速度 + 7 个角加速度
    # X.shape = (N, 21)
    X = np.hstack((
        np.asarray(q_mes_all),
        np.asarray(qd_mes_all),
        np.asarray(qdd_mes_all),
    ))

    # 预测第七关节力矩，与现有线性模型使用相同标签 （目标输出）
    y = last_joint_torque

    # 跳过启动阶段，后 30% 数据用于测试
    split = fit_start + int(0.7 * (len(X) - fit_start))
    if not fit_start < split < len(X):
        raise ValueError("采样不足，请延长仿真时间")

    train = slice(fit_start, split)
    test = slice(split, None)

    tree = DecisionTreeRegressor(
        max_depth=8,       # 可以改成 2、4、8、16 比较
        random_state=42,
    )
    tree.fit(X[train], y[train])

    tau_tree = tree.predict(X)
    error_tree = tau_tree - y

    # 原有线性模型使用了整个拟合时段。
    # 为公平比较，重新只用前 70% 数据拟合线性模型。
    train_rows = slice(
        fit_start * num_joints,
        split * num_joints,
    )
    W_train = regressor_all[train_rows]
    a1 = np.linalg.pinv(W_train[:, 60:]) @ (
        tau_all[train_rows] - W_train[:, :60] @ a_GT[:60]
    )
    tau_est = regressor_all @ np.vstack((a_GT[:60], a1))
    error1 = tau_est - tau_all

    linear_last = tau_est[last_joint::num_joints, 0]

    print(
        "Linear test RMSE:",
        np.sqrt(mean_squared_error(y[test], linear_last[test])),
    )
    print(
        "Decision tree test RMSE:",
        np.sqrt(mean_squared_error(y[test], tau_tree[test])),
    )

    # 原有图在下面才创建，因此先保存 plt.subplots，
    # 在创建图时加入决策树曲线是不必要的；
    # 直接额外画一张完整的比较图。
    fig_tree, (ax_torque, ax_err) = plt.subplots(
        1, 2, figsize=(14, 6)
    )

    ax_torque.plot(y, label="Torque target", alpha=0.7)
    ax_torque.plot(linear_last, label="Linear regression")
    ax_torque.plot(tau_tree, label="Decision tree", alpha=0.8)

    ax_err.plot(linear_last - y, label="Linear error")
    ax_err.plot(error_tree, label="Decision tree error")

    ax_torque.set_title("Joint 7 torque prediction")
    ax_err.set_title("Prediction error")

    for ax in (ax_torque, ax_err):
        ax.axvline(
            split, color="black", linestyle="--",
            label="Test starts",
        )
        ax.set_xlabel("Time step")
        ax.set_ylabel("Torque (Nm)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig_tree.tight_layout()
    fig, (ax_pred, ax_error) = plt.subplots(
        1, 2, figsize=(14, 6), sharey=True
    )
    ax_pred.plot(
        tau_est[last_joint::num_joints],
        label="Linear regression",
        color="tab:green",
    )
    ax_pred.plot(
        last_joint_torque,
        label="Measured torque",
        alpha=0.7,
        color="tab:blue",
        zorder=2,
    )
    ax_error.plot(
        error1[last_joint::num_joints],
        label="Linear regression error",
    )

    ax_pred.set_title(f"Torque Estimates vs Measured Torque (Joint {last_joint + 1})")
    ax_error.set_title("Torque Fitting Error")
    for ax in (ax_pred, ax_error):
        ax.set_xlabel("Time step")
        ax.set_ylabel("Torque")
        ax.set_ylim(-250, 250)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()