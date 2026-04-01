import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure local package imports work
sys.path.append(os.path.dirname(__file__))

import ins_sig_gen
import ins_ekf


def ensure_out_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def run_demo(out_dir):
    ensure_out_dir(out_dir)

    # Demo parameters (small example)
    speed_changes = [ (5.0, 5.0), (-2.0, 5.0) ]
    rot_changes_x = [ (0.0, 10.0) ]
    rot_changes_y = [ (0.0, 10.0) ]
    rot_changes_z = [ (0.0, 10.0) ]
    attitude0 = np.matrix([[0.0],[0.0],[0.0]])

    imu_period = 0.01
    acc_bias0 = np.matrix([[0.0],[0.0],[0.0]])
    acc_w_std = 0.05
    gyro_bias0 = np.matrix([[0.0],[0.0],[0.0]])
    gyro_w_std = 0.01

    gnss_period = 1.0
    gnss_speed_w_std = 0.1
    gnss_dist_w_std = 1.0

    # Generate signals
    ( imu_time, imu_accel, imu_gyro,
      gnss_time, gnss_dist,
      imu_accel_bias, imu_gyro_bias, global_attitude,
      global_accel, global_speed, global_speed_norm, global_dist ) = ins_sig_gen.generate_signals(
        speed_changes,
        rot_changes_x, rot_changes_y, rot_changes_z,
        attitude0,
        imu_period,
        acc_bias0,
        acc_w_std,
        gyro_bias0,
        gyro_w_std,
        gnss_period,
        gnss_speed_w_std,
        gnss_dist_w_std
    )

    # EKF parameters
    accel_bias_std = 0.1
    accel_w_std = 0.05
    gyro_bias_std = 0.01
    gyro_w_std = 0.01
    attitude0_std = 0.01

    # Run filter
    # ins_ext_kfilter signature still accepts a gnss_speed argument but the EKF
    # measurement model no longer uses GNSS speed. Pass an empty list here.
    state_list, var_list = ins_ekf.ins_ext_kfilter(
        imu_time, imu_accel, imu_gyro,
        accel_bias_std, accel_w_std, gyro_bias_std, gyro_w_std,
        attitude0, attitude0_std, gyro_bias0,
        gnss_time, [], gnss_dist, gnss_speed_w_std, gnss_dist_w_std
    )

    # Extract position and speed for plotting
    pos_x = [ s.item((0,0)) for s in state_list ]
    pos_y = [ s.item((1,0)) for s in state_list ]
    pos_z = [ s.item((2,0)) for s in state_list ]
    spd_x = [ s.item((3,0)) for s in state_list ]
    spd_y = [ s.item((4,0)) for s in state_list ]
    spd_z = [ s.item((5,0)) for s in state_list ]

    # Extract state covariance (std dev) for positions from var_list
    pos_std_x = [ np.sqrt(P.item((0,0))) for P in var_list ]
    pos_std_y = [ np.sqrt(P.item((1,1))) for P in var_list ]
    pos_std_z = [ np.sqrt(P.item((2,2))) for P in var_list ]

    t = imu_time

    # Plot positions
    plt.figure()
    plt.plot(t, pos_x, label='pos_x')
    plt.fill_between(t, np.array(pos_x) - np.array(pos_std_x), np.array(pos_x) + np.array(pos_std_x), color='C0', alpha=0.2)
    plt.plot(t, pos_y, label='pos_y')
    plt.fill_between(t, np.array(pos_y) - np.array(pos_std_y), np.array(pos_y) + np.array(pos_std_y), color='C1', alpha=0.2)
    plt.plot(t, pos_z, label='pos_z')
    plt.fill_between(t, np.array(pos_z) - np.array(pos_std_z), np.array(pos_z) + np.array(pos_std_z), color='C2', alpha=0.2)
    # Reference (true) trajectory from global_dist
    ref_x = [ d.item((0,0)) for d in global_dist ]
    ref_y = [ d.item((1,0)) for d in global_dist ]
    ref_z = [ d.item((2,0)) for d in global_dist ]
    plt.plot(t, ref_x, '--', label='ref_x')
    plt.plot(t, ref_y, '--', label='ref_y')
    plt.plot(t, ref_z, '--', label='ref_z')
    # GNSS measurements (sparser)
    gnss_x = [ g.item((0,0)) for g in gnss_dist ]
    gnss_y = [ g.item((1,0)) for g in gnss_dist ]
    gnss_z = [ g.item((2,0)) for g in gnss_dist ]
    plt.scatter(gnss_time, gnss_x, marker='x', color='k', label='gnss_x')
    plt.scatter(gnss_time, gnss_y, marker='x', color='gray', label='gnss_y')
    plt.scatter(gnss_time, gnss_z, marker='x', color='brown', label='gnss_z')
    plt.xlabel('time (s)')
    plt.ylabel('position (m)')
    plt.legend()
    pos_plot = os.path.join(out_dir, 'positions.png')
    plt.savefig(pos_plot)
    plt.close()

    # Plot speeds
    plt.figure()
    plt.plot(t, spd_x, label='spd_x')
    plt.plot(t, spd_y, label='spd_y')
    plt.plot(t, spd_z, label='spd_z')
    plt.xlabel('time (s)')
    plt.ylabel('speed (m/s)')
    plt.legend()
    spd_plot = os.path.join(out_dir, 'speeds.png')
    plt.savefig(spd_plot)
    plt.close()

    # 2D trajectory plot (estimated vs reference vs GNSS)
    traj_plot = os.path.join(out_dir, 'trajectory.png')
    plt.figure()
    plt.plot(pos_x, pos_y, label='est (x,y)')
    # optionally plot covariance ellipses could be added later
    plt.plot(ref_x, ref_y, '--', label='ref (x,y)')
    plt.scatter(gnss_x, gnss_y, marker='x', color='k', label='gnss (x,y)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.axis('equal')
    plt.savefig(traj_plot)
    plt.close()

    # --- EKF vs GNSS comparisons ---
    # Map GNSS times to nearest IMU indices and get EKF states at those times
    ekf_at_gnss = []
    ekf_pos_x = []
    ekf_pos_y = []
    ekf_pos_z = []
    for gt in gnss_time:
        imu_idx = min(range(len(t)), key=lambda k: abs(t[k] - gt))
        s = state_list[imu_idx]
        ekf_at_gnss.append(s)
        ekf_pos_x.append(s.item((0,0)))
        ekf_pos_y.append(s.item((1,0)))
        ekf_pos_z.append(s.item((2,0)))

    # Compute errors (EKF - GNSS) at GNSS timestamps
    err_x = np.array(ekf_pos_x) - np.array(gnss_x)
    err_y = np.array(ekf_pos_y) - np.array(gnss_y)
    err_z = np.array(ekf_pos_z) - np.array(gnss_z)
    err_norm = np.sqrt(err_x**2 + err_y**2 + err_z**2)

    # Error statistics
    rmse_x = float(np.sqrt(np.mean(err_x**2)))
    rmse_y = float(np.sqrt(np.mean(err_y**2)))
    rmse_z = float(np.sqrt(np.mean(err_z**2)))
    rmse_norm = float(np.sqrt(np.mean(err_norm**2)))
    mean_abs_x = float(np.mean(np.abs(err_x)))
    mean_abs_y = float(np.mean(np.abs(err_y)))
    mean_abs_z = float(np.mean(np.abs(err_z)))
    mean_abs_norm = float(np.mean(err_norm))

    # Component-wise comparison plot (EKF @ GNSS times vs GNSS)
    comp_plot = os.path.join(out_dir, 'ekf_vs_gnss_components.png')
    plt.figure()
    plt.plot(gnss_time, ekf_pos_x, '-o', label='ekf_x')
    plt.plot(gnss_time, ekf_pos_y, '-o', label='ekf_y')
    plt.plot(gnss_time, ekf_pos_z, '-o', label='ekf_z')
    plt.scatter(gnss_time, gnss_x, marker='x', color='k', label='gnss_x')
    plt.scatter(gnss_time, gnss_y, marker='x', color='gray', label='gnss_y')
    plt.scatter(gnss_time, gnss_z, marker='x', color='brown', label='gnss_z')
    plt.xlabel('time (s)')
    plt.ylabel('position (m)')
    plt.legend()
    plt.savefig(comp_plot)
    plt.close()

    # Error norm plot
    err_plot = os.path.join(out_dir, 'ekf_gnss_error.png')
    plt.figure()
    plt.plot(gnss_time, err_norm, '-o', label='error norm')
    plt.xlabel('time (s)')
    plt.ylabel('position error (m)')
    plt.legend()
    plt.savefig(err_plot)
    plt.close()

    # Final state table
    final_state = state_list[-1]
    final_vals = [ final_state.item((i,0)) for i in range(final_state.shape[0]) ]

    # Write HTML
    html_path = os.path.join(out_dir, 'results.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('<!doctype html>\n<html><head><meta charset="utf-8"><title>INS EKF Results</title></head><body>\n')
        f.write('<h1>INS EKF Demo Results</h1>\n')
        f.write('<h2>Plots</h2>\n')
        f.write(f'<div><h3>Positions (estimated vs reference)</h3><img src="{os.path.basename(pos_plot)}" style="max-width:100%;height:auto"></div>\n')
        f.write(f'<div><h3>Speeds</h3><img src="{os.path.basename(spd_plot)}" style="max-width:100%;height:auto"></div>\n')
        f.write(f'<div><h3>2D Trajectory (X vs Y)</h3><img src="{os.path.basename(traj_plot)}" style="max-width:100%;height:auto"></div>\n')
        f.write('<h2>Final filter state</h2>\n')
        f.write('<table border="1" cellspacing="0" cellpadding="4">\n')
        f.write('<tr><th>Index</th><th>Name</th><th>Value</th></tr>\n')
        names = [ 'pos_x','pos_y','pos_z','spd_x','spd_y','spd_z', 'accel_bias_x','accel_bias_y','accel_bias_z', 'w_bias_x','w_bias_y','w_bias_z', 'psi','theta','gamma' ]
        for i, name in enumerate(names):
            f.write(f'<tr><td>{i}</td><td>{name}</td><td>{final_vals[i]:.6f}</td></tr>\n')
        f.write('</table>\n')
        # GNSS measurement table
        f.write('<h2>GNSS Measurements</h2>\n')
        f.write('<table border="1" cellspacing="0" cellpadding="4">\n')
        f.write('<tr><th>Index</th><th>Time (s)</th><th>X (m)</th><th>Y (m)</th><th>Z (m)</th></tr>\n')
        for i, (gt, gd) in enumerate(zip(gnss_time, gnss_dist)):
            f.write(f'<tr><td>{i}</td><td>{gt:.3f}</td><td>{gd.item((0,0)):.3f}</td><td>{gd.item((1,0)):.3f}</td><td>{gd.item((2,0)):.3f}</td></tr>\n')
        f.write('</table>\n')

        # EKF states at GNSS times
        f.write('<h2>EKF state (at GNSS timestamps)</h2>\n')
        f.write('<table border="1" cellspacing="0" cellpadding="4">\n')
        hdr = '<tr><th>Index</th><th>Time (s)</th><th>pos_x</th><th>pos_y</th><th>pos_z</th><th>spd_x</th><th>spd_y</th><th>spd_z</th><th>psi</th><th>theta</th><th>gamma</th></tr>\n'
        f.write(hdr)
        # Map GNSS times to nearest IMU index and print state
        for i, gt in enumerate(gnss_time):
            # find nearest imu_time index
            imu_idx = min(range(len(t)), key=lambda k: abs(t[k] - gt))
            s = state_list[imu_idx]
            f.write(f'<tr><td>{i}</td><td>{gt:.3f}</td><td>{s.item((0,0)):.6f}</td><td>{s.item((1,0)):.6f}</td><td>{s.item((2,0)):.6f}</td><td>{s.item((3,0)):.6f}</td><td>{s.item((4,0)):.6f}</td><td>{s.item((5,0)):.6f}</td><td>{s.item((12,0)):.6f}</td><td>{s.item((13,0)):.6f}</td><td>{s.item((14,0)):.6f}</td></tr>\n')
        f.write('</table>\n')
        f.write('<p>Generated by <code>run_and_export.py</code></p>\n')
        # EKF vs GNSS comparison summary
        f.write('<h2>EKF vs GNSS error summary</h2>\n')
        f.write('<table border="1" cellspacing="0" cellpadding="4">\n')
        f.write('<tr><th>Metric</th><th>X (m)</th><th>Y (m)</th><th>Z (m)</th><th>Norm (m)</th></tr>\n')
        f.write(f'<tr><td>RMSE</td><td>{rmse_x:.4f}</td><td>{rmse_y:.4f}</td><td>{rmse_z:.4f}</td><td>{rmse_norm:.4f}</td></tr>\n')
        f.write(f'<tr><td>Mean abs error</td><td>{mean_abs_x:.4f}</td><td>{mean_abs_y:.4f}</td><td>{mean_abs_z:.4f}</td><td>{mean_abs_norm:.4f}</td></tr>\n')
        f.write('</table>\n')
        # Include comparison plots
        f.write('<h3>EKF vs GNSS component-wise</h3>\n')
        f.write(f'<img src="{os.path.basename(comp_plot)}" style="max-width:100%;height:auto">\n')
        f.write('<h3>EKF-GNSS error norm</h3>\n')
        f.write(f'<img src="{os.path.basename(err_plot)}" style="max-width:100%;height:auto">\n')
        f.write('</body></html>')

    print('Report generated at', html_path)


if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(__file__), 'out')
    run_demo(out_dir)
