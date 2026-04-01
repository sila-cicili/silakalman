# ============================================================
# INS + Extended Kalman Filter — Google Colab notebook script
#
# Kullanım:
#   1. Bu dosyayı Google Colab'a yükleyin ya da doğrudan
#      Colab'ın kod hücrelerine kopyalayın.
#   2. Hücreyi çalıştırın; dosya yükleme widget'ı açılır.
#   3. imu-data.xlsx ve gps-data.xlsx dosyalarını seçin.
#   4. Sonuçlar otomatik olarak plotlanır ve
#      ekf_results.csv / ekf_plots.png olarak indirilir.
#
# Excel sütun düzeni:
#   imu-data.xlsx : timestamp | accel_x | accel_y | accel_z
#                             | gyro_x  | gyro_y  | gyro_z
#   gps-data.xlsx : timestamp | x       | y       | z   (metre)
# ============================================================

import math
import io
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# 1.  Ortam tespiti (Colab mu, yerel mi?)
# ─────────────────────────────────────────────────────────────
def _is_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False

IS_COLAB = _is_colab()

# ─────────────────────────────────────────────────────────────
# 2.  Yardımcı fonksiyonlar  (utils.py içeriği)
# ─────────────────────────────────────────────────────────────

def get_dcm(attitude_euler):
    """Euler açılarından (psi, theta, gamma) DCM hesapla."""
    cos_psi   = math.cos(attitude_euler.item((0, 0)))
    sin_psi   = math.sin(attitude_euler.item((0, 0)))
    cos_theta = math.cos(attitude_euler.item((1, 0)))
    sin_theta = math.sin(attitude_euler.item((1, 0)))
    cos_gamma = math.cos(attitude_euler.item((2, 0)))
    sin_gamma = math.sin(attitude_euler.item((2, 0)))

    return np.matrix([
        [ cos_theta*cos_psi,
         -cos_gamma*cos_psi*sin_theta + sin_gamma*sin_psi,
          sin_gamma*cos_psi*sin_theta + cos_gamma*sin_psi ],
        [ sin_theta,
          cos_gamma*cos_theta,
         -sin_gamma*cos_theta ],
        [-cos_theta*sin_psi,
          cos_gamma*sin_psi*sin_theta + sin_gamma*cos_psi,
         -sin_gamma*sin_psi*sin_theta + cos_gamma*cos_psi ]
    ])


def get_inv_dcm(attitude_euler):
    return get_dcm(attitude_euler).transpose()


def attitude_euler_update(att_euler_prev, rot_speed, period):
    """Bir adım için Euler açısı güncellemesi."""
    cos_theta = np.cos(att_euler_prev.item((1, 0)))
    sin_theta = np.sin(att_euler_prev.item((1, 0)))
    cos_gamma = np.cos(att_euler_prev.item((2, 0)))
    sin_gamma = np.sin(att_euler_prev.item((2, 0)))

    wx = rot_speed.item((0, 0))
    wy = rot_speed.item((1, 0))
    wz = rot_speed.item((2, 0))

    tmp = (1.0 / cos_theta) * (wy * cos_gamma - wz * sin_gamma)
    return np.matrix([
        [att_euler_prev.item((0, 0)) + tmp * period],
        [att_euler_prev.item((1, 0)) + (wy * sin_gamma + wz * cos_gamma) * period],
        [att_euler_prev.item((2, 0)) + (wx - sin_theta * tmp) * period]
    ])


# ─────────────────────────────────────────────────────────────
# 3.  EKF fonksiyonları  (ins_ekf.py içeriği)
# ─────────────────────────────────────────────────────────────

def _exec_f_func(x_vect, u_vect, period):
    """Durum tahmin fonksiyonu f(x, u)."""
    pos_gx       = x_vect.item((0,  0))
    pos_gy       = x_vect.item((1,  0))
    pos_gz       = x_vect.item((2,  0))
    speed_gx     = x_vect.item((3,  0))
    speed_gy     = x_vect.item((4,  0))
    speed_gz     = x_vect.item((5,  0))
    accel_bias_x = x_vect.item((6,  0))
    accel_bias_y = x_vect.item((7,  0))
    accel_bias_z = x_vect.item((8,  0))
    w_bias_x     = x_vect.item((9,  0))
    w_bias_y     = x_vect.item((10, 0))
    w_bias_z     = x_vect.item((11, 0))
    psi          = x_vect.item((12, 0))
    theta        = x_vect.item((13, 0))
    gamma        = x_vect.item((14, 0))

    est_accel_ix = u_vect.item((0, 0)) - accel_bias_x
    est_accel_iy = u_vect.item((1, 0)) - accel_bias_y
    est_accel_iz = u_vect.item((2, 0)) - accel_bias_z
    est_wx       = u_vect.item((3, 0)) - w_bias_x
    est_wy       = u_vect.item((4, 0)) - w_bias_y
    est_wz       = u_vect.item((5, 0)) - w_bias_z

    accel_g = get_dcm(np.matrix([[psi], [theta], [gamma]])) * \
              np.matrix([[est_accel_ix], [est_accel_iy], [est_accel_iz]])
    accel_g = accel_g - np.matrix([[0], [9.81], [0]])
    accel_gx = accel_g.item((0, 0))
    accel_gy = accel_g.item((1, 0))
    accel_gz = accel_g.item((2, 0))

    attitude_new = attitude_euler_update(
        np.matrix([[psi], [theta], [gamma]]),
        np.matrix([[est_wx], [est_wy], [est_wz]]),
        period
    )

    dt2 = 0.5 * period ** 2
    dt  = period

    return np.matrix([
        [pos_gx   + speed_gx * dt + accel_gx * dt2],
        [pos_gy   + speed_gy * dt + accel_gy * dt2],
        [pos_gz   + speed_gz * dt + accel_gz * dt2],
        [speed_gx + accel_gx * dt],
        [speed_gy + accel_gy * dt],
        [speed_gz + accel_gz * dt],
        [accel_bias_x],
        [accel_bias_y],
        [accel_bias_z],
        [w_bias_x],
        [w_bias_y],
        [w_bias_z],
        [attitude_new.item((0, 0))],
        [attitude_new.item((1, 0))],
        [attitude_new.item((2, 0))]
    ])


def _get_F_matrix(x_vect, u_vect, period):
    """Durum Jacobian matrisi F."""
    accel_bias_x = x_vect.item((6,  0))
    accel_bias_y = x_vect.item((7,  0))
    accel_bias_z = x_vect.item((8,  0))
    w_bias_x     = x_vect.item((9,  0))
    w_bias_y     = x_vect.item((10, 0))
    w_bias_z     = x_vect.item((11, 0))
    psi          = x_vect.item((12, 0))
    theta        = x_vect.item((13, 0))
    gamma        = x_vect.item((14, 0))

    dcm = get_dcm(np.matrix([[psi], [theta], [gamma]]))

    cos_psi   = math.cos(psi)
    sin_psi   = math.sin(psi)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    cos_gamma = math.cos(gamma)
    sin_gamma = math.sin(gamma)

    est_accel_ix = u_vect.item((0, 0)) - accel_bias_x
    est_accel_iy = u_vect.item((1, 0)) - accel_bias_y
    est_accel_iz = u_vect.item((2, 0)) - accel_bias_z
    est_wx = u_vect.item((3, 0)) - w_bias_x
    est_wy = u_vect.item((4, 0)) - w_bias_y
    est_wz = u_vect.item((5, 0)) - w_bias_z

    # DCM türevleri — psi
    d_c11_d_psi =  cos_theta * (-sin_psi)
    d_c12_d_psi = -cos_gamma * (-sin_psi) * sin_theta + sin_gamma *  cos_psi
    d_c13_d_psi =  sin_gamma * (-sin_psi) * sin_theta + cos_gamma *  cos_psi
    d_c21_d_psi =  0
    d_c22_d_psi =  0
    d_c23_d_psi =  0
    d_c31_d_psi = -cos_theta *  cos_psi
    d_c32_d_psi =  cos_gamma *  cos_psi * sin_theta + sin_gamma * (-sin_psi)
    d_c33_d_psi = -sin_gamma *  cos_psi * sin_theta + cos_gamma * (-sin_psi)

    # DCM türevleri — theta
    d_c11_d_theta = (-sin_theta) *  cos_psi
    d_c12_d_theta = -cos_gamma   *  cos_psi * cos_theta
    d_c13_d_theta =  sin_gamma   *  cos_psi * cos_theta
    d_c21_d_theta =  cos_theta
    d_c22_d_theta =  cos_gamma   * (-sin_theta)
    d_c23_d_theta = -sin_gamma   * (-sin_theta)
    d_c31_d_theta = -(-sin_theta)*  sin_psi
    d_c32_d_theta =  cos_gamma   *  sin_psi * cos_theta
    d_c33_d_theta = -sin_gamma   *  sin_psi * cos_theta

    # DCM türevleri — gamma
    d_c11_d_gamma =  0
    d_c12_d_gamma = -(-sin_gamma) * cos_psi * sin_theta + cos_gamma  * sin_psi
    d_c13_d_gamma =  cos_gamma    * cos_psi * sin_theta + (-sin_gamma)* sin_psi
    d_c21_d_gamma =  0
    d_c22_d_gamma =  (-sin_gamma) * cos_theta
    d_c23_d_gamma = -cos_gamma    * cos_theta
    d_c31_d_gamma =  0
    d_c32_d_gamma =  (-sin_gamma) * sin_psi * sin_theta + cos_gamma  * cos_psi
    d_c33_d_gamma = -cos_gamma    * sin_psi * sin_theta + (-sin_gamma)* cos_psi

    # accel_g türevleri
    d_agx_d_abx = -dcm.item((0, 0)); d_agx_d_aby = -dcm.item((0, 1)); d_agx_d_abz = -dcm.item((0, 2))
    d_agx_d_psi   = d_c11_d_psi   * est_accel_ix + d_c12_d_psi   * est_accel_iy + d_c13_d_psi   * est_accel_iz
    d_agx_d_theta = d_c11_d_theta * est_accel_ix + d_c12_d_theta * est_accel_iy + d_c13_d_theta * est_accel_iz
    d_agx_d_gamma = d_c11_d_gamma * est_accel_ix + d_c12_d_gamma * est_accel_iy + d_c13_d_gamma * est_accel_iz

    d_agy_d_abx = -dcm.item((1, 0)); d_agy_d_aby = -dcm.item((1, 1)); d_agy_d_abz = -dcm.item((1, 2))
    d_agy_d_psi   = d_c21_d_psi   * est_accel_ix + d_c22_d_psi   * est_accel_iy + d_c23_d_psi   * est_accel_iz
    d_agy_d_theta = d_c21_d_theta * est_accel_ix + d_c22_d_theta * est_accel_iy + d_c23_d_theta * est_accel_iz
    d_agy_d_gamma = d_c21_d_gamma * est_accel_ix + d_c22_d_gamma * est_accel_iy + d_c23_d_gamma * est_accel_iz

    d_agz_d_abx = -dcm.item((2, 0)); d_agz_d_aby = -dcm.item((2, 1)); d_agz_d_abz = -dcm.item((2, 2))
    d_agz_d_psi   = d_c31_d_psi   * est_accel_ix + d_c32_d_psi   * est_accel_iy + d_c33_d_psi   * est_accel_iz
    d_agz_d_theta = d_c31_d_theta * est_accel_ix + d_c32_d_theta * est_accel_iy + d_c33_d_theta * est_accel_iz
    d_agz_d_gamma = d_c31_d_gamma * est_accel_ix + d_c32_d_gamma * est_accel_iy + d_c33_d_gamma * est_accel_iz

    # Açı türevleri
    d_psi_d_wbx   = 0
    d_psi_d_wby   =  1.0 / cos_theta * (-cos_gamma)     * period
    d_psi_d_wbz   =  1.0 / cos_theta * sin_gamma        * period
    d_psi_d_psi   =  1
    d_psi_d_theta =  sin_theta / (cos_theta ** 2) * (est_wy * cos_gamma - est_wz * sin_gamma) * period
    d_psi_d_gamma =  1.0 / cos_theta * (est_wy * (-sin_gamma) - est_wz * cos_gamma) * period

    d_theta_d_wbx   =  0
    d_theta_d_wby   = -sin_gamma * period
    d_theta_d_wbz   = -cos_gamma * period
    d_theta_d_psi   =  0
    d_theta_d_theta =  1
    d_theta_d_gamma =  (est_wy * cos_gamma + est_wz * (-sin_gamma)) * period

    d_gamma_d_wbx   = -period
    d_gamma_d_wby   = -sin_theta / cos_theta * (-cos_gamma) * period
    d_gamma_d_wbz   = -sin_theta / cos_theta * sin_gamma    * period
    d_gamma_d_psi   =  0
    d_gamma_d_theta = -1.0 / (cos_theta ** 2) * (est_wy * cos_gamma - est_wz * sin_gamma) * period
    d_gamma_d_gamma =  1 - sin_theta / cos_theta * (est_wy * (-sin_gamma) - est_wz * cos_gamma) * period

    dt2 = 0.5 * period ** 2
    dt  = period

    F = np.matrix([
    #   rx  ry  rz   vx  vy  vz   abx                aby                abz                wbx            wby            wbz            psi                theta              gamma
        [1,  0,  0,   dt, 0,  0,   d_agx_d_abx*dt2,   d_agx_d_aby*dt2,   d_agx_d_abz*dt2,   0,             0,             0,             d_agx_d_psi*dt2,   d_agx_d_theta*dt2, d_agx_d_gamma*dt2],
        [0,  1,  0,   0,  dt, 0,   d_agy_d_abx*dt2,   d_agy_d_aby*dt2,   d_agy_d_abz*dt2,   0,             0,             0,             d_agy_d_psi*dt2,   d_agy_d_theta*dt2, d_agy_d_gamma*dt2],
        [0,  0,  1,   0,  0,  dt,  d_agz_d_abx*dt2,   d_agz_d_aby*dt2,   d_agz_d_abz*dt2,   0,             0,             0,             d_agz_d_psi*dt2,   d_agz_d_theta*dt2, d_agz_d_gamma*dt2],
        [0,  0,  0,   1,  0,  0,   d_agx_d_abx*dt,    d_agx_d_aby*dt,    d_agx_d_abz*dt,    0,             0,             0,             d_agx_d_psi*dt,    d_agx_d_theta*dt,  d_agx_d_gamma*dt ],
        [0,  0,  0,   0,  1,  0,   d_agy_d_abx*dt,    d_agy_d_aby*dt,    d_agy_d_abz*dt,    0,             0,             0,             d_agy_d_psi*dt,    d_agy_d_theta*dt,  d_agy_d_gamma*dt ],
        [0,  0,  0,   0,  0,  1,   d_agz_d_abx*dt,    d_agz_d_aby*dt,    d_agz_d_abz*dt,    0,             0,             0,             d_agz_d_psi*dt,    d_agz_d_theta*dt,  d_agz_d_gamma*dt ],
        [0,  0,  0,   0,  0,  0,   1,                  0,                  0,                  0,             0,             0,             0,                  0,                 0                ],
        [0,  0,  0,   0,  0,  0,   0,                  1,                  0,                  0,             0,             0,             0,                  0,                 0                ],
        [0,  0,  0,   0,  0,  0,   0,                  0,                  1,                  0,             0,             0,             0,                  0,                 0                ],
        [0,  0,  0,   0,  0,  0,   0,                  0,                  0,                  1,             0,             0,             0,                  0,                 0                ],
        [0,  0,  0,   0,  0,  0,   0,                  0,                  0,                  0,             1,             0,             0,                  0,                 0                ],
        [0,  0,  0,   0,  0,  0,   0,                  0,                  0,                  0,             0,             1,             0,                  0,                 0                ],
        [0,  0,  0,   0,  0,  0,   0,                  0,                  0,                  d_psi_d_wbx,   d_psi_d_wby,   d_psi_d_wbz,   d_psi_d_psi,        d_psi_d_theta,     d_psi_d_gamma  ],
        [0,  0,  0,   0,  0,  0,   0,                  0,                  0,                  d_theta_d_wbx, d_theta_d_wby, d_theta_d_wbz, d_theta_d_psi,      d_theta_d_theta,   d_theta_d_gamma],
        [0,  0,  0,   0,  0,  0,   0,                  0,                  0,                  d_gamma_d_wbx, d_gamma_d_wby, d_gamma_d_wbz, d_gamma_d_psi,      d_gamma_d_theta,   d_gamma_d_gamma],
    ])

    return F


def _exec_h_func(x_vect, _period):
    """Ölçüm fonksiyonu h(x) — yalnızca GPS konumu."""
    return np.matrix([
        [x_vect.item((0, 0))],
        [x_vect.item((1, 0))],
        [x_vect.item((2, 0))]
    ])


def _get_H_matrix(_x_vect, _period):
    """Ölçüm Jacobian'ı H."""
    return np.matrix([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])


def ins_ext_kfilter(imu_time, imu_accel, imu_gyro,
                    accel_bias_std, accel_w_std, gyro_bias_std, gyro_w_std,
                    attitude0, attitude0_std, gyro_bias0,
                    gnss_time, gnss_speed, gnss_dist,
                    gnss_speed_std, gnss_dist_std):
    """Extended Kalman Filter — INS + GPS füzyonu."""
    state_list = []
    var_list   = []

    imu_dt = imu_time[1] - imu_time[0]

    X = np.matrix([
        [0.0], [0.0], [0.0],        # konum
        [0.0], [0.0], [0.0],        # hız
        [0.0], [0.0], [0.0],        # ivme biası
        [gyro_bias0.item((0, 0))],
        [gyro_bias0.item((1, 0))],
        [gyro_bias0.item((2, 0))],  # jiroskop biası
        [attitude0.item((0, 0))],
        [attitude0.item((1, 0))],
        [attitude0.item((2, 0))]    # Euler açıları
    ])

    pos_q_std   = accel_w_std * imu_dt ** 2 / 2
    speed_q_std = accel_w_std * imu_dt
    angle_q_std = gyro_w_std  * imu_dt

    Q = np.matrix([
        [pos_q_std**2,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, pos_q_std**2,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, pos_q_std**2,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, speed_q_std**2,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, speed_q_std**2,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, speed_q_std**2,   0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, angle_q_std**2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, angle_q_std**2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, angle_q_std**2]
    ])

    R = np.matrix([
        [gnss_dist_std**2, 0,                0              ],
        [0,                gnss_dist_std**2, 0              ],
        [0,                0,                gnss_dist_std**2]
    ])

    P = np.matrix([
        [0, 0, 0, 0, 0, 0, 0,                  0,                  0,                  0,              0,              0,              0,                0,                0              ],
        [0, 0, 0, 0, 0, 0, 0,                  0,                  0,                  0,              0,              0,              0,                0,                0              ],
        [0, 0, 0, 0, 0, 0, 0,                  0,                  0,                  0,              0,              0,              0,                0,                0              ],
        [0, 0, 0, 0, 0, 0, 0,                  0,                  0,                  0,              0,              0,              0,                0,                0              ],
        [0, 0, 0, 0, 0, 0, 0,                  0,                  0,                  0,              0,              0,              0,                0,                0              ],
        [0, 0, 0, 0, 0, 0, 0,                  0,                  0,                  0,              0,              0,              0,                0,                0              ],
        [0, 0, 0, 0, 0, 0, accel_bias_std**2,  0,                  0,                  0,              0,              0,              0,                0,                0              ],
        [0, 0, 0, 0, 0, 0, 0,                  accel_bias_std**2,  0,                  0,              0,              0,              0,                0,                0              ],
        [0, 0, 0, 0, 0, 0, 0,                  0,                  accel_bias_std**2,  0,              0,              0,              0,                0,                0              ],
        [0, 0, 0, 0, 0, 0, 0,                  0,                  0,                  gyro_w_std**2,  0,              0,              0,                0,                0              ],
        [0, 0, 0, 0, 0, 0, 0,                  0,                  0,                  0,              gyro_w_std**2,  0,              0,                0,                0              ],
        [0, 0, 0, 0, 0, 0, 0,                  0,                  0,                  0,              0,              gyro_w_std**2,  0,                0,                0              ],
        [0, 0, 0, 0, 0, 0, 0,                  0,                  0,                  0,              0,              0,              attitude0_std**2, 0,                0              ],
        [0, 0, 0, 0, 0, 0, 0,                  0,                  0,                  0,              0,              0,              0,                attitude0_std**2, 0              ],
        [0, 0, 0, 0, 0, 0, 0,                  0,                  0,                  0,              0,              0,              0,                0,                attitude0_std**2]
    ])

    gnss_i = 0
    for t, accel, gyro in zip(imu_time, imu_accel, imu_gyro):
        U = np.matrix([
            [accel.item((0, 0))],
            [accel.item((1, 0))],
            [accel.item((2, 0))],
            [gyro.item((0, 0))],
            [gyro.item((1, 0))],
            [gyro.item((2, 0))]
        ])

        F = _get_F_matrix(X, U, imu_dt)
        X = _exec_f_func(X, U, imu_dt)
        P = F * P * F.transpose() + Q

        if gnss_i < len(gnss_time) and t > gnss_time[gnss_i]:
            Z = np.matrix([
                [gnss_dist[gnss_i].item((0, 0))],
                [gnss_dist[gnss_i].item((1, 0))],
                [gnss_dist[gnss_i].item((2, 0))]
            ])
            H = _get_H_matrix(X, imu_dt)
            K = P * H.transpose() * np.linalg.inv(H * P * H.transpose() + R)
            X = X + K * (Z - _exec_h_func(X, imu_dt))
            P = P - K * H * P
            gnss_i += 1

        state_list.append(X.copy())
        var_list.append(P.copy())

    return state_list, var_list


# ─────────────────────────────────────────────────────────────
# 4.  Excel veri yükleyici
# ─────────────────────────────────────────────────────────────

_REQUIRED_IMU_COLS = {'timestamp', 'accel_x', 'accel_y', 'accel_z',
                      'gyro_x', 'gyro_y', 'gyro_z'}
_REQUIRED_GPS_COLS = {'timestamp', 'x', 'y', 'z'}


def _validate_columns(df, required, filename):
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"'{filename}' dosyasında eksik sütunlar: {sorted(missing)}. "
            f"Beklenen sütunlar: {sorted(required)}"
        )


def load_imu_excel(source):
    """
    Excel'den IMU verisini oku.

    Parameters
    ----------
    source : str veya bytes-like
        Dosya yolu ya da dosya içeriği (bytes).

    Returns
    -------
    imu_time  : list[float]
    imu_accel : list[np.matrix]  (3×1)
    imu_gyro  : list[np.matrix]  (3×1)
    """
    df = pd.read_excel(source)
    df.columns = [c.strip().lower() for c in df.columns]
    _validate_columns(df, _REQUIRED_IMU_COLS, 'imu-data.xlsx')

    df = df.sort_values('timestamp').reset_index(drop=True)

    imu_time  = df['timestamp'].tolist()
    imu_accel = [
        np.matrix([[r['accel_x']], [r['accel_y']], [r['accel_z']]])
        for _, r in df.iterrows()
    ]
    imu_gyro  = [
        np.matrix([[r['gyro_x']], [r['gyro_y']], [r['gyro_z']]])
        for _, r in df.iterrows()
    ]
    return imu_time, imu_accel, imu_gyro


def load_gps_excel(source):
    """
    Excel'den GPS verisini oku.

    Parameters
    ----------
    source : str veya bytes-like
        Dosya yolu ya da dosya içeriği (bytes).

    Returns
    -------
    gnss_time : list[float]
    gnss_dist : list[np.matrix]  (3×1)
    """
    df = pd.read_excel(source)
    df.columns = [c.strip().lower() for c in df.columns]
    _validate_columns(df, _REQUIRED_GPS_COLS, 'gps-data.xlsx')

    df = df.sort_values('timestamp').reset_index(drop=True)

    gnss_time = df['timestamp'].tolist()
    gnss_dist = [
        np.matrix([[r['x']], [r['y']], [r['z']]])
        for _, r in df.iterrows()
    ]
    return gnss_time, gnss_dist


# ─────────────────────────────────────────────────────────────
# 5.  Grafik çizimi
# ─────────────────────────────────────────────────────────────

def plot_results(imu_time, state_list, var_list, gnss_time, gnss_dist):
    """
    EKF sonuçlarını 4 panelde çiz:
      - Konum (X, Y, Z) + GPS ölçümleri + ±1σ bandı
      - Hız  (Vx, Vy, Vz)
      - Açılar (psi, theta, gamma) — derece cinsinden
      - Bias kestirimleri (ivme & jiroskop)
    """
    t = imu_time

    pos_x = [s.item((0,  0)) for s in state_list]
    pos_y = [s.item((1,  0)) for s in state_list]
    pos_z = [s.item((2,  0)) for s in state_list]
    spd_x = [s.item((3,  0)) for s in state_list]
    spd_y = [s.item((4,  0)) for s in state_list]
    spd_z = [s.item((5,  0)) for s in state_list]
    ab_x  = [s.item((6,  0)) for s in state_list]
    ab_y  = [s.item((7,  0)) for s in state_list]
    ab_z  = [s.item((8,  0)) for s in state_list]
    wb_x  = [s.item((9,  0)) for s in state_list]
    wb_y  = [s.item((10, 0)) for s in state_list]
    wb_z  = [s.item((11, 0)) for s in state_list]
    psi   = [math.degrees(s.item((12, 0))) for s in state_list]
    theta = [math.degrees(s.item((13, 0))) for s in state_list]
    gamma = [math.degrees(s.item((14, 0))) for s in state_list]

    std_px = [math.sqrt(max(P.item((0, 0)), 0)) for P in var_list]
    std_py = [math.sqrt(max(P.item((1, 1)), 0)) for P in var_list]
    std_pz = [math.sqrt(max(P.item((2, 2)), 0)) for P in var_list]

    gnss_x = [g.item((0, 0)) for g in gnss_dist]
    gnss_y = [g.item((1, 0)) for g in gnss_dist]
    gnss_z = [g.item((2, 0)) for g in gnss_dist]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('INS + EKF Sonuçları', fontsize=14, fontweight='bold')

    # ── Panel 1: Konum ──────────────────────────────────────
    ax = axes[0, 0]
    for arr, std, lbl, col in [
        (pos_x, std_px, 'X (m)', 'C0'),
        (pos_y, std_py, 'Y (m)', 'C1'),
        (pos_z, std_pz, 'Z (m)', 'C2'),
    ]:
        ax.plot(t, arr, label=f'EKF {lbl}', color=col)
        ax.fill_between(
            t,
            np.array(arr) - np.array(std),
            np.array(arr) + np.array(std),
            color=col, alpha=0.2
        )
    ax.scatter(gnss_time, gnss_x, marker='x', color='C0', zorder=5, label='GPS X')
    ax.scatter(gnss_time, gnss_y, marker='x', color='C1', zorder=5, label='GPS Y')
    ax.scatter(gnss_time, gnss_z, marker='x', color='C2', zorder=5, label='GPS Z')
    ax.set_xlabel('Zaman (s)')
    ax.set_ylabel('Konum (m)')
    ax.set_title('Konum Tahmini vs GPS')
    ax.legend(fontsize=7)
    ax.grid(True)

    # ── Panel 2: Hız ────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(t, spd_x, label='Vx (m/s)', color='C0')
    ax.plot(t, spd_y, label='Vy (m/s)', color='C1')
    ax.plot(t, spd_z, label='Vz (m/s)', color='C2')
    ax.set_xlabel('Zaman (s)')
    ax.set_ylabel('Hız (m/s)')
    ax.set_title('Hız Tahmini')
    ax.legend()
    ax.grid(True)

    # ── Panel 3: Açılar ─────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(t, psi,   label='Psi (°)',   color='C3')
    ax.plot(t, theta, label='Theta (°)', color='C4')
    ax.plot(t, gamma, label='Gamma (°)', color='C5')
    ax.set_xlabel('Zaman (s)')
    ax.set_ylabel('Açı (derece)')
    ax.set_title('Attitude (Euler Açıları)')
    ax.legend()
    ax.grid(True)

    # ── Panel 4: Bias kestirimleri ───────────────────────────
    ax = axes[1, 1]
    ax.plot(t, ab_x, label='Accel bias X', linestyle='-',  color='C0')
    ax.plot(t, ab_y, label='Accel bias Y', linestyle='--', color='C1')
    ax.plot(t, ab_z, label='Accel bias Z', linestyle=':',  color='C2')
    ax.plot(t, wb_x, label='Gyro bias X',  linestyle='-',  color='C3')
    ax.plot(t, wb_y, label='Gyro bias Y',  linestyle='--', color='C4')
    ax.plot(t, wb_z, label='Gyro bias Z',  linestyle=':',  color='C5')
    ax.set_xlabel('Zaman (s)')
    ax.set_ylabel('Bias değeri')
    ax.set_title('Bias Kestirimleri')
    ax.legend(fontsize=7)
    ax.grid(True)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# 6.  Sonuç dışa aktarma
# ─────────────────────────────────────────────────────────────

def export_csv(imu_time, state_list, csv_path='ekf_results.csv'):
    """EKF sonuçlarını CSV olarak kaydet."""
    rows = []
    names = ['time',
             'pos_x', 'pos_y', 'pos_z',
             'spd_x', 'spd_y', 'spd_z',
             'accel_bias_x', 'accel_bias_y', 'accel_bias_z',
             'gyro_bias_x',  'gyro_bias_y',  'gyro_bias_z',
             'psi_deg', 'theta_deg', 'gamma_deg']
    for t, s in zip(imu_time, state_list):
        rows.append([
            t,
            s.item((0,  0)), s.item((1,  0)), s.item((2,  0)),
            s.item((3,  0)), s.item((4,  0)), s.item((5,  0)),
            s.item((6,  0)), s.item((7,  0)), s.item((8,  0)),
            s.item((9,  0)), s.item((10, 0)), s.item((11, 0)),
            math.degrees(s.item((12, 0))),
            math.degrees(s.item((13, 0))),
            math.degrees(s.item((14, 0))),
        ])
    df = pd.DataFrame(rows, columns=names)
    df.to_csv(csv_path, index=False)
    print(f'CSV kaydedildi → {csv_path}')
    return csv_path


def export_png(fig, png_path='ekf_plots.png'):
    """Grafikleri PNG olarak kaydet."""
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f'PNG kaydedildi → {png_path}')
    return png_path


# ─────────────────────────────────────────────────────────────
# 7.  Ana işlem akışı
# ─────────────────────────────────────────────────────────────

def run(imu_source, gps_source,
        accel_bias_std=0.3,
        accel_w_std=0.05,
        gyro_bias_std=None,
        gyro_w_std=None,
        attitude0_std=None,
        gnss_dist_std=0.5,
        gnss_speed_std=0.2,
        attitude0=None,
        gyro_bias0=None):
    """
    Tüm işlemi çalıştır: yükle → filtrele → çiz → dışa aktar.

    Parameters
    ----------
    imu_source : str | bytes
        IMU Excel dosyasının yolu ya da içeriği.
    gps_source : str | bytes
        GPS Excel dosyasının yolu ya da içeriği.

    Returns
    -------
    (fig, csv_path, png_path)
    """
    defaults = {
        'gyro_w_std':    math.radians(0.2),
        'gyro_bias_std': math.radians(1.0),
        'attitude0_std': math.radians(1.0),
        'attitude0':     np.matrix([[0.0], [0.0], [0.0]]),
        'gyro_bias0':    np.matrix([[0.0], [0.0], [0.0]]),
    }
    if gyro_w_std    is None: gyro_w_std    = defaults['gyro_w_std']
    if gyro_bias_std is None: gyro_bias_std = defaults['gyro_bias_std']
    if attitude0_std is None: attitude0_std = defaults['attitude0_std']
    if attitude0     is None: attitude0     = defaults['attitude0']
    if gyro_bias0    is None: gyro_bias0    = defaults['gyro_bias0']

    print('IMU verisi yükleniyor…')
    imu_time, imu_accel, imu_gyro = load_imu_excel(imu_source)
    print(f'  {len(imu_time)} IMU örneği yüklendi.')

    print('GPS verisi yükleniyor…')
    gnss_time, gnss_dist = load_gps_excel(gps_source)
    print(f'  {len(gnss_time)} GPS örneği yüklendi.')

    if len(imu_time) < 2:
        raise ValueError('IMU verisinde en az 2 satır gereklidir.')
    if len(gnss_time) < 1:
        raise ValueError('GPS verisinde en az 1 satır gereklidir.')

    print('EKF çalıştırılıyor…')
    state_list, var_list = ins_ext_kfilter(
        imu_time, imu_accel, imu_gyro,
        accel_bias_std, accel_w_std, gyro_bias_std, gyro_w_std,
        attitude0, attitude0_std, gyro_bias0,
        gnss_time, [], gnss_dist, gnss_speed_std, gnss_dist_std
    )
    print('  EKF tamamlandı.')

    print('Grafikler oluşturuluyor…')
    fig = plot_results(imu_time, state_list, var_list, gnss_time, gnss_dist)

    csv_path = export_csv(imu_time, state_list)
    png_path = export_png(fig)

    return fig, csv_path, png_path


# ─────────────────────────────────────────────────────────────
# 8.  Colab giriş noktası
# ─────────────────────────────────────────────────────────────

def main_colab():
    """
    Google Colab'da çalıştırılacak ana fonksiyon.
    Dosya yükleme widget'ını açar, EKF'yi çalıştırır
    ve sonuçları indirir.
    """
    from google.colab import files

    print('=' * 60)
    print('INS + EKF — Google Colab Arayüzü')
    print('=' * 60)
    print()

    # ── IMU dosyası yükleme ──────────────────────────────────
    print('► Lütfen imu-data.xlsx dosyasını seçin:')
    imu_uploaded = files.upload()
    if not imu_uploaded:
        raise RuntimeError('IMU dosyası yüklenmedi.')
    imu_filename = list(imu_uploaded.keys())[0]
    imu_bytes    = io.BytesIO(imu_uploaded[imu_filename])
    print(f'  ✓ "{imu_filename}" yüklendi.')
    print()

    # ── GPS dosyası yükleme ──────────────────────────────────
    print('► Lütfen gps-data.xlsx dosyasını seçin:')
    gps_uploaded = files.upload()
    if not gps_uploaded:
        raise RuntimeError('GPS dosyası yüklenmedi.')
    gps_filename = list(gps_uploaded.keys())[0]
    gps_bytes    = io.BytesIO(gps_uploaded[gps_filename])
    print(f'  ✓ "{gps_filename}" yüklendi.')
    print()

    # ── EKF çalıştır ────────────────────────────────────────
    fig, csv_path, png_path = run(imu_bytes, gps_bytes)

    # ── Colab'da göster ─────────────────────────────────────
    matplotlib.use('inline')   # Colab inline backend'e geç
    plt.show()

    # ── İndir ───────────────────────────────────────────────
    print()
    print('► Sonuç dosyaları indiriliyor…')
    files.download(csv_path)
    files.download(png_path)
    print('✓ Tamamlandı!')


# ─────────────────────────────────────────────────────────────
# 9.  Doğrudan çalıştırma (yerel test)
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if IS_COLAB:
        main_colab()
    else:
        # Yerel test: dosya yollarını buraya girin
        IMU_FILE = 'imu-data.xlsx'
        GPS_FILE = 'gps-data.xlsx'

        if not os.path.isfile(IMU_FILE) or not os.path.isfile(GPS_FILE):
            print(
                f'Yerel test için "{IMU_FILE}" ve "{GPS_FILE}" '
                'dosyalarını bu betiğin yanına koyun.'
            )
        else:
            fig, csv_path, png_path = run(IMU_FILE, GPS_FILE)
            plt.show()
            print('Yerel çalıştırma tamamlandı.')
