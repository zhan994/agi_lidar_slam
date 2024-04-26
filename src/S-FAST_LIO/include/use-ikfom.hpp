#ifndef USE_IKFOM_H1
#define USE_IKFOM_H1

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <boost/bind.hpp>
#include <cstdlib>
#include <vector>

#include "common_lib.h"
#include "sophus/so3.h"

// 该hpp主要包含：状态变量x，输入量u的定义，以及正向传播中相关矩阵的函数

// 24维的状态量x
struct state_ikfom {
  Eigen::Vector3d pos = Eigen::Vector3d(0, 0, 0);
  Sophus::SO3 rot = Sophus::SO3(Eigen::Matrix3d::Identity());
  Sophus::SO3 offset_R_L_I = Sophus::SO3(Eigen::Matrix3d::Identity());
  Eigen::Vector3d offset_T_L_I = Eigen::Vector3d(0, 0, 0);
  Eigen::Vector3d vel = Eigen::Vector3d(0, 0, 0);
  Eigen::Vector3d bg = Eigen::Vector3d(0, 0, 0);
  Eigen::Vector3d ba = Eigen::Vector3d(0, 0, 0);
  Eigen::Vector3d grav = Eigen::Vector3d(0, 0, -G_m_s2);
};

// 输入u
struct input_ikfom {
  Eigen::Vector3d acc = Eigen::Vector3d(0, 0, 0);
  Eigen::Vector3d gyro = Eigen::Vector3d(0, 0, 0);
};

/**
 * \brief // api: 噪声协方差Q的初始化，对应公式(8)的Q
 *
 * \return Eigen::Matrix<double, 12, 12>
 */
Eigen::Matrix<double, 12, 12> process_noise_cov() {
  Eigen::Matrix<double, 12, 12> Q = Eigen::MatrixXd::Zero(12, 12);
  Q.block<3, 3>(0, 0) = 0.0001 * Eigen::Matrix3d::Identity();
  Q.block<3, 3>(3, 3) = 0.0001 * Eigen::Matrix3d::Identity();
  Q.block<3, 3>(6, 6) = 0.00001 * Eigen::Matrix3d::Identity();
  Q.block<3, 3>(9, 9) = 0.00001 * Eigen::Matrix3d::Identity();

  return Q;
}

/**
 * \brief // api: 对应公式(2)中的f(x,u.w)
 *
 * \param s 状态量
 * \param in 输入
 * \return Eigen::Matrix<double, 24, 1> 输出f(x,u.w)
 */
Eigen::Matrix<double, 24, 1> get_f(state_ikfom s, input_ikfom in) {
  // note: 对应顺序与论文公式顺序不一致
  // 速度(3)，角速度(3),外参T(3),外参旋转R(3)，加速度(3),角速度偏置(3),加速度偏置(3),位置(3)
  // step: 1 定义数据
  Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
  // step: 2 输入的imu的角速度(也就是实际测量值) - 估计的bias值(对应公式的第1行)
  Eigen::Vector3d omega = in.gyro - s.bg;
  // step: 3 输入的imu的加速度，先转到世界坐标系（对应公式的第3行）
  Eigen::Vector3d a_inertial = s.rot.matrix() * (in.acc - s.ba);

  // step: 4 其余数据赋值
  for (int i = 0; i < 3; i++) {
    res(i) = s.vel[i];      // 速度（对应公式第2行）
    res(i + 3) = omega[i];  // 角速度（对应公式第1行）
    res(i + 12) = a_inertial[i] + s.grav[i];  //加速度（对应公式第3行）
  }

  return res;
}

/**
 * \brief // api: 对应公式(7)的Fx
 *
 * // note: 该矩阵没乘dt，没加单位阵
 * \param s 状态量
 * \param in 输入
 * \return Eigen::Matrix<double, 24, 24> Fx
 */
Eigen::Matrix<double, 24, 24> df_dx(state_ikfom s, input_ikfom in) {
  // step: 1 定义数据
  Eigen::Matrix<double, 24, 24> cov = Eigen::Matrix<double, 24, 24>::Zero();
  // step: 2 对应公式(7)第2行第3列   I
  cov.block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
  // step: 3 测量加速度 = a_m - bias, 对应公式(7)第3行第1列
  Eigen::Vector3d acc_ = in.acc - s.ba;
  cov.block<3, 3>(12, 3) = -s.rot.matrix() * Sophus::SO3::hat(acc_);
  // step: 4 对应公式(7)第3行第5列
  cov.block<3, 3>(12, 18) = -s.rot.matrix();
  // step: 5 对应公式(7)第3行第6列   I
  cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity();
  // step: 6 对应公式(7)第1行第4列 (简化为-I)
  cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity();

  return cov;
}

/**
 * \brief // api: 对应公式(7)的Fw
 *
 * // note: 该矩阵没乘dt
 * \param s 状态量
 * \param in 输入
 * \return Eigen::Matrix<double, 24, 12> Fw
 */
Eigen::Matrix<double, 24, 12> df_dw(state_ikfom s, input_ikfom in) {
  // step: 1 定义数据
  Eigen::Matrix<double, 24, 12> cov = Eigen::Matrix<double, 24, 12>::Zero();
  // step: 2 对应公式(7)第3行第2列  -R
  cov.block<3, 3>(12, 3) = -s.rot.matrix();
  // step: 3 对应公式(7)第1行第1列  -A(w dt)简化为-I
  cov.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
  // step: 4 对应公式(7)第4行第3列  I
  cov.block<3, 3>(15, 6) = Eigen::Matrix3d::Identity();
  // step: 5 对应公式(7)第5行第4列  I
  cov.block<3, 3>(18, 9) = Eigen::Matrix3d::Identity();
  return cov;
}

#endif