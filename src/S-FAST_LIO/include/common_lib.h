#ifndef COMMON_LIB_H1
#define COMMON_LIB_H1

#include <Eigen/Eigen>
#include <eigen_conversions/eigen_msg.h>
#include <nav_msgs/Odometry.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/Imu.h>
#include <sfast_lio/Pose6D.h>
#include <tf/transform_broadcaster.h>

using namespace std;
using namespace Eigen;

#define PI_M (3.14159265358)
#define G_m_s2 (9.81)  // Gravaty const in GuangDong/China
#define NUM_MATCH_POINTS (5)

#define VEC_FROM_ARRAY(v) v[0], v[1], v[2]
#define MAT_FROM_ARRAY(v) v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]
#define SKEW_SYM_MATRX(v) 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0
#define DEBUG_FILE_DIR(name) (string(string(ROOT_DIR) + "Log/" + name))

typedef sfast_lio::Pose6D Pose6D;
typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;
typedef vector<PointType, Eigen::aligned_allocator<PointType>> PointVector;
typedef Vector3d V3D;
typedef Matrix3d M3D;
typedef Vector3f V3F;
typedef Matrix3f M3F;

M3D Eye3d(M3D::Identity());
M3F Eye3f(M3F::Identity());
V3D Zero3d(0, 0, 0);
V3F Zero3f(0, 0, 0);

// Lidar data and imu dates for the current process
struct MeasureGroup {
  MeasureGroup() {
    lidar_beg_time = 0.0;
    this->lidar.reset(new PointCloudXYZI());
  };
  double lidar_beg_time;
  double lidar_end_time;
  PointCloudXYZI::Ptr lidar;
  deque<sensor_msgs::Imu::ConstPtr> imu;
};

/**
 * \brief // api: Set the pose6d object
 *
 * \tparam T
 * \param t 时间间隔
 * \param a 加速度
 * \param g 角速度
 * \param v 速度
 * \param p 位置
 * \param R 姿态
 * \return auto Pose6D
 */
template <typename T>
auto set_pose6d(const double t, const Matrix<T, 3, 1>& a,
                const Matrix<T, 3, 1>& g, const Matrix<T, 3, 1>& v,
                const Matrix<T, 3, 1>& p, const Matrix<T, 3, 3>& R) {
  Pose6D rot_kp;
  rot_kp.offset_time = t;
  for (int i = 0; i < 3; i++) {
    rot_kp.acc[i] = a(i);
    rot_kp.gyr[i] = g(i);
    rot_kp.vel[i] = v(i);
    rot_kp.pos[i] = p(i);
    for (int j = 0; j < 3; j++) rot_kp.rot[i * 3 + j] = R(i, j);
  }
  return move(rot_kp);
}

/**
 * \brief // api: 计算两点之间距离
 *
 * \param p1
 * \param p2
 * \return float
 */
float calc_dist(PointType p1, PointType p2) {
  float d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) +
            (p1.z - p2.z) * (p1.z - p2.z);
  return d;
}

/**
 * \brief // api: 平面估计
 *
 * \tparam T
 * \param pca_result 平面方程参数
 * \param point 平面上的点
 * \param threshold 点到平面距离阈值
 * \return true
 * \return false
 */
template <typename T>
bool esti_plane(Matrix<T, 4, 1>& pca_result, const PointVector& point,
                const T& threshold) {
  Matrix<T, NUM_MATCH_POINTS, 3> A;
  Matrix<T, NUM_MATCH_POINTS, 1> b;
  A.setZero();
  b.setOnes();
  b *= -1.0f;

  // step: 1 求A/Dx + B/Dy + C/Dz + 1 = 0 的参数
  for (int j = 0; j < NUM_MATCH_POINTS; j++) {
    A(j, 0) = point[j].x;
    A(j, 1) = point[j].y;
    A(j, 2) = point[j].z;
  }

  // note: pca_result是平面方程的4个参数  /n是为了归一化
  Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);
  T n = normvec.norm();
  pca_result(0) = normvec(0) / n;
  pca_result(1) = normvec(1) / n;
  pca_result(2) = normvec(2) / n;
  pca_result(3) = 1.0 / n;

  // step: 2 如果几个点中有距离该平面>threshold的点 认为是不好的平面返回false
  for (int j = 0; j < NUM_MATCH_POINTS; j++) {
    if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y +
             pca_result(2) * point[j].z + pca_result(3)) > threshold) {
      return false;
    }
  }
  return true;
}

#endif