#include <livox_ros_driver/CustomMsg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

using namespace std;

#define IS_VALID(a) ((abs(a) > 1e8) ? true : false)

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

enum LID_TYPE { AVIA = 1, VELO16, OUST64, RS32 };  // {1, 2, 3, 4}
enum TIME_UNIT { SEC = 0, MS = 1, US = 2, NS = 3 };
enum Feature {
  Nor,
  Poss_Plane,
  Real_Plane,
  Edge_Jump,
  Edge_Plane,
  Wire,
  ZeroPoint
};
enum Surround { Prev, Next };
enum E_jump { Nr_nor, Nr_zero, Nr_180, Nr_inf, Nr_blind };

/**
 * \brief 原始数据格式
 * 
 */
struct orgtype {
  double range;
  double dista;
  double angle[2];
  double intersect;
  E_jump edj[2];
  Feature ftype;
  orgtype() {
    range = 0;
    edj[Prev] = Nr_nor;
    edj[Next] = Nr_nor;
    ftype = Nor;
    intersect = 2;
  }
};

namespace velodyne_ros {
struct EIGEN_ALIGN16 Point {
  PCL_ADD_POINT4D;
  float intensity;
  float time;
  uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace velodyne_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (float, intensity, intensity)
  (float, time, time)
  (std::uint16_t, ring, ring)
)

namespace rslidar_ros {
struct EIGEN_ALIGN16 Point {
  PCL_ADD_POINT4D;
  uint8_t intensity;
  uint16_t ring = 0;
  double timestamp = 0;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace rslidar_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(rslidar_ros::Point,
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (uint8_t, intensity, intensity)
  (uint16_t, ring, ring)
  (double, timestamp, timestamp)
)

namespace ouster_ros {
struct EIGEN_ALIGN16 Point {
  PCL_ADD_POINT4D;
  float intensity;
  uint32_t t;
  uint16_t reflectivity;
  uint8_t ring;
  uint16_t ambient;
  uint32_t range;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace ouster_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (float, intensity, intensity)
  (std::uint32_t, t, t)
  (std::uint16_t, reflectivity, reflectivity)
  (std::uint8_t, ring, ring)
  (std::uint16_t, ambient, ambient)
  (std::uint32_t, range, range)
)

class Preprocess
{
  public:
//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /**
   * \brief // api: 构造函数
   * 
   */
  Preprocess();

  /**
   * \brief // api: 析构函数
   * 
   */
  ~Preprocess();
  
  /**
   * \brief // api: livox自定义数据处理
   * 
   * \param msg 消息
   * \param pcl_out 处理后点云
   */
  void process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);

  /**
   * \brief // api: PointCloud2消息数据处理
   * 
   * \param msg 消息
   * \param pcl_out 处理后点云
   */
  void process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);

  /**
   * \brief // api: 设置是否提取特征、雷达类型、盲区距离以及降采样的点数
   * 
   * \param feat_en 是否提取特征
   * \param lid_type 雷达类型
   * \param bld 盲区距离
   * \param pfilt_num 降采样的点数
   */
  void set(bool feat_en, int lid_type, double bld, int pfilt_num);

  // sensor_msgs::PointCloud2::ConstPtr pointcloud;
  PointCloudXYZI pl_full, pl_corn, pl_surf;
  PointCloudXYZI pl_buff[128]; // maximum 128 line lidar
  vector<orgtype> typess[128]; // maximum 128 line lidar
  float time_unit_scale;
  int lidar_type, point_filter_num, N_SCANS, SCAN_RATE, time_unit;
  double blind;
  bool feature_enabled, given_offset_time;
  ros::Publisher pub_full, pub_surf, pub_corn;
    

  private:
  /**
   * \brief // api: 速腾回调
   * 
   * \param msg 
   */
  void rs_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);

  /**
   * \brief // api: avia回调
   * 
   * \param msg 
   */
  void avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg);

  /**
   * \brief // api: ouster64回调
   * 
   * \param msg 
   */
  void oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);

  /**
   * \brief // api: velodyne回调
   * 
   * \param msg 
   */
  void velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);

  /**
   * \brief // api: 特征提取
   * 
   * \param pl 点云
   * \param types 
   */
  void give_feature(PointCloudXYZI &pl, vector<orgtype> &types);

  /**
   * \brief //
   * 
   * \param pl 
   * \param ct 
   */
  void pub_func(PointCloudXYZI &pl, const ros::Time &ct);
  
  /**
   * \brief //
   * 
   * \param pl 
   * \param types 
   * \param i 
   * \param i_nex 
   * \param curr_direct 
   * \return int 
   */
  int  plane_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, uint &i_nex, Eigen::Vector3d &curr_direct);
  
  /**
   * \brief 
   * 
   * \param pl 
   * \param types 
   * \param i 
   * \param nor_dir 
   * \return true 
   * \return false 
   */
  bool edge_jump_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, Surround nor_dir);
  
  int group_size;
  double disA, disB, inf_bound;
  double limit_maxmid, limit_midmin, limit_maxmin;
  double p2l_ratio;
  double jump_up_limit, jump_down_limit;
  double cos160;
  double edgea, edgeb;
  double smallp_intersect, smallp_ratio;
  double vx, vy, vz;
};
