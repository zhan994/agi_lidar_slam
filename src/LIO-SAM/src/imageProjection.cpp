#include "lio_sam/cloud_info.h"
#include "utility.h"

struct VelodynePointXYZIRT {
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY;
  uint16_t ring;
  float time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(
    VelodynePointXYZIRT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity,
                                            intensity)(uint16_t, ring,
                                                       ring)(float, time, time))

struct OusterPointXYZIRT {
  PCL_ADD_POINT4D;
  float intensity;
  uint32_t t;
  uint16_t reflectivity;
  uint8_t ring;
  uint16_t noise;
  uint32_t range;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(
    OusterPointXYZIRT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
        uint32_t, t, t)(uint16_t, reflectivity, reflectivity)(
        uint8_t, ring, ring)(uint16_t, noise, noise)(uint32_t, range, range))

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;

const int queueLength = 2000;

class ImageProjection : public ParamServer {
 private:
  std::mutex imuLock;
  std::mutex odoLock;

  ros::Subscriber subLaserCloud;
  ros::Publisher pubLaserCloud;

  ros::Publisher pubExtractedCloud;
  ros::Publisher pubLaserCloudInfo;

  ros::Subscriber subImu;
  std::deque<sensor_msgs::Imu> imuQueue;

  ros::Subscriber subOdom;
  std::deque<nav_msgs::Odometry> odomQueue;

  std::deque<sensor_msgs::PointCloud2> cloudQueue;
  sensor_msgs::PointCloud2 currentCloudMsg;

  double* imuTime = new double[queueLength];
  double* imuRotX = new double[queueLength];
  double* imuRotY = new double[queueLength];
  double* imuRotZ = new double[queueLength];

  int imuPointerCur;
  bool firstPointFlag;
  Eigen::Affine3f transStartInverse;

  pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
  pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
  pcl::PointCloud<PointType>::Ptr fullCloud;
  pcl::PointCloud<PointType>::Ptr extractedCloud;

  int deskewFlag;
  cv::Mat rangeMat;

  bool odomDeskewFlag;
  float odomIncreX;
  float odomIncreY;
  float odomIncreZ;

  lio_sam::cloud_info cloudInfo;
  double timeScanCur;
  double timeScanEnd;
  std_msgs::Header cloudHeader;

 public:
  /**
   * \brief // api: 构造函数
   *
   */
  ImageProjection() : deskewFlag(0) {
    // step: 1 订阅imu数据，后端里程记数据，原始点云数据
    subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000,
                                            &ImageProjection::imuHandler, this,
                                            ros::TransportHints().tcpNoDelay());
    subOdom = nh.subscribe<nav_msgs::Odometry>(
        odomTopic + "_incremental", 2000, &ImageProjection::odometryHandler,
        this, ros::TransportHints().tcpNoDelay());
    subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(
        pointCloudTopic, 5, &ImageProjection::cloudHandler, this,
        ros::TransportHints().tcpNoDelay());

    // step: 2 发布去畸变数据
    pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2>(
        "lio_sam/deskew/cloud_deskewed", 1);
    pubLaserCloudInfo =
        nh.advertise<lio_sam::cloud_info>("lio_sam/deskew/cloud_info", 1);

    // step: 3 分配空间
    allocateMemory();

    // step: 4 初始化参数
    resetParameters();

    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
  }

  /**
   * \brief // api: 分配相关内存
   *
   */
  void allocateMemory() {
    laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
    tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
    fullCloud.reset(new pcl::PointCloud<PointType>());
    extractedCloud.reset(new pcl::PointCloud<PointType>());

    fullCloud->points.resize(N_SCAN * Horizon_SCAN);

    cloudInfo.startRingIndex.assign(N_SCAN, 0);
    cloudInfo.endRingIndex.assign(N_SCAN, 0);

    cloudInfo.pointColInd.assign(N_SCAN * Horizon_SCAN, 0);
    cloudInfo.pointRange.assign(N_SCAN * Horizon_SCAN, 0);

    resetParameters();
  }

  /**
   * \brief // api: 重置参数
   *
   */
  void resetParameters() {
    laserCloudIn->clear();
    extractedCloud->clear();
    // reset range matrix for range image projection
    rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

    imuPointerCur = 0;
    firstPointFlag = true;
    odomDeskewFlag = false;

    for (int i = 0; i < queueLength; ++i) {
      imuTime[i] = 0;
      imuRotX[i] = 0;
      imuRotY[i] = 0;
      imuRotZ[i] = 0;
    }
  }

  ~ImageProjection() {}

  /**
   * \brief // api: IMU回调
   *
   * \param imuMsg 消息
   */
  void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg) {
    sensor_msgs::Imu thisImu = imuConverter(*imuMsg);  // 对imu做一个坐标转换
    // 加一个线程锁，把imu数据保存进队列
    std::lock_guard<std::mutex> lock1(imuLock);
    imuQueue.push_back(thisImu);

    // debug IMU data
    // cout << std::setprecision(6);
    // cout << "IMU acc: " << endl;
    // cout << "x: " << thisImu.linear_acceleration.x <<
    //       ", y: " << thisImu.linear_acceleration.y <<
    //       ", z: " << thisImu.linear_acceleration.z << endl;
    // cout << "IMU gyro: " << endl;
    // cout << "x: " << thisImu.angular_velocity.x <<
    //       ", y: " << thisImu.angular_velocity.y <<
    //       ", z: " << thisImu.angular_velocity.z << endl;
    // double imuRoll, imuPitch, imuYaw;
    // tf::Quaternion orientation;
    // tf::quaternionMsgToTF(thisImu.orientation, orientation);
    // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
    // cout << "IMU roll pitch yaw: " << endl;
    // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " <<
    // imuYaw << endl << endl;
  }

  /**
   * \brief // api: 预积分计算的增量里程计
   *
   * \param odometryMsg 消息
   */
  void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg) {
    std::lock_guard<std::mutex> lock2(odoLock);
    odomQueue.push_back(*odometryMsg);
  }

  /**
   * \brief // api: 点云回调
   *
   * \param laserCloudMsg 消息
   */
  void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg) {
    // step: 1 点云消息预处理
    if (!cachePointCloud(laserCloudMsg)) return;

    // step: 2 去畸变补偿信息
    if (!deskewInfo()) return;

    // step: 3 点云去畸变并映射到矩阵中
    projectPointCloud();

    // step: 4 提取出有效的点的信息
    cloudExtraction();

    // step: 5 发布点云
    publishClouds();

    // step: 6 重置点云处理相关参数
    resetParameters();
  }

  /**
   * \brief // api: 点云消息预处理
   *
   * \param laserCloudMsg 消息
   * \return true
   * \return false
   */
  bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg) {
    // step: 1 点云数据保存进队列，确保队列里大于两帧点云数据
    cloudQueue.push_back(*laserCloudMsg);
    if (cloudQueue.size() <= 2) return false;

    // step: 2 缓存了足够多的点云之后，转换数据
    currentCloudMsg = std::move(cloudQueue.front());
    cloudQueue.pop_front();
    if (sensor == SensorType::VELODYNE) {
      pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);  // 转成pcl的点云格式
    } else if (sensor == SensorType::OUSTER) {
      // Convert to Velodyne format
      pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
      laserCloudIn->points.resize(tmpOusterCloudIn->size());
      laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
      for (size_t i = 0; i < tmpOusterCloudIn->size(); i++) {
        auto& src = tmpOusterCloudIn->points[i];
        auto& dst = laserCloudIn->points[i];
        dst.x = src.x;
        dst.y = src.y;
        dst.z = src.z;
        dst.intensity = src.intensity;
        dst.ring = src.ring;
        dst.time = src.t * 1e-9f;
      }
    } else {
      ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
      ros::shutdown();
    }

    // step: 3 get timestamp
    cloudHeader = currentCloudMsg.header;
    timeScanCur = cloudHeader.stamp.toSec();
    timeScanEnd = timeScanCur + laserCloudIn->points.back().time;

    // step: 4 is_dense是点云是否有序排列的标志
    if (laserCloudIn->is_dense == false) {
      ROS_ERROR(
          "Point cloud is not in dense format, please remove NaN points "
          "first!");
      ros::shutdown();
    }

    // step: 5 查看驱动里是否把每个点属于哪一根扫描scan这个信息
    static int ringFlag = 0;
    if (ringFlag == 0) {
      ringFlag = -1;
      for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i) {
        if (currentCloudMsg.fields[i].name == "ring") {
          ringFlag = 1;
          break;
        }
      }
      // 如果没有这个信息就需要像loam或者lego loam那样手动计算scan
      // id，现在velodyne的驱动里都会携带这些信息的
      if (ringFlag == -1) {
        ROS_ERROR(
            "Point cloud ring channel not available, please configure your "
            "point cloud data!");
        ros::shutdown();
      }
    }

    // step: 6 检查是否有时间戳信息
    if (deskewFlag == 0) {
      deskewFlag = -1;
      for (auto& field : currentCloudMsg.fields) {
        if (field.name == "time" || field.name == "t") {
          deskewFlag = 1;
          break;
        }
      }
      if (deskewFlag == -1)
        ROS_WARN(
            "Point cloud timestamp not available, deskew function disabled, "
            "system will drift significantly!");
    }

    return true;
  }

  /**
   * \brief // api: 去畸变信息收集
   *
   * \return true
   * \return false
   */
  bool deskewInfo() {
    std::lock_guard<std::mutex> lock1(imuLock);
    std::lock_guard<std::mutex> lock2(odoLock);

    // step: 1 确保imu的数据覆盖这一帧的点云
    if (imuQueue.empty() ||
        imuQueue.front().header.stamp.toSec() > timeScanCur ||
        imuQueue.back().header.stamp.toSec() < timeScanEnd) {
      ROS_DEBUG("Waiting for IMU data ...");
      return false;
    }

    // step: 2 imu补偿旋转
    imuDeskewInfo();

    // step: 3 里程计补偿位移
    odomDeskewInfo();

    return true;
  }

  /**
   * \brief // api: IMU畸变补偿信息
   *
   */
  void imuDeskewInfo() {
    cloudInfo.imuAvailable = false;

    // step: 1 扔掉把过早的imu
    while (!imuQueue.empty()) {
      if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
        imuQueue.pop_front();
      else
        break;
    }

    // step: 2 给当前帧获取姿态角
    if (imuQueue.empty()) return;
    imuPointerCur = 0;
    for (int i = 0; i < (int)imuQueue.size(); ++i) {
      sensor_msgs::Imu thisImuMsg = imuQueue[i];
      // step: 2.1 计算早于当前帧的IMU姿态
      double currentImuTime = thisImuMsg.header.stamp.toSec();
      if (currentImuTime <= timeScanCur)
        // 把imu的姿态转成欧拉角
        imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit,
                      &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);
      // 这一帧遍历完了就break
      if (currentImuTime > timeScanEnd + 0.01) break;

      // step: 2.2 起始帧
      if (imuPointerCur == 0) {
        imuRotX[0] = 0;
        imuRotY[0] = 0;
        imuRotZ[0] = 0;
        imuTime[0] = currentImuTime;
        ++imuPointerCur;
        continue;
      }

      // step: 2.3 计算每一个时刻的姿态角方便后续查找对应每个点云时间的值
      double angular_x, angular_y, angular_z;
      imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);
      double timeDiff = currentImuTime - imuTime[imuPointerCur - 1];
      imuRotX[imuPointerCur] =
          imuRotX[imuPointerCur - 1] + angular_x * timeDiff;
      imuRotY[imuPointerCur] =
          imuRotY[imuPointerCur - 1] + angular_y * timeDiff;
      imuRotZ[imuPointerCur] =
          imuRotZ[imuPointerCur - 1] + angular_z * timeDiff;
      imuTime[imuPointerCur] = currentImuTime;
      ++imuPointerCur;
    }

    --imuPointerCur;
    if (imuPointerCur <= 0) return;

    // 可以使用imu数据进行运动补偿
    cloudInfo.imuAvailable = true;
  }

  /**
   * \brief // api: 里程计畸变补偿信息
   *
   */
  void odomDeskewInfo() {
    cloudInfo.odomAvailable = false;

    // step: 1 扔掉过早的数据，覆盖整个点云的时间
    while (!odomQueue.empty()) {
      if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
        odomQueue.pop_front();
      else
        break;
    }
    if (odomQueue.empty()) return;
    // 点云时间   ×××××××
    // odom时间     ×××××
    if (odomQueue.front().header.stamp.toSec() > timeScanCur) return;

    // step: 2 找到对应的最早的点云时间的odom数据
    nav_msgs::Odometry startOdomMsg;
    for (int i = 0; i < (int)odomQueue.size(); ++i) {
      startOdomMsg = odomQueue[i];
      if (ROS_TIME(&startOdomMsg) < timeScanCur)
        continue;
      else
        break;
    }
    // 将ros消息格式中的姿态转成tf的格式，然后将四元数转成欧拉角
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);
    double roll, pitch, yaw;
    tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

    // step: 3 记录点云起始时刻的对应的odom姿态，用于mapOptimization的Initial
    cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
    cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
    cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
    cloudInfo.initialGuessRoll = roll;
    cloudInfo.initialGuessPitch = pitch;
    cloudInfo.initialGuessYaw = yaw;
    cloudInfo.odomAvailable = true;  // odom提供了这一帧点云的初始位姿

    // step: 4 找到对应的最晚的点云时间的odom数据
    odomDeskewFlag = false;
    // 这里发现没有覆盖到最后的点云，那就不能用odom数据来做运动补偿
    if (odomQueue.back().header.stamp.toSec() < timeScanEnd) return;
    nav_msgs::Odometry endOdomMsg;
    for (int i = 0; i < (int)odomQueue.size(); ++i) {
      endOdomMsg = odomQueue[i];

      if (ROS_TIME(&endOdomMsg) < timeScanEnd)
        continue;
      else
        break;
    }

    // note: 这个代表odom退化了，就置信度不高了
    if (int(round(startOdomMsg.pose.covariance[0])) !=
        int(round(endOdomMsg.pose.covariance[0])))
      return;

    // step: 5 起始位姿和结束位姿都转成Affine3f这个数据结构
    Eigen::Affine3f transBegin = pcl::getTransformation(
        startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y,
        startOdomMsg.pose.pose.position.z, roll, pitch, yaw);
    tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
    Eigen::Affine3f transEnd = pcl::getTransformation(
        endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y,
        endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

    // step: 6 计算起始位姿和结束位姿之间的delta pose，增量转成xyz和欧拉角的形式
    Eigen::Affine3f transBt = transBegin.inverse() * transEnd;
    float rollIncre, pitchIncre, yawIncre;
    pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY,
                                      odomIncreZ, rollIncre, pitchIncre,
                                      yawIncre);

    odomDeskewFlag = true;  // 表示可以用odom来做运动补偿
  }

  /**
   * \brief // api: 计算相对旋转
   *
   * \param pointTime 点云时间戳
   * \param rotXCur roll
   * \param rotYCur pitch
   * \param rotZCur yaw
   */
  void findRotation(double pointTime, float* rotXCur, float* rotYCur,
                    float* rotZCur) {
    *rotXCur = 0;
    *rotYCur = 0;
    *rotZCur = 0;

    int imuPointerFront = 0;
    // step: 1 找到第一个超过当前帧的imu数据
    // imuPointerBack     imuPointerFront
    //    ×                      ×
    //               ×
    //           imuPointerCur
    // note: imuPointerCur是imu计算旋转buffer，确保不越界
    while (imuPointerFront < imuPointerCur) {
      if (pointTime < imuTime[imuPointerFront]) break;
      ++imuPointerFront;
    }

    // step: 2 如果时间戳不在两个imu的旋转之间，就直接赋值了
    if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0) {
      *rotXCur = imuRotX[imuPointerFront];
      *rotYCur = imuRotY[imuPointerFront];
      *rotZCur = imuRotZ[imuPointerFront];
    } else {
      // step: 3 否则 做一个线性插值，得到相对旋转
      int imuPointerBack = imuPointerFront - 1;
      double ratioFront = (pointTime - imuTime[imuPointerBack]) /
                          (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
      double ratioBack = (imuTime[imuPointerFront] - pointTime) /
                         (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
      *rotXCur = imuRotX[imuPointerFront] * ratioFront +
                 imuRotX[imuPointerBack] * ratioBack;
      *rotYCur = imuRotY[imuPointerFront] * ratioFront +
                 imuRotY[imuPointerBack] * ratioBack;
      *rotZCur = imuRotZ[imuPointerFront] * ratioFront +
                 imuRotZ[imuPointerBack] * ratioBack;
    }
  }

  /**
   * \brief // api: 计算相对位移
   *
   * \param relTime 点云时间戳
   * \param posXCur x
   * \param posYCur y
   * \param posZCur z
   */
  void findPosition(double relTime, float* posXCur, float* posYCur,
                    float* posZCur) {
    *posXCur = 0;
    *posYCur = 0;
    *posZCur = 0;

    // If the sensor moves relatively slow, like walking speed, positional
    // deskew seems to have little benefits. Thus code below is commented.

    // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
    //     return;

    // float ratio = relTime / (timeScanEnd - timeScanCur);

    // *posXCur = ratio * odomIncreX;
    // *posYCur = ratio * odomIncreY;
    // *posZCur = ratio * odomIncreZ;
  }

  /**
   * \brief // api: 去畸变
   *
   * \param point 点云输入
   * \param relTime 相对时间
   * \return PointType 去畸变后输出
   */
  PointType deskewPoint(PointType* point, double relTime) {
    if (deskewFlag == -1 || cloudInfo.imuAvailable == false) return *point;

    // step: 1 relTime是相对时间，加上起始时间就是绝对时间
    double pointTime = timeScanCur + relTime;

    // step: 2 计算当前点相对起始点的相对旋转
    float rotXCur, rotYCur, rotZCur;
    findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

    // step: 3 这里没有计算平移补偿
    float posXCur, posYCur, posZCur;
    findPosition(relTime, &posXCur, &posYCur, &posZCur);

    // step: 4 第一个点做单位变换
    if (firstPointFlag == true) {
      // 计算第一个点的相对位姿
      transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur,
                                                  rotXCur, rotYCur, rotZCur))
                              .inverse();
      firstPointFlag = false;
    }

    // step: 5 计算当前点和第一个点的相对位姿
    Eigen::Affine3f transFinal = pcl::getTransformation(
        posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
    Eigen::Affine3f transBt = transStartInverse * transFinal;

    // step: 6 R × p + t，把点补偿到第一个点对应时刻的位姿
    PointType newPoint;
    newPoint.x = transBt(0, 0) * point->x + transBt(0, 1) * point->y +
                 transBt(0, 2) * point->z + transBt(0, 3);
    newPoint.y = transBt(1, 0) * point->x + transBt(1, 1) * point->y +
                 transBt(1, 2) * point->z + transBt(1, 3);
    newPoint.z = transBt(2, 0) * point->x + transBt(2, 1) * point->y +
                 transBt(2, 2) * point->z + transBt(2, 3);
    newPoint.intensity = point->intensity;

    return newPoint;
  }

  /**
   * \brief // api: 将点云投影到一个矩阵上，并且保存每个点的信息
   *
   */
  void projectPointCloud() {
    int cloudSize = laserCloudIn->points.size();
    // range image projection
    for (int i = 0; i < cloudSize; ++i) {
      PointType thisPoint;
      // step: 1 取出对应的某个点
      thisPoint.x = laserCloudIn->points[i].x;
      thisPoint.y = laserCloudIn->points[i].y;
      thisPoint.z = laserCloudIn->points[i].z;
      thisPoint.intensity = laserCloudIn->points[i].intensity;

      // step: 2 计算这个点距离lidar中心的距离，距离太小或者太远都认为是异常点
      float range = pointDistance(thisPoint);
      if (range < lidarMinRange || range > lidarMaxRange) continue;

      // step: 3 取出对应的在第几根scan上，scanid合理，降采样根据scan id适当跳过
      int rowIdn = laserCloudIn->points[i].ring;
      if (rowIdn < 0 || rowIdn >= N_SCAN) continue;
      if (rowIdn % downsampleRate != 0) continue;

      // step: 4 计算水平角和水平线束id，对水平id检查，已经有填充了就跳过
      float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
      static float ang_res_x = 360.0 / float(Horizon_SCAN);
      // note: x负方向e为起始，顺时针为正方向，范围[0,H]
      int columnIdn =
          -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN / 2;
      if (columnIdn >= Horizon_SCAN) columnIdn -= Horizon_SCAN;
      if (columnIdn < 0 || columnIdn >= Horizon_SCAN) continue;
      if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX) continue;

      // step: 5 对点做运动补偿
      thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);

      // step: 6 将这个点的距离数据保存进这个range矩阵中
      rangeMat.at<float>(rowIdn, columnIdn) = range;

      // step: 7 算出这个点的索引，保存这个点的坐标
      int index = columnIdn + rowIdn * Horizon_SCAN;
      fullCloud->points[index] = thisPoint;
    }
  }

  /**
   * \brief // api: 提取出有效的点的信息
   *
   */
  void cloudExtraction() {
    int count = 0;
    for (int i = 0; i < N_SCAN; ++i) {
      // step: 1 计算曲率需要左右各五个点
      cloudInfo.startRingIndex[i] = count - 1 + 5;

      // step: 2 保存有效信息
      for (int j = 0; j < Horizon_SCAN; ++j) {
        if (rangeMat.at<float>(i, j) != FLT_MAX) {
          // step: 2.1 点对应着哪一根垂直线，marking occlusion later
          cloudInfo.pointColInd[count] = j;
          // step: 2.2 距离信息
          cloudInfo.pointRange[count] = rangeMat.at<float>(i, j);
          // step: 2.3 3d坐标信息
          extractedCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]);
          // step: 2.4 count只在有效点才会累加
          ++count;
        }
      }

      cloudInfo.endRingIndex[i] = count - 1 - 5;
    }
  }

  /**
   * \brief // api: 发布点云
   *
   */
  void publishClouds() {
    cloudInfo.header = cloudHeader;
    // 发布提取出来的有效的点
    cloudInfo.cloud_deskewed = publishCloud(&pubExtractedCloud, extractedCloud,
                                            cloudHeader.stamp, lidarFrame);
    pubLaserCloudInfo.publish(cloudInfo);
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "lio_sam");

  ImageProjection IP;

  ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");

  ros::MultiThreadedSpinner spinner(3);
  spinner.spin();

  return 0;
}
