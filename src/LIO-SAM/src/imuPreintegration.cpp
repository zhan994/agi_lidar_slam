#include "utility.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

using gtsam::symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)

// 位姿融合，将预积分后半部分预测的位姿变化补偿到回环约束优化后的关键帧位姿
class TransformFusion : public ParamServer {
 public:
  std::mutex mtx;

  ros::Subscriber subImuOdometry;
  ros::Subscriber subLaserOdometry;

  ros::Publisher pubImuOdometry;
  ros::Publisher pubImuPath;

  Eigen::Affine3f lidarOdomAffine;
  Eigen::Affine3f imuOdomAffineFront;
  Eigen::Affine3f imuOdomAffineBack;

  tf::TransformListener tfListener;
  tf::StampedTransform lidar2Baselink;

  double lidarOdomTime = -1;
  deque<nav_msgs::Odometry> imuOdomQueue;

  /**
   * \brief // api: 构造函数
   *
   */
  TransformFusion() {
    // step: 1 静态tf
    // 如果lidar帧和baselink帧不是同一个坐标系
    // 通常baselink指车体系
    if (lidarFrame != baselinkFrame) {
      try {
        // 查询一下lidar和baselink之间的tf变换
        tfListener.waitForTransform(lidarFrame, baselinkFrame, ros::Time(0),
                                    ros::Duration(3.0));
        tfListener.lookupTransform(lidarFrame, baselinkFrame, ros::Time(0),
                                   lidar2Baselink);
      } catch (tf::TransformException ex) {
        ROS_ERROR("%s", ex.what());
      }
    }
    // step: 2 订阅地图优化节点的全局位姿和预积分节点的增量位姿
    subLaserOdometry = nh.subscribe<nav_msgs::Odometry>(
        "lio_sam/mapping/odometry", 5, &TransformFusion::lidarOdometryHandler,
        this, ros::TransportHints().tcpNoDelay());
    subImuOdometry = nh.subscribe<nav_msgs::Odometry>(
        odomTopic + "_incremental", 2000, &TransformFusion::imuOdometryHandler,
        this, ros::TransportHints().tcpNoDelay());

    // step: 3 发布融合优化odom和imu增量后的里程计以及轨迹
    pubImuOdometry = nh.advertise<nav_msgs::Odometry>(odomTopic, 2000);
    pubImuPath = nh.advertise<nav_msgs::Path>("lio_sam/imu/path", 1);
  }

  /**
   * \brief // api: nav_msgs里程计转Affine
   *
   * \param odom 消息
   * \return Eigen::Affine3f 转换后的位姿
   */
  Eigen::Affine3f odom2affine(nav_msgs::Odometry odom) {
    double x, y, z, roll, pitch, yaw;
    x = odom.pose.pose.position.x;
    y = odom.pose.pose.position.y;
    z = odom.pose.pose.position.z;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
    return pcl::getTransformation(x, y, z, roll, pitch, yaw);
  }

  /**
   * \brief // api: 将全局位姿保存下来
   *
   * \param odomMsg 消息
   */
  void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg) {
    std::lock_guard<std::mutex> lock(mtx);

    lidarOdomAffine = odom2affine(*odomMsg);

    lidarOdomTime = odomMsg->header.stamp.toSec();
  }

  /**
   * \brief // api: IMU预积分发布的incremental的里程计回调
   *
   * \param odomMsg 消息
   */
  void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg) {
    // step: 1 发送静态tf，odom系和map系将他们重合
    static tf::TransformBroadcaster tfMap2Odom;
    static tf::Transform map_to_odom = tf::Transform(
        tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));
    tfMap2Odom.sendTransform(tf::StampedTransform(
        map_to_odom, odomMsg->header.stamp, mapFrame, odometryFrame));

    // step: 2 没有优化位姿就return，弹出早于最新优化时刻之前的imu里程记数据
    std::lock_guard<std::mutex> lock(mtx);
    imuOdomQueue.push_back(*odomMsg);
    if (lidarOdomTime == -1) return;
    while (!imuOdomQueue.empty()) {
      if (imuOdomQueue.front().header.stamp.toSec() <= lidarOdomTime)
        imuOdomQueue.pop_front();
      else
        break;
    }

    // step: 3 计算最新队列里imu里程记的增量
    Eigen::Affine3f imuOdomAffineFront = odom2affine(imuOdomQueue.front());
    Eigen::Affine3f imuOdomAffineBack = odom2affine(imuOdomQueue.back());
    Eigen::Affine3f imuOdomAffineIncre =
        imuOdomAffineFront.inverse() * imuOdomAffineBack;

    // step: 4 增量补偿到优化位姿上去，就得到了最新的预测的位姿
    Eigen::Affine3f imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre;
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch,
                                      yaw);

    // step: 5 发送全局一致位姿的最新位姿
    nav_msgs::Odometry laserOdometry = imuOdomQueue.back();
    laserOdometry.pose.pose.position.x = x;
    laserOdometry.pose.pose.position.y = y;
    laserOdometry.pose.pose.position.z = z;
    laserOdometry.pose.pose.orientation =
        tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
    pubImuOdometry.publish(laserOdometry);

    // step: 6 更新tf
    static tf::TransformBroadcaster tfOdom2BaseLink;
    tf::Transform tCur;
    tf::poseMsgToTF(laserOdometry.pose.pose, tCur);
    if (lidarFrame != baselinkFrame) tCur = tCur * lidar2Baselink;
    // 更新odom到baselink的tf
    tf::StampedTransform odom_2_baselink = tf::StampedTransform(
        tCur, odomMsg->header.stamp, odometryFrame, baselinkFrame);
    tfOdom2BaseLink.sendTransform(odom_2_baselink);

    // step: 7 发送imu里程记的轨迹
    static nav_msgs::Path imuPath;
    static double last_path_time = -1;
    double imuTime = imuOdomQueue.back().header.stamp.toSec();
    // 控制一下更新频率，不超过10hz
    if (imuTime - last_path_time > 0.1) {
      last_path_time = imuTime;
      geometry_msgs::PoseStamped pose_stamped;
      pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
      pose_stamped.header.frame_id = odometryFrame;
      pose_stamped.pose = laserOdometry.pose.pose;
      // 将最新的位姿送入轨迹中
      imuPath.poses.push_back(pose_stamped);
      // 把lidar时间戳之前的轨迹全部擦除
      while (!imuPath.poses.empty() &&
             imuPath.poses.front().header.stamp.toSec() < lidarOdomTime - 1.0)
        imuPath.poses.erase(imuPath.poses.begin());
      // 发布轨迹，这个轨迹实际上是可视化imu预积分节点输出的预测值
      if (pubImuPath.getNumSubscribers() != 0) {
        imuPath.header.stamp = imuOdomQueue.back().header.stamp;
        imuPath.header.frame_id = odometryFrame;
        pubImuPath.publish(imuPath);
      }
    }
  }
};

// 预积分，因子图优化零偏和位姿
class IMUPreintegration : public ParamServer {
 public:
  std::mutex mtx;

  ros::Subscriber subImu;
  ros::Subscriber subOdometry;
  ros::Publisher pubImuOdometry;

  bool systemInitialized = false;

  gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
  gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
  gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
  gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
  gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
  gtsam::Vector noiseModelBetweenBias;

  gtsam::PreintegratedImuMeasurements* imuIntegratorOpt_;
  gtsam::PreintegratedImuMeasurements* imuIntegratorImu_;

  std::deque<sensor_msgs::Imu> imuQueOpt;
  std::deque<sensor_msgs::Imu> imuQueImu;

  gtsam::Pose3 prevPose_;
  gtsam::Vector3 prevVel_;
  gtsam::NavState prevState_;
  gtsam::imuBias::ConstantBias prevBias_;

  gtsam::NavState prevStateOdom;
  gtsam::imuBias::ConstantBias prevBiasOdom;

  bool doneFirstOpt = false;
  double lastImuT_imu = -1;
  double lastImuT_opt = -1;

  gtsam::ISAM2 optimizer;
  gtsam::NonlinearFactorGraph graphFactors;
  gtsam::Values graphValues;

  const double delta_t = 0;

  int key = 1;

  gtsam::Pose3 imu2Lidar =
      gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0),
                   gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
  gtsam::Pose3 lidar2Imu =
      gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0),
                   gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));

  /**
   * \brief // api: 构造函数
   *
   */
  IMUPreintegration() {
    // step: 1 订阅imu和后端的incremental的里程计
    subImu = nh.subscribe<sensor_msgs::Imu>(
        imuTopic, 2000, &IMUPreintegration::imuHandler, this,
        ros::TransportHints().tcpNoDelay());
    subOdometry = nh.subscribe<nav_msgs::Odometry>(
        "lio_sam/mapping/odometry_incremental", 5,
        &IMUPreintegration::odometryHandler, this,
        ros::TransportHints().tcpNoDelay());

    // step: 2 按照imu数据频率预测的incremental的里程计发布
    pubImuOdometry =
        nh.advertise<nav_msgs::Odometry>(odomTopic + "_incremental", 2000);

    // step: 3 预积分初始化
    // 初始化重力
    boost::shared_ptr<gtsam::PreintegrationParams> p =
        gtsam::PreintegrationParams::MakeSharedU(imuGravity);
    // 初始化协方差
    // acc white noise in continuous
    p->accelerometerCovariance =
        gtsam::Matrix33::Identity(3, 3) * pow(imuAccNoise, 2);
    // gyro white noise in continuous
    p->gyroscopeCovariance =
        gtsam::Matrix33::Identity(3, 3) * pow(imuGyrNoise, 2);
    // error committed in integrating position from velocities
    p->integrationCovariance = gtsam::Matrix33::Identity(3, 3) * pow(1e-4, 2);

    // 假设零偏为0
    // assume zero initial bias
    gtsam::imuBias::ConstantBias prior_imu_bias(
        (gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());

    // 初始位姿置信度设置比较高 rad,rad,rad,m, m, m
    priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished());
    // 初始速度置信度就设置差一些 m/s
    priorVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e4);
    // 零偏的置信度也设置高一些 1e-2 ~ 1e-3 seems to be good
    priorBiasNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);
    // rad,rad,rad,m, m, m
    correctionNoise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());
    // rad,rad,rad,m, m, m
    correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished());
    noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN,
                             imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN)
                                .finished();

    // 一个用来推测位姿，一个用来优化
    // setting up the IMU integration for IMU message thread
    imuIntegratorImu_ =
        new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias);
    // setting up the IMU integration for optimization
    imuIntegratorOpt_ =
        new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias);
  }

  /**
   * \brief // api: 复位优化
   *
   */
  void resetOptimization() {
    // gtsam初始化
    gtsam::ISAM2Params optParameters;
    optParameters.relinearizeThreshold = 0.1;
    optParameters.relinearizeSkip = 1;
    optimizer = gtsam::ISAM2(optParameters);

    // 因子图初始化
    gtsam::NonlinearFactorGraph newGraphFactors;
    graphFactors = newGraphFactors;

    gtsam::Values NewGraphValues;
    graphValues = NewGraphValues;
  }

  /**
   * \brief // api: 复位系统参数
   *
   */
  void resetParams() {
    lastImuT_imu = -1;
    doneFirstOpt = false;
    systemInitialized = false;
  }

  /**
   * \brief // api: 订阅地图优化节点的增量里程记消息
   *
   * \param odomMsg 消息
   */
  void odometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg) {
    std::lock_guard<std::mutex> lock(mtx);

    // step: 1 提取时间戳，并检查IMU数据
    double currentCorrectionTime = ROS_TIME(odomMsg);
    // make sure we have imu data to integrate
    // 确保imu队列中有数据
    if (imuQueOpt.empty()) return;

    // step: 2 获取里程记位姿，把位姿转成gtsam的格式
    float p_x = odomMsg->pose.pose.position.x;
    float p_y = odomMsg->pose.pose.position.y;
    float p_z = odomMsg->pose.pose.position.z;
    float r_x = odomMsg->pose.pose.orientation.x;
    float r_y = odomMsg->pose.pose.orientation.y;
    float r_z = odomMsg->pose.pose.orientation.z;
    float r_w = odomMsg->pose.pose.orientation.w;
    // 该位姿是否出现退化
    bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false;
    gtsam::Pose3 lidarPose =
        gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z),
                     gtsam::Point3(p_x, p_y, p_z));

    // step: 3 首先初始化系统
    // initialize system
    if (systemInitialized == false) {
      // step: 3.1 优化问题进行复位
      resetOptimization();

      // pop old IMU message
      // step: 3.2 将这个里程记消息之前的imu信息全部扔掉
      while (!imuQueOpt.empty()) {
        if (ROS_TIME(&imuQueOpt.front()) < currentCorrectionTime - delta_t) {
          lastImuT_opt = ROS_TIME(&imuQueOpt.front());
          imuQueOpt.pop_front();
        } else
          break;
      }

      // step: 3.3 将lidar的位姿转移到imu坐标系下，设置其初始位姿和置信度
      prevPose_ = lidarPose.compose(lidar2Imu);
      gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_,
                                                 priorPoseNoise);
      graphFactors.add(priorPose);

      // step: 3.4 初始化速度，这里就直接赋0了
      prevVel_ = gtsam::Vector3(0, 0, 0);
      gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_,
                                                  priorVelNoise);
      graphFactors.add(priorVel);

      // step: 3.5 初始化零偏
      prevBias_ = gtsam::imuBias::ConstantBias();
      gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(
          B(0), prevBias_, priorBiasNoise);
      graphFactors.add(priorBias);

      // step: 3.6 以上把约束加入完毕，下面开始添加状态量
      // 将各个状态量赋成初始值
      graphValues.insert(X(0), prevPose_);
      graphValues.insert(V(0), prevVel_);
      graphValues.insert(B(0), prevBias_);

      // step: 3.7 optimize once 约束和状态量更新进isam优化器
      optimizer.update(graphFactors, graphValues);

      // step: 3.8 进优化器之后保存约束和状态量的变量就清零
      graphFactors.resize(0);
      graphValues.clear();

      // step: 3.9 预积分的接口，使用初始零偏进行初始化
      imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
      imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

      key = 1;
      systemInitialized = true;
      return;
    }

    // step: 4 isam优化器中加入了较多的约束后，避免资源消耗过大直接清空
    if (key == 100) {
      // step: 4.1 取出最新时刻位姿 速度 零偏的协方差矩阵
      gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise =
          gtsam::noiseModel::Gaussian::Covariance(
              optimizer.marginalCovariance(X(key - 1)));
      gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise =
          gtsam::noiseModel::Gaussian::Covariance(
              optimizer.marginalCovariance(V(key - 1)));
      gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise =
          gtsam::noiseModel::Gaussian::Covariance(
              optimizer.marginalCovariance(B(key - 1)));

      // step: 4.2 复位整个优化问题
      resetOptimization();

      // step: 4.3 将最新的位姿，速度，零偏以及对应的协方差矩阵加入到因子图中
      // 添加先验约束
      gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_,
                                                 updatedPoseNoise);
      graphFactors.add(priorPose);
      gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_,
                                                  updatedVelNoise);
      graphFactors.add(priorVel);
      gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(
          B(0), prevBias_, updatedBiasNoise);
      graphFactors.add(priorBias);
      // 添加状态值
      graphValues.insert(X(0), prevPose_);
      graphValues.insert(V(0), prevVel_);
      graphValues.insert(B(0), prevBias_);

      // step: 4.4 优化一次，保存因子图数据
      optimizer.update(graphFactors, graphValues);
      graphFactors.resize(0);
      graphValues.clear();

      key = 1;
    }

    // 1. integrate imu data and optimize
    // step: 5 将两帧之间的imu做积分
    while (!imuQueOpt.empty()) {
      // pop and integrate imu data that is between two optimizations
      // step: 5.1 时间上小于当前lidar位姿的imu消息都取出来
      sensor_msgs::Imu* thisImu = &imuQueOpt.front();
      double imuTime = ROS_TIME(thisImu);
      if (imuTime < currentCorrectionTime - delta_t) {
        // 计算两个imu量之间的时间差
        double dt =
            (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
        // 调用预积分接口将imu数据送进去处理
        imuIntegratorOpt_->integrateMeasurement(
            gtsam::Vector3(thisImu->linear_acceleration.x,
                           thisImu->linear_acceleration.y,
                           thisImu->linear_acceleration.z),
            gtsam::Vector3(thisImu->angular_velocity.x,
                           thisImu->angular_velocity.y,
                           thisImu->angular_velocity.z),
            dt);
        // 记录当前imu时间
        lastImuT_opt = imuTime;
        imuQueOpt.pop_front();
      } else
        break;
    }

    // add imu factor to graph
    // step: 5.2 两帧间imu预积分完成之后，就将其转换成预积分约束
    // note: 预积分约束对相邻两帧之间的位姿 速度 零偏形成约束
    const gtsam::PreintegratedImuMeasurements& preint_imu =
        dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(
            *imuIntegratorOpt_);
    gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key),
                                B(key - 1), preint_imu);
    graphFactors.add(imu_factor);

    // add imu bias between factor
    // step: 5.3 零偏的约束，两帧间零偏相差不会太大使用常量约束
    graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
        B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
        gtsam::noiseModel::Diagonal::Sigmas(
            sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));

    // step: 5.4 lidar位姿补偿到imu坐标系下作为这一帧的先验估计
    // note: 根据是否退化选择不同的置信度
    gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
    gtsam::PriorFactor<gtsam::Pose3> pose_factor(
        X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
    graphFactors.add(pose_factor);

    // step: 5.5 根据上一时刻的状态对当前状态进行预测，并插入因子图
    gtsam::NavState propState_ =
        imuIntegratorOpt_->predict(prevState_, prevBias_);
    graphValues.insert(X(key), propState_.pose());
    graphValues.insert(V(key), propState_.v());
    graphValues.insert(B(key), prevBias_);

    // step: 5.6 执行优化并清空因子图
    optimizer.update(graphFactors, graphValues);
    optimizer.update();
    graphFactors.resize(0);
    graphValues.clear();

    // step: 5.7 获取优化后的当前状态作为当前帧的最佳估计
    gtsam::Values result = optimizer.calculateEstimate();
    prevPose_ = result.at<gtsam::Pose3>(X(key));
    prevVel_ = result.at<gtsam::Vector3>(V(key));
    prevState_ = gtsam::NavState(prevPose_, prevVel_);
    prevBias_ = result.at<gtsam::imuBias::ConstantBias>(B(key));

    // step: 5.8 预积分约束复位，同时需要设置一下零偏作为下一次积分的先决条件
    imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

    // step: 5.9 一个简单的失败检测
    if (failureDetection(prevVel_, prevBias_)) {
      // 状态异常就直接复位了
      resetParams();
      return;
    }

    // step: 6 优化之后，根据最新的imu状态进行传播
    prevStateOdom = prevState_;
    prevBiasOdom = prevBias_;
    // step: 6.1 首先把lidar帧之前的imu状态全部弹出去
    double lastImuQT = -1;
    while (!imuQueImu.empty() &&
           ROS_TIME(&imuQueImu.front()) < currentCorrectionTime - delta_t) {
      lastImuQT = ROS_TIME(&imuQueImu.front());
      imuQueImu.pop_front();
    }
    // step: 6.2 如果有新于lidar状态时刻的imu
    if (!imuQueImu.empty()) {
      // reset bias use the newly optimized bias
      // step: 6.2.1 预积分变量复位
      imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
      // step: 6.2.2 把优化时间后的imu状态积分，避免IMU回调函数中积分的部分丢失
      for (int i = 0; i < (int)imuQueImu.size(); ++i) {
        sensor_msgs::Imu* thisImu = &imuQueImu[i];
        double imuTime = ROS_TIME(thisImu);
        double dt = (lastImuQT < 0) ? (1.0 / 500.0) : (imuTime - lastImuQT);
        imuIntegratorImu_->integrateMeasurement(
            gtsam::Vector3(thisImu->linear_acceleration.x,
                           thisImu->linear_acceleration.y,
                           thisImu->linear_acceleration.z),
            gtsam::Vector3(thisImu->angular_velocity.x,
                           thisImu->angular_velocity.y,
                           thisImu->angular_velocity.z),
            dt);
        lastImuQT = imuTime;
      }
    }

    ++key;
    doneFirstOpt = true;
  }

  /**
   * \brief // api: 状态失效检测
   *
   * \param velCur 速度
   * \param biasCur 零偏
   * \return true
   * \return false
   */
  bool failureDetection(const gtsam::Vector3& velCur,
                        const gtsam::imuBias::ConstantBias& biasCur) {
    Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
    // 如果当前速度大于30m/s，108km/h就认为是异常状态，
    if (vel.norm() > 30) {
      ROS_WARN("Large velocity, reset IMU-preintegration!");
      return true;
    }

    Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(),
                       biasCur.accelerometer().z());
    Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(),
                       biasCur.gyroscope().z());
    // 如果零偏太大，那也不太正常
    if (ba.norm() > 1.0 || bg.norm() > 1.0) {
      ROS_WARN("Large bias, reset IMU-preintegration!");
      return true;
    }

    return false;
  }

  /**
   * \brief // api: IMU回调
   *
   * \param imu_raw 消息
   */
  void imuHandler(const sensor_msgs::Imu::ConstPtr& imu_raw) {
    std::lock_guard<std::mutex> lock(mtx);
    // step: 1 首先把imu的状态做一个简单的转换，前左上
    sensor_msgs::Imu thisImu = imuConverter(*imu_raw);

    // note: 注意有两个imu的队列，一个预积分和位姿的优化，一个更新最新imu状态
    imuQueOpt.push_back(thisImu);
    imuQueImu.push_back(thisImu);
    // step: 2 如果没有发生过优化就return
    if (doneFirstOpt == false) return;

    // step: 3 每来一个imu值就加入预积分状态中
    double imuTime = ROS_TIME(&thisImu);
    double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
    lastImuT_imu = imuTime;
    imuIntegratorImu_->integrateMeasurement(
        gtsam::Vector3(thisImu.linear_acceleration.x,
                       thisImu.linear_acceleration.y,
                       thisImu.linear_acceleration.z),
        gtsam::Vector3(thisImu.angular_velocity.x, thisImu.angular_velocity.y,
                       thisImu.angular_velocity.z),
        dt);

    // step: 4 根据这个值预测最新的状态
    gtsam::NavState currentState =
        imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

    // step: 5 将这个状态转到lidar坐标系下去发送出去
    nav_msgs::Odometry odometry;
    odometry.header.stamp = thisImu.header.stamp;
    odometry.header.frame_id = odometryFrame;
    odometry.child_frame_id = "odom_imu";
    gtsam::Pose3 imuPose =
        gtsam::Pose3(currentState.quaternion(), currentState.position());
    gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);
    odometry.pose.pose.position.x = lidarPose.translation().x();
    odometry.pose.pose.position.y = lidarPose.translation().y();
    odometry.pose.pose.position.z = lidarPose.translation().z();
    odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
    odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
    odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
    odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();
    odometry.twist.twist.linear.x = currentState.velocity().x();
    odometry.twist.twist.linear.y = currentState.velocity().y();
    odometry.twist.twist.linear.z = currentState.velocity().z();
    odometry.twist.twist.angular.x =
        thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
    odometry.twist.twist.angular.y =
        thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
    odometry.twist.twist.angular.z =
        thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
    pubImuOdometry.publish(odometry);
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "roboat_loam");

  IMUPreintegration ImuP;

  TransformFusion TF;

  ROS_INFO("\033[1;32m----> IMU Preintegration Started.\033[0m");

  ros::MultiThreadedSpinner spinner(4);
  spinner.spin();

  return 0;
}
