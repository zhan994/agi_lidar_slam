#include "lio_sam/cloud_info.h"
#include "lio_sam/save_map.h"
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

using namespace gtsam;

using symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G;  // GPS pose
using symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)

/*
 * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is
 * time stamp)
 */
struct PointXYZIRPYT {
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY;  // preferred way of adding a XYZ+padding
  float roll;
  float pitch;
  float yaw;
  double time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // make sure our new allocators are aligned
} EIGEN_ALIGN16;  // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZIRPYT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
        float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, time,
                                                                 time))

typedef PointXYZIRPYT PointTypePose;

class mapOptimization : public ParamServer {
 public:
  // gtsam
  NonlinearFactorGraph gtSAMgraph;
  Values initialEstimate;
  Values optimizedEstimate;
  ISAM2* isam;
  Values isamCurrentEstimate;
  Eigen::MatrixXd poseCovariance;

  ros::Publisher pubLaserCloudSurround;
  ros::Publisher pubLaserOdometryGlobal;
  ros::Publisher pubLaserOdometryIncremental;
  ros::Publisher pubKeyPoses;
  ros::Publisher pubPath;

  ros::Publisher pubHistoryKeyFrames;
  ros::Publisher pubIcpKeyFrames;
  ros::Publisher pubRecentKeyFrames;
  ros::Publisher pubRecentKeyFrame;
  ros::Publisher pubCloudRegisteredRaw;
  ros::Publisher pubLoopConstraintEdge;

  ros::Subscriber subCloud;
  ros::Subscriber subGPS;
  ros::Subscriber subLoop;

  ros::ServiceServer srvSaveMap;

  std::deque<nav_msgs::Odometry> gpsQueue;
  lio_sam::cloud_info cloudInfo;

  vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
  vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;

  //  存储关键帧的位置信息的点云
  pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
  // 存储关键帧的6D位姿信息的点云
  pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
  pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
  pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

  // corner feature set from odoOptimization
  pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
  // surf feature set from odoOptimization
  pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;
  // downsampled corner featuer set from odoOptimization
  pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS;
  // downsampled surf featuer set from odoOptimization
  pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS;

  pcl::PointCloud<PointType>::Ptr laserCloudOri;
  pcl::PointCloud<PointType>::Ptr coeffSel;

  // corner point holder for parallel computation
  std::vector<PointType> laserCloudOriCornerVec;
  std::vector<PointType> coeffSelCornerVec;
  std::vector<bool> laserCloudOriCornerFlag;

  // surf point holder for parallel computation
  std::vector<PointType> laserCloudOriSurfVec;
  std::vector<PointType> coeffSelSurfVec;
  std::vector<bool> laserCloudOriSurfFlag;

  // 局部地图的一个容器
  map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>>
      laserCloudMapContainer;
  pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
  pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
  // 角点局部地图的下采样后的点云
  pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
  // 面点局部地图的下采样后的点云
  pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

  // 角点局部地图的kdtree
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
  // 面点局部地图的kdtree
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

  pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

  pcl::VoxelGrid<PointType> downSizeFilterCorner;
  pcl::VoxelGrid<PointType> downSizeFilterSurf;
  pcl::VoxelGrid<PointType> downSizeFilterICP;
  // for surrounding key poses of scan-to-map optimization
  pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses;

  ros::Time timeLaserInfoStamp;
  double timeLaserInfoCur;

  float transformTobeMapped[6];

  std::mutex mtx;
  std::mutex mtxLoopInfo;

  bool isDegenerate = false;
  cv::Mat matP;

  // 当前局部地图下采样后的角点数目
  int laserCloudCornerFromMapDSNum = 0;
  // 当前局部地图下采样后的面点数目
  int laserCloudSurfFromMapDSNum = 0;
  // 当前帧下采样后的角点的数目
  int laserCloudCornerLastDSNum = 0;
  // 当前帧下采样后的面点数目
  int laserCloudSurfLastDSNum = 0;

  bool aLoopIsClosed = false;
  // 回环检测存储容器 key是当前 val是较早
  map<int, int> loopIndexContainer;
  vector<pair<int, int>> loopIndexQueue;
  vector<gtsam::Pose3> loopPoseQueue;
  vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
  deque<std_msgs::Float64MultiArray> loopInfoVec;

  nav_msgs::Path globalPath;

  Eigen::Affine3f transPointAssociateToMap;
  Eigen::Affine3f incrementalOdometryAffineFront;
  Eigen::Affine3f incrementalOdometryAffineBack;

  /**
   * \brief // api: 构造函数
   *
   */
  mapOptimization() {
    // step: 1 isam2参数
    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);

    // step: 2 订阅特征提取后的点云信息、gps和回环检测
    subCloud = nh.subscribe<lio_sam::cloud_info>(
        "lio_sam/feature/cloud_info", 1,
        &mapOptimization::laserCloudInfoHandler, this,
        ros::TransportHints().tcpNoDelay());
    subGPS = nh.subscribe<nav_msgs::Odometry>(
        gpsTopic, 200, &mapOptimization::gpsHandler, this,
        ros::TransportHints().tcpNoDelay());
    subLoop = nh.subscribe<std_msgs::Float64MultiArray>(
        "lio_loop/loop_closure_detection", 1, &mapOptimization::loopInfoHandler,
        this, ros::TransportHints().tcpNoDelay());

    // step: 3 订阅一个保存地图功能的服务
    srvSaveMap = nh.advertiseService("lio_sam/save_map",
                                     &mapOptimization::saveMapService, this);

    // step: 4 发布关键帧位置、点云地图、优化后里程计、incremental里程计
    pubKeyPoses =
        nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/trajectory", 1);
    pubLaserCloudSurround =
        nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_global", 1);
    pubLaserOdometryGlobal =
        nh.advertise<nav_msgs::Odometry>("lio_sam/mapping/odometry", 1);
    pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry>(
        "lio_sam/mapping/odometry_incremental", 1);
    pubPath = nh.advertise<nav_msgs::Path>("lio_sam/mapping/path", 1);

    // step: 5 回环局部地图、回环修正后点云、回环约束
    pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>(
        "lio_sam/mapping/icp_loop_closure_history_cloud", 1);
    pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>(
        "lio_sam/mapping/icp_loop_closure_corrected_cloud", 1);
    pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>(
        "/lio_sam/mapping/loop_closure_constraints", 1);

    // step: 6 发布局部点云地图、地图系当前特征点云、地图系当前原始点云
    pubRecentKeyFrames =
        nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_local", 1);
    pubRecentKeyFrame = nh.advertise<sensor_msgs::PointCloud2>(
        "lio_sam/mapping/cloud_registered", 1);
    pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>(
        "lio_sam/mapping/cloud_registered_raw", 1);

    // step: 7 体素滤波设置珊格大小
    downSizeFilterCorner.setLeafSize(
        mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
    downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize,
                                   mappingSurfLeafSize);
    downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize,
                                  mappingSurfLeafSize);
    downSizeFilterSurroundingKeyPoses.setLeafSize(
        surroundingKeyframeDensity, surroundingKeyframeDensity,
        surroundingKeyframeDensity);  // for surrounding key poses of
                                      // scan-to-map optimization

    // step: 8 分配内存
    allocateMemory();
  }

  /**
   * \brief // api: 预先分配内存
   *
   */
  void allocateMemory() {
    cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
    copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

    kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

    laserCloudCornerLast.reset(
        new pcl::PointCloud<PointType>());  // corner feature set from
                                            // odoOptimization
    laserCloudSurfLast.reset(
        new pcl::PointCloud<PointType>());  // surf feature set from
                                            // odoOptimization
    laserCloudCornerLastDS.reset(
        new pcl::PointCloud<PointType>());  // downsampled corner featuer set
                                            // from odoOptimization
    laserCloudSurfLastDS.reset(
        new pcl::PointCloud<PointType>());  // downsampled surf featuer set from
                                            // odoOptimization

    laserCloudOri.reset(new pcl::PointCloud<PointType>());
    coeffSel.reset(new pcl::PointCloud<PointType>());

    laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
    coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
    laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
    laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
    coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
    laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

    std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(),
              false);
    std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(),
              false);

    laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
    laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

    kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

    for (int i = 0; i < 6; ++i) {
      transformTobeMapped[i] = 0;
    }

    matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
  }

  /**
   * \brief // api: 点云数据回调
   *
   * \param msgIn 消息
   */
  void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr& msgIn) {
    // step: 1 提取当前时间戳
    timeLaserInfoStamp = msgIn->header.stamp;
    timeLaserInfoCur = msgIn->header.stamp.toSec();

    // step: 2 提取cloudinfo中的角点和面点
    cloudInfo = *msgIn;
    pcl::fromROSMsg(msgIn->cloud_corner, *laserCloudCornerLast);
    pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);

    std::lock_guard<std::mutex> lock(mtx);

    static double timeLastProcessing = -1;
    // note: 控制后端频率，两帧处理一帧
    if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval) {
      timeLastProcessing = timeLaserInfoCur;
      // step: 3 更新当前匹配结果的初始位姿
      updateInitialGuess();

      // step: 4 提取当前帧相关的关键帧并且构建点云局部地图
      extractSurroundingKeyFrames();

      // step: 5 对当前帧进行下采样
      downsampleCurrentScan();

      // step: 6 对点云配准进行优化问题构建求解
      scan2MapOptimization();

      // step: 7 根据配准结果确定是否是关键帧
      saveKeyFramesAndFactor();

      // step: 8 调整全局轨迹
      correctPoses();

      // step: 9 将lidar里程记信息发送出去
      publishOdometry();

      // step: 10 发送可视化点云信息
      publishFrames();
    }
  }

  /**
   * \brief // api: 收集gps信息
   *
   * \param gpsMsg 消息
   */
  void gpsHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg) {
    gpsQueue.push_back(*gpsMsg);
  }

  /**
   * \brief // api: 点云转换到地图坐标系下
   *
   * \param pi 输入
   * \param po 输出
   */
  void pointAssociateToMap(PointType const* const pi, PointType* const po) {
    po->x = transPointAssociateToMap(0, 0) * pi->x +
            transPointAssociateToMap(0, 1) * pi->y +
            transPointAssociateToMap(0, 2) * pi->z +
            transPointAssociateToMap(0, 3);
    po->y = transPointAssociateToMap(1, 0) * pi->x +
            transPointAssociateToMap(1, 1) * pi->y +
            transPointAssociateToMap(1, 2) * pi->z +
            transPointAssociateToMap(1, 3);
    po->z = transPointAssociateToMap(2, 0) * pi->x +
            transPointAssociateToMap(2, 1) * pi->y +
            transPointAssociateToMap(2, 2) * pi->z +
            transPointAssociateToMap(2, 3);
    po->intensity = pi->intensity;
  }

  /**
   * \brief // api: 点云坐标变换
   *
   * \param cloudIn 输入
   * \param transformIn 位姿
   * \return pcl::PointCloud<PointType>::Ptr 输出
   */
  pcl::PointCloud<PointType>::Ptr transformPointCloud(
      pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn) {
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(
        transformIn->x, transformIn->y, transformIn->z, transformIn->roll,
        transformIn->pitch, transformIn->yaw);
// 使用openmp进行并行加速
#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i) {
      const auto& pointFrom = cloudIn->points[i];
      // 每个点都施加RX+t这样一个过程
      cloudOut->points[i].x = transCur(0, 0) * pointFrom.x +
                              transCur(0, 1) * pointFrom.y +
                              transCur(0, 2) * pointFrom.z + transCur(0, 3);
      cloudOut->points[i].y = transCur(1, 0) * pointFrom.x +
                              transCur(1, 1) * pointFrom.y +
                              transCur(1, 2) * pointFrom.z + transCur(1, 3);
      cloudOut->points[i].z = transCur(2, 0) * pointFrom.x +
                              transCur(2, 1) * pointFrom.y +
                              transCur(2, 2) * pointFrom.z + transCur(2, 3);
      cloudOut->points[i].intensity = pointFrom.intensity;
    }
    return cloudOut;
  }

  /**
   * \brief // api: pcl格式位姿转gtsam
   *
   * \param thisPoint pcl输入
   * \return gtsam::Pose3 gtsam输出
   */
  gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint) {
    return gtsam::Pose3(
        gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch),
                            double(thisPoint.yaw)),
        gtsam::Point3(double(thisPoint.x), double(thisPoint.y),
                      double(thisPoint.z)));
  }

  /**
   * \brief // api: 转成gtsam的数据结构
   *
   * \param transformIn 数组输入
   * \return gtsam::Pose3 gtsam输出
   */
  gtsam::Pose3 trans2gtsamPose(float transformIn[]) {
    return gtsam::Pose3(
        gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
        gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
  }

  /**
   * \brief // api: pcl格式位姿转eigen
   *
   * \param thisPoint pcl输入
   * \return Eigen::Affine3f eigen输出
   */
  Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint) {
    return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z,
                                  thisPoint.roll, thisPoint.pitch,
                                  thisPoint.yaw);
  }

  /**
   * \brief // api: 转成eigen的位姿数据结构
   *
   * \param transformIn 数组输入
   * \return Eigen::Affine3f eigen输出
   */
  Eigen::Affine3f trans2Affine3f(float transformIn[]) {
    return pcl::getTransformation(transformIn[3], transformIn[4],
                                  transformIn[5], transformIn[0],
                                  transformIn[1], transformIn[2]);
  }

  /**
   * \brief // api: 转成pcl的位姿数据结构
   *
   * \param transformIn 数组输入
   * \return PointTypePose pcl输出
   */
  PointTypePose trans2PointTypePose(float transformIn[]) {
    PointTypePose thisPose6D;
    thisPose6D.x = transformIn[3];
    thisPose6D.y = transformIn[4];
    thisPose6D.z = transformIn[5];
    thisPose6D.roll = transformIn[0];
    thisPose6D.pitch = transformIn[1];
    thisPose6D.yaw = transformIn[2];
    return thisPose6D;
  }

  /**
   * \brief // api: 保存地图服务
   *
   * \param req 请求
   * \param res 回应
   * \return true
   * \return false
   */
  bool saveMapService(lio_sam::save_mapRequest& req,
                      lio_sam::save_mapResponse& res) {
    string saveMapDirectory;

    cout << "****************************************************" << endl;
    cout << "Saving map to pcd files ..." << endl;
    // step: 1 空说明是程序结束的自动保存，否则是中途调用服务
    if (req.destination.empty())
      saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
    else
      saveMapDirectory = std::getenv("HOME") + req.destination;
    cout << "Save destination: " << saveMapDirectory << endl;

    // step: 2 删掉之前有可能保存过的地图
    int unused =
        system((std::string("exec rm -r ") + saveMapDirectory).c_str());
    unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());

    // step: 3 首先保存关键帧轨迹
    pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd",
                               *cloudKeyPoses3D);
    pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd",
                               *cloudKeyPoses6D);
    // extract global point cloud map
    pcl::PointCloud<PointType>::Ptr globalCornerCloud(
        new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(
        new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalSurfCloud(
        new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(
        new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapCloud(
        new pcl::PointCloud<PointType>());
    // 遍历所有关键帧，将点云全部转移到世界坐标系下去
    for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
      *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],
                                                 &cloudKeyPoses6D->points[i]);
      *globalSurfCloud += *transformPointCloud(surfCloudKeyFrames[i],
                                               &cloudKeyPoses6D->points[i]);
      // 类似进度条的功能
      cout << "\r" << std::flush << "Processing feature cloud " << i << " of "
           << cloudKeyPoses6D->size() << " ...";
    }

    // step: 4 如果没有指定分辨率，就是直接保存
    if (req.resolution != 0) {
      cout << "\n\nSave resolution: " << req.resolution << endl;

      // 使用指定分辨率降采样，分别保存角点地图和面点地图
      downSizeFilterCorner.setInputCloud(globalCornerCloud);
      downSizeFilterCorner.setLeafSize(req.resolution, req.resolution,
                                       req.resolution);
      downSizeFilterCorner.filter(*globalCornerCloudDS);
      pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd",
                                 *globalCornerCloudDS);

      downSizeFilterSurf.setInputCloud(globalSurfCloud);
      downSizeFilterSurf.setLeafSize(req.resolution, req.resolution,
                                     req.resolution);
      downSizeFilterSurf.filter(*globalSurfCloudDS);
      pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd",
                                 *globalSurfCloudDS);
    } else {
      pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd",
                                 *globalCornerCloud);
      pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd",
                                 *globalSurfCloud);
    }

    // step: 5 保存全局地图
    *globalMapCloud += *globalCornerCloud;
    *globalMapCloud += *globalSurfCloud;
    int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd",
                                         *globalMapCloud);
    res.success = ret == 0;

    // step: 6 复位下采样参数
    downSizeFilterCorner.setLeafSize(
        mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
    downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize,
                                   mappingSurfLeafSize);

    cout << "****************************************************" << endl;
    cout << "Saving map to pcd files completed\n" << endl;

    return true;
  }

  /**
   * \brief // api: 全局可视化线程
   *
   */
  void visualizeGlobalMapThread() {
    // step: 1 更新频率设置为0.2hz
    ros::Rate rate(0.2);
    while (ros::ok()) {
      rate.sleep();
      publishGlobalMap();
    }

    // step: 2 当ros被杀死之后，执行保存地图功能
    if (savePCD == false) return;
    lio_sam::save_mapRequest req;
    lio_sam::save_mapResponse res;
    if (!saveMapService(req, res)) {
      cout << "Fail to save map" << endl;
    }
  }

  /**
   * \brief // api: 发布可视化全局地图
   *
   */
  void publishGlobalMap() {
    // step: 1 如果没有订阅者就不发布，节省系统负载
    if (pubLaserCloudSurround.getNumSubscribers() == 0) return;

    // step: 2 没有关键帧自然也没有全局地图了
    if (cloudKeyPoses3D->points.empty() == true) return;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(
        new pcl::KdTreeFLANN<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(
        new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(
        new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(
        new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(
        new pcl::PointCloud<PointType>());

    // step:3 搜索附近的数据进行可视化
    std::vector<int> pointSearchIndGlobalMap;
    std::vector<float> pointSearchSqDisGlobalMap;
    mtx.lock();
    // step: 3.1 把所有关键帧送入kdtree
    kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);

    // step: 3.2 寻找具体最新关键帧一定范围内的其他关键帧
    kdtreeGlobalMap->radiusSearch(
        cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius,
        pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
    mtx.unlock();

    // step: 3.3 把这些找到的关键帧的位姿保存起来
    for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
      globalMapKeyPoses->push_back(
          cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);

    // step: 3.4 简单的下采样
    pcl::VoxelGrid<PointType>
        downSizeFilterGlobalMapKeyPoses;  // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setLeafSize(
        globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity,
        globalMapVisualizationPoseDensity);  // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
    for (auto& pt : globalMapKeyPosesDS->points) {
      kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap,
                                      pointSearchSqDisGlobalMap);
      // 找到这些下采样后的关键帧的索引，并保存下来
      pt.intensity =
          cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
    }

    // step: 4 extract visualized and downsampled key frames
    for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i) {
      if (pointDistance(globalMapKeyPosesDS->points[i],
                        cloudKeyPoses3D->back()) >
          globalMapVisualizationSearchRadius)
        continue;

      // step: 4.1 将每一帧的点云通过位姿转到世界坐标系下
      int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;

      *globalMapKeyFrames +=
          *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],
                               &cloudKeyPoses6D->points[thisKeyInd]);
      *globalMapKeyFrames += *transformPointCloud(
          surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
    }

    // step: 4.2 转换后的点云也进行一个下采样
    pcl::VoxelGrid<PointType>
        downSizeFilterGlobalMapKeyFrames;  // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setLeafSize(
        globalMapVisualizationLeafSize, globalMapVisualizationLeafSize,
        globalMapVisualizationLeafSize);  // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
    downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
    // 最终发布出去
    publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS,
                 timeLaserInfoStamp, odometryFrame);
  }

  /**
   * \brief // api: 回环检测线程
   *
   */
  void loopClosureThread() {
    // step: 1 如果不需要进行回环检测，那么就退出这个线程
    if (loopClosureEnableFlag == false) return;

    // step: 2 设置回环检测的频率
    ros::Rate rate(loopClosureFrequency);
    while (ros::ok()) {
      // step: 3 执行完一次就必须sleep一段时间，否则该线程的cpu占用会非常高
      rate.sleep();

      // step: 4 执行回环检测
      performLoopClosure();

      // step: 5 可视化回环
      visualizeLoopClosure();
    }
  }

  /**
   * \brief // api: 接收外部告知的回环信息
   *
   * \param loopMsg 消息
   */
  void loopInfoHandler(const std_msgs::Float64MultiArray::ConstPtr& loopMsg) {
    std::lock_guard<std::mutex> lock(mtxLoopInfo);
    // step: 1 回环信息必须是配对的，因此大小检查一下是不是2
    if (loopMsg->data.size() != 2) return;

    // step: 2 把当前回环信息送进队列用于外部指定回环
    loopInfoVec.push_back(*loopMsg);

    // step: 3 如果队列里回环信息太多没有处理，就把老的回环信息扔掉
    while (loopInfoVec.size() > 5) loopInfoVec.pop_front();
  }

  /**
   * \brief // api: 执行回环检测
   *
   */
  void performLoopClosure() {
    // step: 1 如果没有关键帧，就没法进行回环检测了
    if (cloudKeyPoses3D->points.empty() == true) return;

    // step: 2 把存储关键帧的位姿的点云copy出来，避免线程冲突
    mtx.lock();
    *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
    *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
    mtx.unlock();

    // find keys
    int loopKeyCur;
    int loopKeyPre;
    // step: 3 首先看一下外部通知的回环信息
    if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false)
      // step: 4 然后根据里程记的距离来检测回环
      if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false) return;

    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(
        new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(
        new pcl::PointCloud<PointType>());
    {
      // step: 5 稍晚的帧就把自己取了出来
      loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);

      // step: 6 稍早一点的就把自己和周围一些点云取出来
      loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre,
                            historyKeyframeSearchNum);

      // step: 7 如果点云数目太少就算了
      if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
        return;

      // step: 8 把局部地图发布出去供rviz可视化使用
      if (pubHistoryKeyFrames.getNumSubscribers() != 0)
        publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud,
                     timeLaserInfoStamp, odometryFrame);
    }

    // step: 9 使用简单的icp来进行帧到局部地图的配准
    static pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius * 2);
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(prevKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(
        new pcl::PointCloud<PointType>());
    icp.align(*unused_result);
    // note: 检查icp是否收敛且得分是否满足要求
    if (icp.hasConverged() == false ||
        icp.getFitnessScore() > historyKeyframeFitnessScore)
      return;

    // step: 10 把修正后的当前点云发布供可视化使用
    if (pubIcpKeyFrames.getNumSubscribers() != 0) {
      pcl::PointCloud<PointType>::Ptr closed_cloud(
          new pcl::PointCloud<PointType>());
      pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud,
                               icp.getFinalTransformation());
      publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp,
                   odometryFrame);
    }

    // step: 11 获得两个点云的变换矩阵结果
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();

    // step: 12 找到稍晚点云的位姿结果，将icp的结果补偿过去
    Eigen::Affine3f tWrong =
        pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;
    pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
    gtsam::Pose3 poseFrom =
        Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));

    // step: 13 to是修正前的稍早帧的点云位姿
    gtsam::Pose3 poseTo =
        pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);

    // step: 14 使用icp的得分作为他们的约束的噪声项
    gtsam::Vector Vector6(6);
    float noiseScore = icp.getFitnessScore();
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore,
        noiseScore;
    noiseModel::Diagonal::shared_ptr constraintNoise =
        noiseModel::Diagonal::Variances(Vector6);

    // step: 15 将两帧索引，两帧相对位姿和噪声作为回环约束送入队列
    mtx.lock();
    loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
    loopPoseQueue.push_back(poseFrom.between(poseTo));
    loopNoiseQueue.push_back(constraintNoise);
    mtx.unlock();

    // step: 16 保存已存在的约束对
    loopIndexContainer[loopKeyCur] = loopKeyPre;
  }

  /**
   * \brief // api: 根据距离和时间进行回环检测
   *
   * \param latestID 当前帧id
   * \param closestID 回环帧id
   * \return true
   * \return false
   */
  bool detectLoopClosureDistance(int* latestID, int* closestID) {
    // step: 1 检测最新帧是否和其他帧形成回环，id就是最后一个关键帧
    int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
    int loopKeyPre = -1;

    // step: 2 检查一下较晚帧是否和别的形成了回环，如果有就算了
    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end()) return false;

    std::vector<int> pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop;
    // step: 3 把只包含关键帧位移信息的点云填充kdtree
    kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);

    // step: 4 根据最后一个关键帧的平移信息，寻找离他一定距离内的其他关键帧
    kdtreeHistoryKeyPoses->radiusSearch(
        copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius,
        pointSearchIndLoop, pointSearchSqDisLoop, 0);

    // step: 5 遍历找到的候选关键帧
    for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i) {
      int id = pointSearchIndLoop[i];
      // note: 必须满足时间上超过一定阈值，才认为是一个有效的回环
      if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) >
          historyKeyframeSearchTimeDiff) {
        loopKeyPre = id;
        break;
      }
    }

    // step: 6 如果没有找到回环或者回环找到自己，此次回环寻找失败
    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre) return false;

    *latestID = loopKeyCur;
    *closestID = loopKeyPre;

    return true;
  }

  /**
   * \brief // api: 外部指定回环
   *
   * \param latestID 当前帧
   * \param closestID 回环帧id
   * \return true
   * \return false
   */
  bool detectLoopClosureExternal(int* latestID, int* closestID) {
    // this function is not used yet, please ignore it
    // note: 作者表示这个功能还没有使用过，可以忽略它
    int loopKeyCur = -1;
    int loopKeyPre = -1;

    std::lock_guard<std::mutex> lock(mtxLoopInfo);
    // step: 1 外部的先验回环消息队列
    if (loopInfoVec.empty()) return false;

    // step: 2 取出回环信息，这里是把时间戳作为回环信息
    double loopTimeCur = loopInfoVec.front().data[0];
    double loopTimePre = loopInfoVec.front().data[1];
    loopInfoVec.pop_front();

    // step: 3 如果两个回环帧之间的时间差小于30s就算了
    if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
      return false;

    // step: 4 如果现存的关键帧小于2帧那也没有意义
    int cloudSize = copy_cloudKeyPoses6D->size();
    if (cloudSize < 2) return false;

    // step: 5 遍历所有的关键帧，找到离后面一个时间戳更近的关键帧的id
    loopKeyCur = cloudSize - 1;
    for (int i = cloudSize - 1; i >= 0; --i) {
      if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
        loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
      else
        break;
    }

    // step: 6 同理，找到距离前一个时间戳更近的关键帧的id
    loopKeyPre = 0;
    for (int i = 0; i < cloudSize; ++i) {
      if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
        loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
      else
        break;
    }

    // step: 7 两个是同一个关键帧，就没什么意思了吧
    if (loopKeyCur == loopKeyPre) return false;

    // step: 8 检查一下较晚的这个帧有没有被其他时候检测了回环约束
    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end()) return false;

    // step: 9 两个帧的索引输出
    *latestID = loopKeyCur;
    *closestID = loopKeyPre;

    return true;
  }

  /**
   * \brief // api: 找就近的关键帧提取点云数据
   *
   * \param nearKeyframes
   * \param key
   * \param searchNum
   */
  void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes,
                             const int& key, const int& searchNum) {
    nearKeyframes->clear();
    int cloudSize = copy_cloudKeyPoses6D->size();
    // step: 1 searchNum是搜索范围
    for (int i = -searchNum; i <= searchNum; ++i) {
      // step: 1.1 找到这个idx，如果超出范围就算了
      int keyNear = key + i;
      if (keyNear < 0 || keyNear >= cloudSize) continue;

      // step: 1.2 否则把对应角点和面点的点云转到世界坐标系下去
      *nearKeyframes +=
          *transformPointCloud(cornerCloudKeyFrames[keyNear],
                               &copy_cloudKeyPoses6D->points[keyNear]);
      *nearKeyframes += *transformPointCloud(
          surfCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
    }

    // step: 2 如果没有有效的点云就算了
    if (nearKeyframes->empty()) return;

    // step: 3 把点云下采样
    pcl::PointCloud<PointType>::Ptr cloud_temp(
        new pcl::PointCloud<PointType>());
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
  }

  /**
   * \brief // api: 将已有的回环约束可视化出来
   *
   */
  void visualizeLoopClosure() {
    // step: 1 如果没有回环约束就算了
    if (loopIndexContainer.empty()) return;

    // step: 2 先定义一下node的信息，回环约束的两帧作为node添加
    visualization_msgs::MarkerArray markerArray;
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = odometryFrame;
    markerNode.header.stamp = timeLaserInfoStamp;
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.3;
    markerNode.scale.y = 0.3;
    markerNode.scale.z = 0.3;
    markerNode.color.r = 0;
    markerNode.color.g = 0.8;
    markerNode.color.b = 1;
    markerNode.color.a = 1;

    // step: 3 定义edge信息，两帧之间约束作为edge添加
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id = odometryFrame;
    markerEdge.header.stamp = timeLaserInfoStamp;
    markerEdge.action = visualization_msgs::Marker::ADD;
    markerEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = 0.9;
    markerEdge.color.g = 0.9;
    markerEdge.color.b = 0;
    markerEdge.color.a = 1;

    // step: 4 遍历所有回环约束
    for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end();
         ++it) {
      int key_cur = it->first;
      int key_pre = it->second;
      geometry_msgs::Point p;
      p.x = copy_cloudKeyPoses6D->points[key_cur].x;
      p.y = copy_cloudKeyPoses6D->points[key_cur].y;
      p.z = copy_cloudKeyPoses6D->points[key_cur].z;
      // 添加node和edge
      markerNode.points.push_back(p);
      markerEdge.points.push_back(p);
      p.x = copy_cloudKeyPoses6D->points[key_pre].x;
      p.y = copy_cloudKeyPoses6D->points[key_pre].y;
      p.z = copy_cloudKeyPoses6D->points[key_pre].z;
      markerNode.points.push_back(p);
      markerEdge.points.push_back(p);
    }
    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);

    // step: 5 最后发布出去供可视化使用
    pubLoopConstraintEdge.publish(markerArray);
  }

  /**
   * \brief // api: 更新初值
   *
   */
  // note: 作为基于优化方式的点云匹配，初始值是非常重要
  // 好的初始值会帮助优化问题快速收敛且避免局部最优解的情况
  void updateInitialGuess() {
    // step: 1 transformTobeMapped是上一帧优化后的最佳位姿(Eigen)
    incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

    // step: 2 没有关键帧，也就是系统刚刚初始化完成
    static Eigen::Affine3f lastImuTransformation;
    if (cloudKeyPoses3D->points.empty()) {
      // step: 2.1 初始的位姿就由磁力计提供
      transformTobeMapped[0] = cloudInfo.imuRollInit;
      transformTobeMapped[1] = cloudInfo.imuPitchInit;
      transformTobeMapped[2] = cloudInfo.imuYawInit;

      // note: 无论vio还是lio，系统的不可观都是4自由度，平移+yaw角
      // step: 2.2 这里虽然有磁力计将yaw对齐，但是也可以考虑不使用yaw
      if (!useImuHeadingInitialization) transformTobeMapped[2] = 0;

      // step: 2.3 保存磁力计得到的位姿，平移置0
      lastImuTransformation =
          pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit,
                                 cloudInfo.imuPitchInit, cloudInfo.imuYawInit);

      return;
    }

    // step: 3 如果有预积分节点提供的里程记则使用
    static bool lastImuPreTransAvailable = false;
    static Eigen::Affine3f lastImuPreTransformation;
    if (cloudInfo.odomAvailable == true) {
      // step: 3.1 将提供的初值转成eigen的数据结构保存下来
      Eigen::Affine3f transBack = pcl::getTransformation(
          cloudInfo.initialGuessX, cloudInfo.initialGuessY,
          cloudInfo.initialGuessZ, cloudInfo.initialGuessRoll,
          cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);

      // step: 3.2 这个标志位表示是否收到过第一帧预积分里程记信息
      if (lastImuPreTransAvailable == false) {
        // 将当前里程记结果记录下来
        lastImuPreTransformation = transBack;
        // 收到第一个里程记数据以后这个标志位就是true
        lastImuPreTransAvailable = true;
      } else {
        // step: 3.3 计算上一个里程记的结果和当前里程记结果之间的delta pose
        Eigen::Affine3f transIncre =
            lastImuPreTransformation.inverse() * transBack;

        // step: 3.4 增量加到上一帧最佳位姿上去作为当前帧的先验估计位姿
        Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
        Eigen::Affine3f transFinal = transTobe * transIncre;
        // 将eigen变量转成欧拉角和平移的形式
        pcl::getTranslationAndEulerAngles(
            transFinal, transformTobeMapped[3], transformTobeMapped[4],
            transformTobeMapped[5], transformTobeMapped[0],
            transformTobeMapped[1], transformTobeMapped[2]);

        // step: 3.5 同理，把当前帧的值保存下来
        lastImuPreTransformation = transBack;

        // step: 3.6 虽然有里程记信息，仍然需要把imu磁力计得到的旋转记录下来
        lastImuTransformation = pcl::getTransformation(
            0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit,
            cloudInfo.imuYawInit);
        return;
      }
    }

    // step: 4 如果没有里程记信息，就是用imu的旋转信息来更新
    // 因为单纯使用imu无法得到靠谱的平移信息，因此，平移直接置0
    if (cloudInfo.imuAvailable == true) {
      // step: 4.1 初值计算方式和上面相同，只不过注意平移置0
      Eigen::Affine3f transBack =
          pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit,
                                 cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
      Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;
      Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
      Eigen::Affine3f transFinal = transTobe * transIncre;
      pcl::getTranslationAndEulerAngles(
          transFinal, transformTobeMapped[3], transformTobeMapped[4],
          transformTobeMapped[5], transformTobeMapped[0],
          transformTobeMapped[1], transformTobeMapped[2]);

      // step: 4.2 把imu磁力计得到的旋转记录下来
      lastImuTransformation =
          pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit,
                                 cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
      return;
    }
  }

  void extractForLoopClosure() {
    pcl::PointCloud<PointType>::Ptr cloudToExtract(
        new pcl::PointCloud<PointType>());
    int numPoses = cloudKeyPoses3D->size();
    for (int i = numPoses - 1; i >= 0; --i) {
      if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
        cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
      else
        break;
    }

    extractCloud(cloudToExtract);
  }

  /**
   * \brief // api: 提取当前帧相关的关键帧并且构建点云局部地图
   *
   */
  void extractNearby() {
    pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(
        new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(
        new pcl::PointCloud<PointType>());
    std::vector<int> pointSearchInd;  // 保存kdtree提取出来的元素的索引
    std::vector<float> pointSearchSqDis;  // 保存距离查询位置的距离的数组

    // step: 1 构建关键帧位置的kd-tree
    kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);

    // step: 2 根据最后一个KF的位置，提取一定距离内的关键帧
    kdtreeSurroundingKeyPoses->radiusSearch(
        cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius,
        pointSearchInd, pointSearchSqDis);

    // step: 3 根据查询的结果，把这些点的位置存进一个点云结构中
    for (int i = 0; i < (int)pointSearchInd.size(); ++i) {
      int id = pointSearchInd[i];
      surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
    }

    // step: 4 避免关键帧过多，因此做一个下采样
    downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
    downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

    // step: 5 每个下采样后的点的索引，最近邻搜索点的索引赋值给当前点的intensity
    for (auto& pt : surroundingKeyPosesDS->points) {
      kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd,
                                                pointSearchSqDis);
      pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
    }

    // step: 6 提取一些时间较近的关键帧
    int numPoses = cloudKeyPoses3D->size();
    for (int i = numPoses - 1; i >= 0; --i) {
      // 最近十秒的关键帧也保存下来
      if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
        surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
      else
        break;
    }

    // step: 7 根据筛选出来的关键帧进行局部地图构建
    extractCloud(surroundingKeyPosesDS);
  }

  /**
   * \brief // api: 根据筛选出来的关键帧进行局部地图构建
   *
   * \param cloudToExtract 待提取数据
   */
  void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract) {
    // 分别存储角点和面点相关的局部地图
    laserCloudCornerFromMap->clear();
    laserCloudSurfFromMap->clear();

    for (int i = 0; i < (int)cloudToExtract->size(); ++i) {
      // step: 1 简单校验一下关键帧距离不能太远，这个实际上不太会触发
      if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) >
          surroundingKeyframeSearchRadius)
        continue;

      // step: 2 取出提出出来的关键帧的索引
      int thisKeyInd = (int)cloudToExtract->points[i].intensity;

      // step: 3 关键帧对应的点云信息在地图容器里，直接加到局部地图中
      if (laserCloudMapContainer.find(thisKeyInd) !=
          laserCloudMapContainer.end()) {
        *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
        *laserCloudSurfFromMap += laserCloudMapContainer[thisKeyInd].second;
      } else {
        // step: 4 否则通过该帧的位姿，把点云从当前帧的位姿转到世界坐标系下
        pcl::PointCloud<PointType> laserCloudCornerTemp =
            *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],
                                 &cloudKeyPoses6D->points[thisKeyInd]);
        pcl::PointCloud<PointType> laserCloudSurfTemp =
            *transformPointCloud(surfCloudKeyFrames[thisKeyInd],
                                 &cloudKeyPoses6D->points[thisKeyInd]);
        // step: 5 点云转换之后加到局部地图中
        *laserCloudCornerFromMap += laserCloudCornerTemp;
        *laserCloudSurfFromMap += laserCloudSurfTemp;

        // step: 6 转换后的面点和角点存进这个容器中
        laserCloudMapContainer[thisKeyInd] =
            make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
      }
    }

    // step: 7 对面点和角点的局部地图做一个下采样的过程
    downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
    downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
    laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
    downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
    downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
    laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

    // step: 8 如果这个局部地图容器过大，就clear一下，避免占用内存过大
    if (laserCloudMapContainer.size() > 1000) laserCloudMapContainer.clear();
  }

  /**
   * \brief // api: 提取当前帧相关的关键帧并且构建点云局部地图
   *
   */
  void extractSurroundingKeyFrames() {
    // step: 1 如果当前没有关键帧，就return了
    if (cloudKeyPoses3D->points.empty() == true) return;

    // if (loopClosureEnableFlag == true)
    // {
    //     extractForLoopClosure();
    // } else {
    //     extractNearby();
    // }

    // step: 2 提取当前帧相关的关键帧并且构建点云局部地图
    extractNearby();
  }

  /**
   * \brief // api: 当前点云下采样
   *
   */
  void downsampleCurrentScan() {
    // Downsample cloud from current scan
    // 当前帧的角点和面点分别进行下采样，也就是为了减少计算量
    laserCloudCornerLastDS->clear();
    downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
    downSizeFilterCorner.filter(*laserCloudCornerLastDS);
    laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

    laserCloudSurfLastDS->clear();
    downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
    downSizeFilterSurf.filter(*laserCloudSurfLastDS);
    laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
  }

  /**
   * \brief // api: 求点到地图作弊系的变换Eigen
   *
   */
  void updatePointAssociateToMap() {
    // 将欧拉角转成eigen的对象
    transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
  }

  /**
   * \brief // api: 角点优化
   *
   */
  void cornerOptimization() {
    // step: 1 求点到地图作弊系的变换Eigen
    updatePointAssociateToMap();
// 使用openmp并行加速
#pragma omp parallel for num_threads(numberOfCores)
    // step: 2 遍历当前帧的角点
    for (int i = 0; i < laserCloudCornerLastDSNum; i++) {
      PointType pointOri, pointSel, coeff;
      std::vector<int> pointSearchInd;
      std::vector<float> pointSearchSqDis;

      // step: 2.1 将该点从当前帧通过初始的位姿转换到地图坐标系下去
      pointOri = laserCloudCornerLastDS->points[i];
      pointAssociateToMap(&pointOri, &pointSel);

      // step: 2.2 在角点地图里寻找距离当前点比较近的5个点
      kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd,
                                          pointSearchSqDis);

      cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
      cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
      cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
      // step: 2.3 找到的点中距离当前点最远的点不能距离太大
      if (pointSearchSqDis[4] < 1.0) {
        float cx = 0, cy = 0, cz = 0;
        // step: 2.4 计算协方差矩阵
        // 首先计算均值
        for (int j = 0; j < 5; j++) {
          cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
          cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
          cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
        }
        cx /= 5;
        cy /= 5;
        cz /= 5;
        float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
        for (int j = 0; j < 5; j++) {
          float ax =
              laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
          float ay =
              laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
          float az =
              laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

          a11 += ax * ax;
          a12 += ax * ay;
          a13 += ax * az;
          a22 += ay * ay;
          a23 += ay * az;
          a33 += az * az;
        }
        a11 /= 5;
        a12 /= 5;
        a13 /= 5;
        a22 /= 5;
        a23 /= 5;
        a33 /= 5;
        matA1.at<float>(0, 0) = a11;
        matA1.at<float>(0, 1) = a12;
        matA1.at<float>(0, 2) = a13;
        matA1.at<float>(1, 0) = a12;
        matA1.at<float>(1, 1) = a22;
        matA1.at<float>(1, 2) = a23;
        matA1.at<float>(2, 0) = a13;
        matA1.at<float>(2, 1) = a23;
        matA1.at<float>(2, 2) = a33;

        // step: 2.5 特征值分解，线特征性要求最大特征值大于3倍的次大特征值
        cv::eigen(matA1, matD1, matV1);
        if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {
          float x0 = pointSel.x;
          float y0 = pointSel.y;
          float z0 = pointSel.z;
          // note: 最大特征值对应的特征向量对应的就是直线的方向向量
          // step: 2.6 通过点的均值往两边拓展，构成一个线的两个端点
          float x1 = cx + 0.1 * matV1.at<float>(0, 0);
          float y1 = cy + 0.1 * matV1.at<float>(0, 1);
          float z1 = cz + 0.1 * matV1.at<float>(0, 2);
          float x2 = cx - 0.1 * matV1.at<float>(0, 0);
          float y2 = cy - 0.1 * matV1.at<float>(0, 1);
          float z2 = cz - 0.1 * matV1.at<float>(0, 2);

          // step: 2.7 下面是计算点到线的残差和垂线方向（及雅克比方向）
          // |oa × ob|
          float a012 =
              sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) *
                       ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
                   ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) *
                       ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
                   ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) *
                       ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));

          // |ba|
          float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) +
                           (z1 - z2) * (z1 - z2));

          // ba × (oa × ob) 为垂线方向单位向量 (la,lb,lc)
          float la =
              ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
               (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) /
              a012 / l12;

          float lb =
              -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) -
                (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) /
              a012 / l12;

          float lc =
              -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
                (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) /
              a012 / l12;

          // 点到直线的距离即残差
          float ld2 = a012 / l12;
          // 一个简单的核函数，残差越大权重降低
          float s = 1 - 0.9 * fabs(ld2);
          coeff.x = s * la;
          coeff.y = s * lb;
          coeff.z = s * lc;
          coeff.intensity = s * ld2;
          // 如果残差小于10cm，就认为是一个有效的约束
          if (s > 0.1) {
            laserCloudOriCornerVec[i] = pointOri;
            coeffSelCornerVec[i] = coeff;
            laserCloudOriCornerFlag[i] = true;
          }
        }
      }
    }
  }

  /**
   * \brief // api: 面点优化
   *
   */
  void surfOptimization() {
    // step: 1 求点到地图作弊系的变换Eigen
    updatePointAssociateToMap();

#pragma omp parallel for num_threads(numberOfCores)
    // step: 2 遍历当前帧的角点
    for (int i = 0; i < laserCloudSurfLastDSNum; i++) {
      PointType pointOri, pointSel, coeff;
      std::vector<int> pointSearchInd;
      std::vector<float> pointSearchSqDis;
      // step: 2.1 同样找5个面点
      pointOri = laserCloudSurfLastDS->points[i];
      pointAssociateToMap(&pointOri, &pointSel);
      kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd,
                                        pointSearchSqDis);

      Eigen::Matrix<float, 5, 3> matA0;
      Eigen::Matrix<float, 5, 1> matB0;
      Eigen::Vector3f matX0;
      // step: 2.2 平面方程Ax + By + Cz + 1 = 0
      matA0.setZero();
      matB0.fill(-1);
      matX0.setZero();
      // step: 2.3 找到的点中距离当前点最远的点不能距离太大
      if (pointSearchSqDis[4] < 1.0) {
        for (int j = 0; j < 5; j++) {
          matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
          matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
          matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
        }
        // step: 2.4 求解Ax = b这个超定方程，求出来x的就是这个平面的法向量
        matX0 = matA0.colPivHouseholderQr().solve(matB0);
        float pa = matX0(0, 0);
        float pb = matX0(1, 0);
        float pc = matX0(2, 0);
        float pd = 1;
        // note: 归一化，将法向量模长统一为1
        float ps = sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        // step: 2.5 每个点计算点到平面的距离，距离大于0.2m就是无效的平面
        bool planeValid = true;
        for (int j = 0; j < 5; j++) {
          if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                   pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                   pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z +
                   pd) > 0.2) {
            planeValid = false;
            break;
          }
        }

        // step: 2.6 如果通过了平面的校验计算当前点到平面的距离
        if (planeValid) {
          float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;
          // 分母不是很明白，为了更多的面点用起来？
          float s = 1 - 0.9 * fabs(pd2) /
                            sqrt(sqrt(pointSel.x * pointSel.x +
                                      pointSel.y * pointSel.y +
                                      pointSel.z * pointSel.z));

          coeff.x = s * pa;
          coeff.y = s * pb;
          coeff.z = s * pc;
          coeff.intensity = s * pd2;

          if (s > 0.1) {
            laserCloudOriSurfVec[i] = pointOri;
            coeffSelSurfVec[i] = coeff;
            laserCloudOriSurfFlag[i] = true;
          }
        }
      }
    }
  }

  /**
   * \brief // api: 将角点约束和面点约束统一到一起
   *
   */
  void combineOptimizationCoeffs() {
    // step: 1 combine corner coeffs
    for (int i = 0; i < laserCloudCornerLastDSNum; ++i) {
      // 只有标志位为true的时候才是有效约束
      if (laserCloudOriCornerFlag[i] == true) {
        laserCloudOri->push_back(laserCloudOriCornerVec[i]);
        coeffSel->push_back(coeffSelCornerVec[i]);
      }
    }
    // step: 2 combine surf coeffs
    for (int i = 0; i < laserCloudSurfLastDSNum; ++i) {
      if (laserCloudOriSurfFlag[i] == true) {
        laserCloudOri->push_back(laserCloudOriSurfVec[i]);
        coeffSel->push_back(coeffSelSurfVec[i]);
      }
    }

    // step: 3 标志位清零为了下一次迭代
    std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(),
              false);
    std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(),
              false);
  }

  /**
   * \brief // api: GN优化
   *
   * \param iterCount 迭代次数
   * \return true
   * \return false
   */
  bool LMOptimization(int iterCount) {
    // note: 原始的loam代码是将lidar坐标系转到相机坐标系
    // note: 这里把拷贝了loam中的代码，先转到相机系优化，然后转回lidar系
    // lidar <- camera      ---     camera <- lidar
    // x = z                ---     x = y
    // y = x                ---     y = z
    // z = y                ---     z = x
    // roll = yaw           ---     roll = pitch
    // pitch = roll         ---     pitch = yaw
    // yaw = pitch          ---     yaw = roll

    // lidar -> camera
    // step: 1 将lidar系转到相机系
    float srx = sin(transformTobeMapped[1]);
    float crx = cos(transformTobeMapped[1]);
    float sry = sin(transformTobeMapped[2]);
    float cry = cos(transformTobeMapped[2]);
    float srz = sin(transformTobeMapped[0]);
    float crz = cos(transformTobeMapped[0]);

    // step: 2 检查点数
    int laserCloudSelNum = laserCloudOri->size();
    if (laserCloudSelNum < 50) {
      return false;
    }

    cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

    PointType pointOri, coeff;

    // step: 3 计算雅克比
    for (int i = 0; i < laserCloudSelNum; i++) {
      // step: 3.1 首先将当前点以及点到线（面）的单位向量转到相机系
      // lidar -> camera
      pointOri.x = laserCloudOri->points[i].y;
      pointOri.y = laserCloudOri->points[i].z;
      pointOri.z = laserCloudOri->points[i].x;
      // lidar -> camera
      coeff.x = coeffSel->points[i].y;
      coeff.y = coeffSel->points[i].z;
      coeff.z = coeffSel->points[i].x;
      coeff.intensity = coeffSel->points[i].intensity;

      // step: 3.2 相机系下计算雅克比
      // note: 相机系旋转顺序是Y - X - Z对应lidar系下Z -Y -X
      float arx = (crx * sry * srz * pointOri.x + crx * crz * sry * pointOri.y -
                   srx * sry * pointOri.z) *
                      coeff.x +
                  (-srx * srz * pointOri.x - crz * srx * pointOri.y -
                   crx * pointOri.z) *
                      coeff.y +
                  (crx * cry * srz * pointOri.x + crx * cry * crz * pointOri.y -
                   cry * srx * pointOri.z) *
                      coeff.z;

      float ary = ((cry * srx * srz - crz * sry) * pointOri.x +
                   (sry * srz + cry * crz * srx) * pointOri.y +
                   crx * cry * pointOri.z) *
                      coeff.x +
                  ((-cry * crz - srx * sry * srz) * pointOri.x +
                   (cry * srz - crz * srx * sry) * pointOri.y -
                   crx * sry * pointOri.z) *
                      coeff.z;

      float arz = ((crz * srx * sry - cry * srz) * pointOri.x +
                   (-cry * crz - srx * sry * srz) * pointOri.y) *
                      coeff.x +
                  (crx * crz * pointOri.x - crx * srz * pointOri.y) * coeff.y +
                  ((sry * srz + cry * crz * srx) * pointOri.x +
                   (crz * sry - cry * srx * srz) * pointOri.y) *
                      coeff.z;

      // step: 3.3 这里雅克比把camera转到lidar
      // lidar -> camera
      matA.at<float>(i, 0) = arz;
      matA.at<float>(i, 1) = arx;
      matA.at<float>(i, 2) = ary;
      matA.at<float>(i, 3) = coeff.z;
      matA.at<float>(i, 4) = coeff.x;
      matA.at<float>(i, 5) = coeff.y;
      matB.at<float>(i, 0) = -coeff.intensity;
    }

    // step: 4 构造JTJ以及-JTe矩阵，求解增量
    cv::transpose(matA, matAt);
    matAtA = matAt * matA;
    matAtB = matAt * matB;
    cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

    // step: 5 检查一下是否有退化的情况
    if (iterCount == 0) {
      cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
      cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
      cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));
      // 对JTJ进行特征值分解
      cv::eigen(matAtA, matE, matV);
      matV.copyTo(matV2);

      isDegenerate = false;
      float eignThre[6] = {100, 100, 100, 100, 100, 100};
      for (int i = 5; i >= 0; i--) {
        // 特征值从小到大遍历，如果小于阈值就认为退化
        if (matE.at<float>(0, i) < eignThre[i]) {
          // 对应的特征向量全部置0
          for (int j = 0; j < 6; j++) {
            matV2.at<float>(i, j) = 0;
          }
          isDegenerate = true;
        } else {
          break;
        }
      }
      matP = matV.inv() * matV2;
    }

    // step: 6 如果发生退化，就对增量进行修改，退化方向不更新
    if (isDegenerate) {
      cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
      matX.copyTo(matX2);
      matX = matP * matX2;
    }

    // step: 7 增量更新
    transformTobeMapped[0] += matX.at<float>(0, 0);
    transformTobeMapped[1] += matX.at<float>(1, 0);
    transformTobeMapped[2] += matX.at<float>(2, 0);
    transformTobeMapped[3] += matX.at<float>(3, 0);
    transformTobeMapped[4] += matX.at<float>(4, 0);
    transformTobeMapped[5] += matX.at<float>(5, 0);

    // step: 8 计算更新的旋转和平移大小，旋转和平移增量足够小则收敛
    float deltaR = sqrt(pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                        pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                        pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
    float deltaT = sqrt(pow(matX.at<float>(3, 0) * 100, 2) +
                        pow(matX.at<float>(4, 0) * 100, 2) +
                        pow(matX.at<float>(5, 0) * 100, 2));
    if (deltaR < 0.05 && deltaT < 0.05) {
      return true;
    }

    // 否则继续优化
    return false;
  }

  /**
   * \brief // api: 点云配准
   *
   */
  void scan2MapOptimization() {
    // step: 1 如果没有关键帧，那也没办法做当前帧到局部地图的匹配
    if (cloudKeyPoses3D->points.empty()) return;

    // step: 2 判断当前帧的角点数和面点数是否足够
    if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum &&
        laserCloudSurfLastDSNum > surfFeatureMinValidNum) {
      // step: 3 分别把角点面点局部地图构建kdtree
      kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
      kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

      // step: 4 迭代求解
      for (int iterCount = 0; iterCount < 30; iterCount++) {
        laserCloudOri->clear();
        coeffSel->clear();

        // step: 4.1 角点优化
        cornerOptimization();

        // step: 4.2 面点优化
        surfOptimization();

        // step: 4.3 结合角点和面点
        combineOptimizationCoeffs();

        // step: 4.4 执行GN优化
        if (LMOptimization(iterCount) == true) break;
      }

      // step: 5 优化问题结束
      transformUpdate();
    } else {
      ROS_WARN(
          "Not enough features! Only %d edge and %d planar features available.",
          laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
    }
  }

  /**
   * \brief // api: 把结果和imu进行一些加权融合
   *
   */
  void transformUpdate() {
    // step: 1 可以获取九轴imu的世界系下的姿态进行加权
    if (cloudInfo.imuAvailable == true) {
      // note: roll和pitch原则上可观，lidar推算的姿态和磁力计结果做加权平均
      // note: 判断车翻了没有，车翻了好像做slam也没有什么意义了
      if (std::abs(cloudInfo.imuPitchInit) < 1.4) {
        double imuWeight = imuRPYWeight;
        tf::Quaternion imuQuaternion;
        tf::Quaternion transformQuaternion;
        double rollMid, pitchMid, yawMid;

        // step: 1.1 slerp roll
        // lidar匹配获得的roll角转成四元数
        transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
        // imu获得的roll角
        imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
        // 使用四元数球面插值
        tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight))
            .getRPY(rollMid, pitchMid, yawMid);
        // 插值结果作为roll的最终结果
        transformTobeMapped[0] = rollMid;

        // step: 1.2 slerp pitch
        transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
        imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
        tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight))
            .getRPY(rollMid, pitchMid, yawMid);
        transformTobeMapped[1] = pitchMid;
      }
    }

    // step: 2 对roll, pitch和z进行一些约束
    transformTobeMapped[0] =
        constraintTransformation(transformTobeMapped[0], rotation_tollerance);
    transformTobeMapped[1] =
        constraintTransformation(transformTobeMapped[1], rotation_tollerance);
    transformTobeMapped[5] =
        constraintTransformation(transformTobeMapped[5], z_tollerance);

    // step: 3 最终的结果也可以转成eigen的结构
    incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
  }

  /**
   * \brief // api: 约束数值
   *
   * \param value 输入
   * \param limit 限制
   * \return float 输出
   */
  float constraintTransformation(float value, float limit) {
    if (value < -limit) value = -limit;
    if (value > limit) value = limit;

    return value;
  }

  /**
   * \brief // api: 当前帧是否为关键帧
   *
   * \return true
   * \return false
   */
  bool saveFrame() {
    // step: 1 如果没有关键帧，那直接认为是关键帧
    if (cloudKeyPoses3D->points.empty()) return true;

    // step: 2 取出上一个关键帧的位姿，计算两个位姿之间的delta pose
    Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
    Eigen::Affine3f transFinal = pcl::getTransformation(
        transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
        transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
    Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

    // step: 3 任何一个旋转大于给定阈值或者平移大于给定阈值就认为是关键帧
    if (abs(roll) < surroundingkeyframeAddingAngleThreshold &&
        abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
        abs(yaw) < surroundingkeyframeAddingAngleThreshold &&
        sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold)
      return false;

    return true;
  }

  /**
   * \brief // api: 增加odom的因子
   *
   */
  void addOdomFactor() {
    // step: 1 如果是第一帧关键帧则先验约束
    if (cloudKeyPoses3D->points.empty()) {
      // step: 1.1 置信度就设置差一点，尤其是不可观的平移和yaw角
      // rad*rad, meter*meter
      noiseModel::Diagonal::shared_ptr priorNoise =
          noiseModel::Diagonal::Variances(
              (Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8).finished());

      // step: 1.2 增加先验约束，对第0个节点增加约束
      gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped),
                                        priorNoise));

      // step: 1.3 加入节点信息
      initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
    } else {
      // step: 2 如果不是第一帧就增加帧间约束
      // step: 2.1 这时帧间约束置信度就设置高一些
      noiseModel::Diagonal::shared_ptr odometryNoise =
          noiseModel::Diagonal::Variances(
              (Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());

      // step: 2.2 转成gtsam的格式
      gtsam::Pose3 poseFrom =
          pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
      gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped);

      // step: 2.3 帧间约束分别输入两个节点的id，帧间约束大小以及置信度
      gtSAMgraph.add(BetweenFactor<Pose3>(
          cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(),
          poseFrom.between(poseTo), odometryNoise));

      // step: 2.4 加入节点信息
      initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
    }
  }

  /**
   * \brief // api: 增加gps的因子
   *
   */
  void addGPSFactor() {
    // step: 1 如果没有gps信息就算了
    if (gpsQueue.empty()) return;

    // step: 2 如果有gps但是没有关键帧信息也算了
    if (cloudKeyPoses3D->points.empty())
      return;
    else {
      // step: 3 第一个关键帧和最后一个关键帧相差很近要么刚起步，要么会触发回环
      if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) <
          5.0)
        return;
    }

    // step: 4 gtsam反馈的当前x，y的置信度，如果置信度比较高也不需要gps来打扰
    if (poseCovariance(3, 3) < poseCovThreshold &&
        poseCovariance(4, 4) < poseCovThreshold)
      return;

    // step: 5 遍历gps，新增因子
    static PointType lastGPSPoint;
    while (!gpsQueue.empty()) {
      // step: 5.1 把距离当前帧比较早的帧都抛弃
      if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2) {
        gpsQueue.pop_front();
      }
      // step: 5.2 比较晚就索性再等等lidar计算
      else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2) {
        break;
      } else {
        // step: 5.3 gps时间距离当前帧已经比较近了就把这个数据取出来
        nav_msgs::Odometry thisGPS = gpsQueue.front();
        gpsQueue.pop_front();

        // step: 5.4 如果gps的置信度不高，也没有必要使用了
        float noise_x = thisGPS.pose.covariance[0];
        float noise_y = thisGPS.pose.covariance[7];
        float noise_z = thisGPS.pose.covariance[14];
        if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold) continue;

        // step: 5.5 取出gps的位置，通常gps的z没有xy准，因此这里可以不使用z值
        float gps_x = thisGPS.pose.pose.position.x;
        float gps_y = thisGPS.pose.pose.position.y;
        float gps_z = thisGPS.pose.pose.position.z;
        if (!useGpsElevation) {
          gps_z = transformTobeMapped[5];
          noise_z = 0.01;
        }

        // step:5.6 gps的x或者y太小说明还没有初始化好
        if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6) continue;

        // step: 5.7 加入gps观测不宜太频繁，相邻不超过5m
        PointType curGPSPoint;
        curGPSPoint.x = gps_x;
        curGPSPoint.y = gps_y;
        curGPSPoint.z = gps_z;
        if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
          continue;
        else
          lastGPSPoint = curGPSPoint;

        // step: 5.8 gps的置信度，标准差设置成最小1m，也就是不会特别信任gps信号
        gtsam::Vector Vector3(3);
        Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
        noiseModel::Diagonal::shared_ptr gps_noise =
            noiseModel::Diagonal::Variances(Vector3);
        // step: 5.9 调用gtsam中集成的gps的约束
        gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(),
                                    gtsam::Point3(gps_x, gps_y, gps_z),
                                    gps_noise);
        gtSAMgraph.add(gps_factor);

        // step: 5.10 加入gps之后等同于回环，需要触发较多的isam update
        aLoopIsClosed = true;
        break;
      }
    }
  }

  /**
   * \brief // api: 增加回环的因子
   *
   */
  void addLoopFactor() {
    // step: 1 回环检测线程会检测回环，检测到就会给这个队列塞入回环结果
    if (loopIndexQueue.empty()) return;

    // step: 2 把队列里所有的回环约束都添加进来
    for (int i = 0; i < (int)loopIndexQueue.size(); ++i) {
      int indexFrom = loopIndexQueue[i].first;
      int indexTo = loopIndexQueue[i].second;
      gtsam::Pose3 poseBetween = loopPoseQueue[i];
      gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
      gtSAMgraph.add(
          BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
    }

    // step: 3 清空回环相关队列
    loopIndexQueue.clear();
    loopPoseQueue.clear();
    loopNoiseQueue.clear();

    // step: 4 标志位置true
    aLoopIsClosed = true;
  }

  /**
   * \brief // api: 保存关键帧并新增因子
   *
   */
  void saveKeyFramesAndFactor() {
    // step: 1 判断是否关键帧并保存
    if (saveFrame() == false) return;

    // step: 2 如果是关键帧就给isam增加因子
    // step: 2.1 增加odom的因子
    addOdomFactor();

    // step: 2.2 gps的因子
    addGPSFactor();

    // step: 2.3 回环的因子
    addLoopFactor();

    // cout << "****************************************************" << endl;
    // gtSAMgraph.print("GTSAM Graph:\n");

    // step: 3 所有因子加完后调用isam接口更新图模型
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();

    // step: 4 如果加入了gps的约束或者回环约束，isam需要进行更多次的优化
    if (aLoopIsClosed == true) {
      isam->update();
      isam->update();
      isam->update();
      isam->update();
      isam->update();
    }

    // step: 5 将约束和节点信息清空，已经被加入到isam中，清空不会影响优化
    gtSAMgraph.resize(0);
    initialEstimate.clear();

    // step: 6 下面保存关键帧信息
    PointType thisPose3D;
    PointTypePose thisPose6D;
    Pose3 latestEstimate;
    isamCurrentEstimate = isam->calculateEstimate();
    // step: 6.1 取出优化后的最新关键帧位姿
    latestEstimate =
        isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size() - 1);
    // cout << "****************************************************" << endl;
    // isamCurrentEstimate.print("Current estimate: ");
    // step: 6.1.1 平移信息取出来保存
    // 进cloudKeyPoses3D这个结构中，其中索引作为intensity值
    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    thisPose3D.intensity = cloudKeyPoses3D->size();
    cloudKeyPoses3D->push_back(thisPose3D);

    // step: 6.1.2 6D姿态同样保留下来
    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity;
    thisPose6D.roll = latestEstimate.rotation().roll();
    thisPose6D.pitch = latestEstimate.rotation().pitch();
    thisPose6D.yaw = latestEstimate.rotation().yaw();
    thisPose6D.time = timeLaserInfoCur;
    cloudKeyPoses6D->push_back(thisPose6D);

    // cout << "****************************************************" << endl;
    // cout << "Pose covariance:" << endl;
    // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl <<
    // endl;
    // step: 6.1.3 保存当前位姿的置信度
    poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size() - 1);

    // save updated transform
    // step: 6.2 将优化后的位姿更新到transformTobeMapped数组中
    transformTobeMapped[0] = latestEstimate.rotation().roll();
    transformTobeMapped[1] = latestEstimate.rotation().pitch();
    transformTobeMapped[2] = latestEstimate.rotation().yaw();
    transformTobeMapped[3] = latestEstimate.translation().x();
    transformTobeMapped[4] = latestEstimate.translation().y();
    transformTobeMapped[5] = latestEstimate.translation().z();

    // step: 7 当前帧的点云的角点和面点分别拷贝一下保存
    pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(
        new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(
        new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*laserCloudCornerLastDS, *thisCornerKeyFrame);
    pcl::copyPointCloud(*laserCloudSurfLastDS, *thisSurfKeyFrame);
    cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
    surfCloudKeyFrames.push_back(thisSurfKeyFrame);

    // step: 8 根据当前最新位姿更新rviz可视化
    updatePath(thisPose6D);
  }

  /**
   * \brief // api: 调整全局轨迹
   *
   */
  void correctPoses() {
    // step: 1 没有关键帧，自然也没有什么意义
    if (cloudKeyPoses3D->points.empty()) return;

    // step: 2 只有回环以及gps信息这些会触发全局调整信息才会触发
    if (aLoopIsClosed == true) {
      // step: 2.1 位姿会变化，容器内转到世界坐标系下的点云就需要调整，清空
      laserCloudMapContainer.clear();

      // step: 2.2 清空path
      globalPath.poses.clear();

      // step: 2.3 然后更新所有的位姿
      int numPoses = isamCurrentEstimate.size();
      for (int i = 0; i < numPoses; ++i) {
        // step: 2.3.1 更新所有关键帧的位姿
        cloudKeyPoses3D->points[i].x =
            isamCurrentEstimate.at<Pose3>(i).translation().x();
        cloudKeyPoses3D->points[i].y =
            isamCurrentEstimate.at<Pose3>(i).translation().y();
        cloudKeyPoses3D->points[i].z =
            isamCurrentEstimate.at<Pose3>(i).translation().z();

        cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
        cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
        cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
        cloudKeyPoses6D->points[i].roll =
            isamCurrentEstimate.at<Pose3>(i).rotation().roll();
        cloudKeyPoses6D->points[i].pitch =
            isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
        cloudKeyPoses6D->points[i].yaw =
            isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

        // step: 2.3.2 同时更新path
        updatePath(cloudKeyPoses6D->points[i]);
      }

      // step: 3 标志位复位
      aLoopIsClosed = false;
    }
  }

  /**
   * \brief // api: 根据当前最新位姿更新rviz可视化
   *
   * \param pose_in 最新位姿
   */
  void updatePath(const PointTypePose& pose_in) {
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
    pose_stamped.header.frame_id = odometryFrame;
    pose_stamped.pose.position.x = pose_in.x;
    pose_stamped.pose.position.y = pose_in.y;
    pose_stamped.pose.position.z = pose_in.z;
    tf::Quaternion q =
        tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();

    globalPath.poses.push_back(pose_stamped);
  }

  /**
   * \brief // api: 将lidar里程记信息发送出去
   *
   */
  void publishOdometry() {
    // step: 1 发送当前帧的位姿 lio_sam/mapping/odometry
    nav_msgs::Odometry laserOdometryROS;
    laserOdometryROS.header.stamp = timeLaserInfoStamp;
    laserOdometryROS.header.frame_id = odometryFrame;
    laserOdometryROS.child_frame_id = "odom_mapping";
    laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
    laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
    laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
    laserOdometryROS.pose.pose.orientation =
        tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0],
                                                transformTobeMapped[1],
                                                transformTobeMapped[2]);
    pubLaserOdometryGlobal.publish(laserOdometryROS);

    // step: 2 发送lidar在odom坐标系下的tf
    static tf::TransformBroadcaster br;
    tf::Transform t_odom_to_lidar = tf::Transform(
        tf::createQuaternionFromRPY(transformTobeMapped[0],
                                    transformTobeMapped[1],
                                    transformTobeMapped[2]),
        tf::Vector3(transformTobeMapped[3], transformTobeMapped[4],
                    transformTobeMapped[5]));
    tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(
        t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "lidar_link");
    br.sendTransform(trans_odom_to_lidar);

    // step: 3 发送增量位姿变换 lio_sam/mapping/odometry_incremental
    // note: 这里主要用于给imu预积分模块使用，需要里程计是平滑的
    static bool lastIncreOdomPubFlag = false;
    static nav_msgs::Odometry laserOdomIncremental;  // incremental odometry msg
    static Eigen::Affine3f increOdomAffine;  // incremental odometry in affine
    // 该标志位处理一次后始终为true
    if (lastIncreOdomPubFlag == false) {
      lastIncreOdomPubFlag = true;
      // step: 3.1 记录当前位姿
      laserOdomIncremental = laserOdometryROS;
      increOdomAffine = trans2Affine3f(transformTobeMapped);
    } else {
      // step: 3.2 位姿增量叠加到上一帧位姿上，分解成欧拉角+平移向量
      // 上一帧的最佳位姿和当前帧最佳位姿（scanmatch之后，而不是回环或gps调整后的）之间的
      Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() *
                                    incrementalOdometryAffineBack;
      increOdomAffine = increOdomAffine * affineIncre;
      float x, y, z, roll, pitch, yaw;
      pcl::getTranslationAndEulerAngles(increOdomAffine, x, y, z, roll, pitch,
                                        yaw);

      // step: 3.3 如果有imu信号，同样对roll和pitch做插值
      if (cloudInfo.imuAvailable == true) {
        if (std::abs(cloudInfo.imuPitchInit) < 1.4) {
          double imuWeight = 0.1;
          tf::Quaternion imuQuaternion;
          tf::Quaternion transformQuaternion;
          double rollMid, pitchMid, yawMid;

          // slerp roll
          transformQuaternion.setRPY(roll, 0, 0);
          imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
          tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight))
              .getRPY(rollMid, pitchMid, yawMid);
          roll = rollMid;

          // slerp pitch
          transformQuaternion.setRPY(0, pitch, 0);
          imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
          tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight))
              .getRPY(rollMid, pitchMid, yawMid);
          pitch = pitchMid;
        }
      }
      laserOdomIncremental.header.stamp = timeLaserInfoStamp;
      laserOdomIncremental.header.frame_id = odometryFrame;
      laserOdomIncremental.child_frame_id = "odom_mapping";
      laserOdomIncremental.pose.pose.position.x = x;
      laserOdomIncremental.pose.pose.position.y = y;
      laserOdomIncremental.pose.pose.position.z = z;
      laserOdomIncremental.pose.pose.orientation =
          tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);

      // step: 3.4 协方差这一位作为是否退化的标志位
      if (isDegenerate)
        laserOdomIncremental.pose.covariance[0] = 1;
      else
        laserOdomIncremental.pose.covariance[0] = 0;
    }
    pubLaserOdometryIncremental.publish(laserOdomIncremental);
  }

  /**
   * \brief // api: 发送可视化点云信息
   *
   */
  void publishFrames() {
    // step: 1 没有关键帧就退
    if (cloudKeyPoses3D->points.empty()) return;

    // step: 2 发送关键帧
    publishCloud(&pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp,
                 odometryFrame);

    // step: 3 发送周围局部地图点云信息
    publishCloud(&pubRecentKeyFrames, laserCloudSurfFromMapDS,
                 timeLaserInfoStamp, odometryFrame);

    // step: 4 发布校准后的当前关键帧点云
    if (pubRecentKeyFrame.getNumSubscribers() != 0) {
      pcl::PointCloud<PointType>::Ptr cloudOut(
          new pcl::PointCloud<PointType>());
      PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
      // 把当前点云转换到世界坐标系下去
      *cloudOut += *transformPointCloud(laserCloudCornerLastDS, &thisPose6D);
      *cloudOut += *transformPointCloud(laserCloudSurfLastDS, &thisPose6D);
      // 发送当前点云
      publishCloud(&pubRecentKeyFrame, cloudOut, timeLaserInfoStamp,
                   odometryFrame);
    }

    // step: 5 发送配准后原始点云
    if (pubCloudRegisteredRaw.getNumSubscribers() != 0) {
      pcl::PointCloud<PointType>::Ptr cloudOut(
          new pcl::PointCloud<PointType>());
      pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
      PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
      *cloudOut = *transformPointCloud(cloudOut, &thisPose6D);
      publishCloud(&pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp,
                   odometryFrame);
    }

    // step: 6 发送path
    if (pubPath.getNumSubscribers() != 0) {
      globalPath.header.stamp = timeLaserInfoStamp;
      globalPath.header.frame_id = odometryFrame;
      pubPath.publish(globalPath);
    }
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "lio_sam");

  mapOptimization MO;

  ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");

  std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
  std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread,
                                 &MO);

  ros::spin();

  loopthread.join();
  visualizeMapThread.join();

  return 0;
}
