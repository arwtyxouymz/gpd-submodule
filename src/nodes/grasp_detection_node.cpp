#include "../../../gpd/include/nodes/grasp_detection_node.h"


/** constants for input point cloud types */
const int GraspDetectionNode::POINT_CLOUD_2 = 0; ///< sensor_msgs/PointCloud2
const int GraspDetectionNode::CLOUD_INDEXED = 1; ///< cloud with indices
const int GraspDetectionNode::CLOUD_SAMPLES = 2; ///< cloud with (x,y,z) samples


GraspDetectionNode::GraspDetectionNode(ros::NodeHandle& node) : has_cloud_(false), has_normals_(false),
  size_left_cloud_(0), has_samples_(true), frame_("")
{
  cloud_camera_ = NULL;

  nh_ = node; // Assign the NodeHandle to the private variable

  // set camera viewpoint to default origin
  std::vector<double> camera_position;
  nh_.getParam("camera_position", camera_position);
  view_point_ << camera_position[0], camera_position[1], camera_position[2];

  // choose sampling method for grasp detection
  nh_.param("use_importance_sampling", use_importance_sampling_, false);

  if (use_importance_sampling_)
  {
    importance_sampling_ = new SequentialImportanceSampling(nh_);
  }
  grasp_detector_ = new GraspDetector(nh_);

  // Read input cloud and sample ROS topics parameters.
  int cloud_type;
  nh_.param("cloud_type", cloud_type, POINT_CLOUD_2);
  std::string cloud_topic;
  nh_.param("cloud_topic", cloud_topic, std::string("/camera/depth_registered/points"));
  std::string samples_topic;
  nh_.param("samples_topic", samples_topic, std::string(""));
  std::string rviz_topic;
  nh_.param("rviz_topic", rviz_topic, std::string(""));

  if (!rviz_topic.empty())
  {
    grasps_rviz_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(rviz_topic, 1);
    use_rviz_ = true;
  }
  else
  {
    use_rviz_ = false;
  }

  // subscribe to input point cloud ROS topic
  if (cloud_type == POINT_CLOUD_2)
    cloud_sub_ = nh_.subscribe(cloud_topic, 1, &GraspDetectionNode::cloud_callback, this);
  else if (cloud_type == CLOUD_INDEXED)
    cloud_sub_ = nh_.subscribe(cloud_topic, 1, &GraspDetectionNode::cloud_indexed_callback, this);
  else if (cloud_type == CLOUD_SAMPLES)
  {
    cloud_sub_ = nh_.subscribe(cloud_topic, 1, &GraspDetectionNode::cloud_samples_callback, this);
    //    grasp_detector_->setUseIncomingSamples(true);
    has_samples_ = false;
  }

  // subscribe to input samples ROS topic
  if (!samples_topic.empty())
  {
    samples_sub_ = nh_.subscribe(samples_topic, 1, &GraspDetectionNode::samples_callback, this);
    has_samples_ = false;
  }

  // uses ROS topics to publish grasp candidates, antipodal grasps, and grasps after clustering
  grasps_pub_ = nh_.advertise<gpd::GraspSetList>("clustered_grasps", 10);
  index_sub_ = nh_.subscribe("/marker_index", 1, &GraspDetectionNode::marker_index_callback, this);

  // Advertise the SetParameters service
  srv_set_params_ = nh_.advertiseService("/gpd/set_params", &GraspDetectionNode::set_params_callback, this);

  nh_.getParam("workspace", workspace_);
}

void GraspDetectionNode::marker_index_callback(const std_msgs::Int8& msg)
{
    if (!use_rviz_) return;
    ROS_INFO("Generating Index Marker!");
    std::vector<Grasp> grasp;
    grasp.push_back(grasp_for_markers[static_cast<int>(msg.data)]);
    const HandSearch::Parameters& params = grasp_detector_->getHandSearchParameters();
    grasps_rviz_pub_.publish(convertToVisualGraspMsg(grasp, params.hand_outer_diameter_, params.hand_depth_,
                                                     params.finger_width_, params.hand_height_, frame_, 1.0, 1, 0.0, 0.0));
}

void GraspDetectionNode::run()
{
  ros::Rate rate(100);
  ROS_INFO("Waiting for point cloud to arrive ...");

  while (ros::ok())
  {
    if (has_cloud_)
    {
      // detect grasps in point cloud
      std::vector<Grasp> grasps = detectGraspPosesInTopic();
      grasp_for_markers = grasps;

      // visualize grasps in rviz
      if (use_rviz_)
      {
        const HandSearch::Parameters& params = grasp_detector_->getHandSearchParameters();
        grasps_rviz_pub_.publish(convertToVisualGraspMsg(grasps, params.hand_outer_diameter_, params.hand_depth_,
                                                         params.finger_width_, params.hand_height_, frame_));
      }

      // reset the system
      has_cloud_ = false;
      has_samples_ = false;
      has_normals_ = false;
      ROS_INFO("Waiting for point cloud to arrive ...");
    }

    ros::spinOnce();
    rate.sleep();
  }
}

bool GraspDetectionNode::set_params_callback(gpd::SetParameters::Request &req, gpd::SetParameters::Response &resp)
{
  // Delete the existing sampler and detectors
  if (use_importance_sampling_)
  {
    delete importance_sampling_;
  }
  delete grasp_detector_;

  // Set the workspace from the request
  if (req.set_workspace)
  {
    workspace_.clear();
    for (int i = 0; i < req.workspace.size(); i++){

      workspace_.push_back(req.workspace[i]);
    }
    nh_.setParam("workspace", workspace_);
  }

  // Set the workspace_grasps from the request
  if (req.set_workspace_grasps)
  {
    nh_.setParam("filter_grasps", true);
    std::vector<double> workspace_grasps;
    for (int i = 0; i < req.workspace_grasps.size(); i++)
    {
      workspace_grasps.push_back(req.workspace_grasps[i]);
    }
    nh_.setParam("workspace_grasps", workspace_grasps);
  }
  else
  {
    nh_.setParam("filter_grasps", false);
  }

  if (req.set_camera_position)
  {
    view_point_ << req.camera_position[0], req.camera_position[1], req.camera_position[2];
    std::vector<double> camera_position;
    camera_position.push_back(view_point_.x());
    camera_position.push_back(view_point_.y());
    camera_position.push_back(view_point_.z());
    nh_.setParam("camera_position", camera_position);
  }

  // Creating new sampler and detector so they load the new rosparams
  if (use_importance_sampling_)
  {
    importance_sampling_ = new SequentialImportanceSampling(nh_);
  }
  grasp_detector_ = new GraspDetector(nh_);

  resp.success = true;
  return true;
}

std::vector<Grasp> GraspDetectionNode::detectGraspPosesInTopic()
{
  // detect grasp poses
  std::vector<Grasp> grasps;

  if (use_importance_sampling_)
  {
    cloud_camera_->filterWorkspace(workspace_);
    cloud_camera_->voxelizeCloud(0.003);
    cloud_camera_->calculateNormals(4);
    grasps = importance_sampling_->detectGrasps(*cloud_camera_);
  }
  else
  {
    // preprocess the point cloud
    grasp_detector_->preprocessPointCloud(*cloud_camera_);

    // detect grasps in the point cloud
    grasps = grasp_detector_->detectGrasps(*cloud_camera_);
  }

  // Publish the selected grasps.
  // gpd::GraspConfigList selected_grasps_msg = createGraspListMsg(grasps);
  gpd::GraspSetList custom_grasps_msg = createGraspSetListMsg(grasps);
  // grasps_pub_.publish(selected_grasps_msg);
  grasps_pub_.publish(custom_grasps_msg);

  // ROS_INFO_STREAM("Published " << selected_grasps_msg.grasps.size() << " highest-scoring grasps.");
  ROS_INFO_STREAM("Published " << custom_grasps_msg.grasps.size() << " highest-scoring grasps.");

  return grasps;
}


std::vector<int> GraspDetectionNode::getSamplesInBall(const PointCloudRGBA::Ptr& cloud,
  const pcl::PointXYZRGBA& centroid, float radius)
{
  std::vector<int> indices;
  std::vector<float> dists;
  pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
  kdtree.setInputCloud(cloud);
  kdtree.radiusSearch(centroid, radius, indices, dists);
  return indices;
}


void GraspDetectionNode::cloud_callback(const sensor_msgs::PointCloud2& msg)
{
  if (!has_cloud_)
  {
    delete cloud_camera_;
    cloud_camera_ = NULL;

    Eigen::Matrix3Xd view_points(3,1);
    view_points.col(0) = view_point_;

    if (msg.fields.size() == 6 && msg.fields[3].name == "normal_x" && msg.fields[4].name == "normal_y"
      && msg.fields[5].name == "normal_z")
    {
      PointCloudPointNormal::Ptr cloud(new PointCloudPointNormal);
      pcl::fromROSMsg(msg, *cloud);
      cloud_camera_ = new CloudCamera(cloud, 0, view_points);
      cloud_camera_header_ = msg.header;
      ROS_INFO_STREAM("Received cloud with " << cloud_camera_->getCloudProcessed()->size() << " points and normals.");
    }
    else
    {
      PointCloudRGBA::Ptr cloud(new PointCloudRGBA);
      pcl::fromROSMsg(msg, *cloud);
      cloud_camera_ = new CloudCamera(cloud, 0, view_points);
      cloud_camera_header_ = msg.header;
      ROS_INFO_STREAM("Received cloud with " << cloud_camera_->getCloudProcessed()->size() << " points.");
    }

    has_cloud_ = true;
    frame_ = msg.header.frame_id;
  }
}


void GraspDetectionNode::cloud_indexed_callback(const gpd::CloudIndexed& msg)
{
  if (!has_cloud_)
  {
    initCloudCamera(msg.cloud_sources);

    // Set the indices at which to sample grasp candidates.
    std::vector<int> indices(msg.indices.size());
    for (int i=0; i < indices.size(); i++)
    {
      indices[i] = msg.indices[i].data;
    }
    cloud_camera_->setSampleIndices(indices);

    has_cloud_ = true;
    frame_ = msg.cloud_sources.cloud.header.frame_id;

    ROS_INFO_STREAM("Received cloud with " << cloud_camera_->getCloudProcessed()->size() << " points, and "
      << msg.indices.size() << " samples");
  }
}


void GraspDetectionNode::cloud_samples_callback(const gpd::CloudSamples& msg)
{
  if (!has_cloud_)
  {
    initCloudCamera(msg.cloud_sources);

    // Set the samples at which to sample grasp candidates.
    Eigen::Matrix3Xd samples(3, msg.samples.size());
    for (int i=0; i < msg.samples.size(); i++)
    {
      samples.col(i) << msg.samples[i].x, msg.samples[i].y, msg.samples[i].z;
    }
    cloud_camera_->setSamples(samples);

    has_cloud_ = true;
    has_samples_ = true;
    frame_ = msg.cloud_sources.cloud.header.frame_id;

    ROS_INFO_STREAM("Received cloud with " << cloud_camera_->getCloudProcessed()->size() << " points, and "
      << cloud_camera_->getSamples().cols() << " samples");
  }
}


void GraspDetectionNode::samples_callback(const gpd::SamplesMsg& msg)
{
  if (!has_samples_)
  {
    Eigen::Matrix3Xd samples(3, msg.samples.size());

    for (int i=0; i < msg.samples.size(); i++)
    {
      samples.col(i) << msg.samples[i].x, msg.samples[i].y, msg.samples[i].z;
    }

    cloud_camera_->setSamples(samples);
    has_samples_ = true;

    ROS_INFO_STREAM("Received grasp samples message with " << msg.samples.size() << " samples");
  }
}


void GraspDetectionNode::initCloudCamera(const gpd::CloudSources& msg)
{
  // clean up
  delete cloud_camera_;
  cloud_camera_ = NULL;

  // Set view points.
  Eigen::Matrix3Xd view_points(3, msg.view_points.size());
  for (int i = 0; i < msg.view_points.size(); i++)
  {
    view_points.col(i) << msg.view_points[i].x, msg.view_points[i].y, msg.view_points[i].z;
  }

  // Set point cloud.
  if (msg.cloud.fields.size() == 6 && msg.cloud.fields[3].name == "normal_x"
    && msg.cloud.fields[4].name == "normal_y" && msg.cloud.fields[5].name == "normal_z")
  {
    PointCloudPointNormal::Ptr cloud(new PointCloudPointNormal);
    pcl::fromROSMsg(msg.cloud, *cloud);

    // TODO: multiple cameras can see the same point
    Eigen::MatrixXi camera_source = Eigen::MatrixXi::Zero(view_points.cols(), cloud->size());
    for (int i = 0; i < msg.camera_source.size(); i++)
    {
      camera_source(msg.camera_source[i].data, i) = 1;
    }

    cloud_camera_ = new CloudCamera(cloud, camera_source, view_points);
  }
  else
  {
    PointCloudRGBA::Ptr cloud(new PointCloudRGBA);
    pcl::fromROSMsg(msg.cloud, *cloud);

    // TODO: multiple cameras can see the same point
    Eigen::MatrixXi camera_source = Eigen::MatrixXi::Zero(view_points.cols(), cloud->size());
    for (int i = 0; i < msg.camera_source.size(); i++)
    {
      camera_source(msg.camera_source[i].data, i) = 1;
    }

    cloud_camera_ = new CloudCamera(cloud, camera_source, view_points);
    std::cout << "view_points:\n" << view_points << "\n";
  }
}


gpd::GraspConfigList GraspDetectionNode::createGraspListMsg(const std::vector<Grasp>& hands)
{
  gpd::GraspConfigList msg;

  for (int i = 0; i < hands.size(); i++)
    msg.grasps.push_back(convertToGraspMsg(hands[i]));

  msg.header = cloud_camera_header_;

  return msg;
}

gpd::GraspSetList GraspDetectionNode::createGraspSetListMsg(const std::vector<Grasp>& hands)
{
    gpd::GraspSetList msg;
    std::vector<gpd::GraspSet> grasps;
    for (int i = 0; i < hands.size(); i++)
        grasps.push_back(convertToGraspSetMsg(hands[i]));

    std::sort(grasps.begin(), 
              grasps.end(), 
              [](const gpd::GraspSet& a, const gpd::GraspSet& b) {
                return a.score.data > b.score.data;
              });

    for (int j = 0; j < grasps.size(); j++)
        msg.grasps.push_back(grasps[j]);
    msg.header = cloud_camera_header_;

    return msg;
}

gpd::GraspConfig GraspDetectionNode::convertToGraspMsg(const Grasp& hand)
{
  gpd::GraspConfig msg;
  tf::pointEigenToMsg(hand.getGraspBottom(), msg.bottom);
  tf::pointEigenToMsg(hand.getGraspTop(), msg.top);
  tf::pointEigenToMsg(hand.getGraspSurface(), msg.surface);
  tf::vectorEigenToMsg(hand.getApproach(), msg.approach);
  tf::vectorEigenToMsg(hand.getBinormal(), msg.binormal);
  tf::vectorEigenToMsg(hand.getAxis(), msg.axis);
  msg.width.data = hand.getGraspWidth();
  msg.score.data = hand.getScore();
  tf::pointEigenToMsg(hand.getSample(), msg.sample);

  return msg;
}

gpd::GraspSet GraspDetectionNode::convertToGraspSetMsg(const Grasp& hand)
{
    gpd::GraspSet msg;
    tf::vectorEigenToMsg(hand.getApproach(), msg.approach);
    msg.pose = convert_to_ros_msg(hand);
    msg.score.data = hand.getScore();
    msg.grasp = convertToGraspMsg(hand);
    return msg;
}

visualization_msgs::MarkerArray GraspDetectionNode::convertToVisualGraspMsg(const std::vector<Grasp>& hands,
  double outer_diameter, double hand_depth, double finger_width, double hand_height, const std::string& frame_id,
  float a, float r, float g, float b)
{
  double width = outer_diameter;
  double hw = 0.5 * width;

  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker left_finger, right_finger, base, approach;
  Eigen::Vector3d left_bottom, right_bottom, left_top, right_top, left_center, right_center, approach_center,
    base_center;
  std::vector<Grasp> ordered_grasps;
  for (int i = 0; i < hands.size(); i++)
      ordered_grasps.push_back(hands[i]);
  std::sort(ordered_grasps.begin(),
          ordered_grasps.end(),
          [](const Grasp& a, const Grasp& b) {
            return a.getScore() > b.getScore();
          });

  for (int i = 0; i < ordered_grasps.size(); i++)
  {
    left_bottom = ordered_grasps[i].getGraspBottom() - (hw - 0.5*finger_width) * ordered_grasps[i].getBinormal();
    right_bottom = ordered_grasps[i].getGraspBottom() + (hw - 0.5*finger_width) * ordered_grasps[i].getBinormal();
    left_top = left_bottom + hand_depth * ordered_grasps[i].getApproach();
    right_top = right_bottom + hand_depth * ordered_grasps[i].getApproach();
    left_center = left_bottom + 0.5*(left_top - left_bottom);
    right_center = right_bottom + 0.5*(right_top - right_bottom);
    base_center = left_bottom + 0.5*(right_bottom - left_bottom) - 0.01*ordered_grasps[i].getApproach();
    approach_center = base_center - 0.04*ordered_grasps[i].getApproach();
  
    base = createHandBaseMarker(left_bottom, right_bottom, ordered_grasps[i].getFrame(), 0.02, hand_height, i, frame_id, a, r, g, b);
    left_finger = createFingerMarker(left_center, ordered_grasps[i].getFrame(), hand_depth, finger_width, hand_height, i*3, frame_id, a, r, g, b);
    right_finger = createFingerMarker(right_center, ordered_grasps[i].getFrame(), hand_depth, finger_width, hand_height, i*3+1, frame_id, a, r, g, b);
    approach = createFingerMarker(approach_center, ordered_grasps[i].getFrame(), 0.08, finger_width, hand_height, i*3+2, frame_id, a, r, g, b);
  
    marker_array.markers.push_back(left_finger);
    marker_array.markers.push_back(right_finger);
    marker_array.markers.push_back(approach);
    marker_array.markers.push_back(base);
  }

  return marker_array;
}


visualization_msgs::Marker GraspDetectionNode::createFingerMarker(const Eigen::Vector3d& center,
        const Eigen::Matrix3d& frame, double length, double width, double height, int id, const std::string& frame_id,
        float a, float r, float g, float b)
{
    visualization_msgs::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = ros::Time();
    marker.ns = "finger";
    marker.id = id;
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = center(0);
    marker.pose.position.y = center(1);
    marker.pose.position.z = center(2);
    marker.lifetime = ros::Duration(30);

    // use orientation of hand frame
    Eigen::Quaterniond quat(frame);
    marker.pose.orientation.x = quat.x();
    marker.pose.orientation.y = quat.y();
    marker.pose.orientation.z = quat.z();
    marker.pose.orientation.w = quat.w();

    // these scales are relative to the hand frame (unit: meters)
    marker.scale.x = length; // forward direction
    marker.scale.y = width; // hand closing direction
    marker.scale.z = height; // hand vertical direction

    marker.color.a = a;
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;

    return marker;
}


visualization_msgs::Marker GraspDetectionNode::createHandBaseMarker(const Eigen::Vector3d& start,
        const Eigen::Vector3d& end, const Eigen::Matrix3d& frame, double length, double height, int id,
        const std::string& frame_id,
        float a, float r, float g, float b)
{
    Eigen::Vector3d center = start + 0.5 * (end - start);

    visualization_msgs::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = ros::Time();
    marker.ns = "hand_base";
    marker.id = id;
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = center(0);
    marker.pose.position.y = center(1);
    marker.pose.position.z = center(2);
    marker.lifetime = ros::Duration(30);

    // use orientation of hand frame
    Eigen::Quaterniond quat(frame);
    marker.pose.orientation.x = quat.x();
    marker.pose.orientation.y = quat.y();
    marker.pose.orientation.z = quat.z();
    marker.pose.orientation.w = quat.w();

    // these scales are relative to the hand frame (unit: meters)
    marker.scale.x = length; // forward direction
    marker.scale.y = (end - start).norm(); // hand closing direction
    marker.scale.z = height; // hand vertical direction

    marker.color.a = a;
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;

    return marker;
}

geometry_msgs::PoseStamped GraspDetectionNode::convert_to_ros_msg(const Grasp &grasp) {
    const HandSearch::Parameters &params =
        grasp_detector_->getHandSearchParameters();
    double outer_diameter, hand_depth, finger_width, hand_height;
    outer_diameter = params.hand_outer_diameter_;
    double width = outer_diameter;
    double hw = 0.5 * width;

    hand_depth = params.hand_depth_;
    finger_width = params.finger_width_;
    hand_height = params.hand_height_;

    Eigen::Vector3d left_bottom, right_bottom, left_top, right_top, left_center,
        right_center, approach_center, base_center;

    left_bottom =
        grasp.getGraspBottom() - (hw - 0.5 * finger_width) * grasp.getBinormal();
    right_bottom =
        grasp.getGraspBottom() + (hw - 0.5 * finger_width) * grasp.getBinormal();
    left_top = left_bottom + hand_depth * grasp.getApproach();
    right_top = right_bottom + hand_depth * grasp.getApproach();
    left_center = left_bottom + 0.5 * (left_top - left_bottom);
    right_center = right_bottom + 0.5 * (right_top - right_bottom);
    base_center = left_bottom + 0.5 * (right_bottom - left_bottom) - 0.02 * grasp.getApproach();
    Eigen::Quaterniond quat(grasp.getFrame());

    geometry_msgs::PoseStamped pre_pose;
    pre_pose.pose.position.x = (base_center)(0);
    pre_pose.pose.position.y = (base_center)(1);
    pre_pose.pose.position.z = (base_center)(2);
    pre_pose.pose.orientation.x = quat.x();
    pre_pose.pose.orientation.y = quat.y();
    pre_pose.pose.orientation.z = quat.z();
    pre_pose.pose.orientation.w = quat.w();
    pre_pose.header.stamp = ros::Time();
    pre_pose.header.frame_id = frame_;
    return pre_pose;
}

int main(int argc, char** argv)
{
    // seed the random number generator
    std::srand(std::time(0));

    // initialize ROS
    ros::init(argc, argv, "detect_grasps");
    ros::NodeHandle node("~");

    GraspDetectionNode grasp_detection(node);
    grasp_detection.run();

    return 0;
}
