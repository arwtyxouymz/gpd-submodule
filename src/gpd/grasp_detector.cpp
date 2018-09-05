#include "../../include/gpd/grasp_detector.h"
//#include "Static_Image_Publisher/ArrayImages.h"

using namespace std;
GraspDetector::GraspDetector(ros::NodeHandle &node) {
  Eigen::initParallel();
  // Create objects to store parameters.
  CandidatesGenerator::Parameters generator_params;
  HandSearch::Parameters hand_search_params;
  // std::vector<cv::Mat> valid_images;

  // Read hand geometry parameters.
  node.param("/detect_grasps/finger_width", hand_search_params.finger_width_,
             0.01);
  node.param("/detect_grasps/is_benchmark", is_benchmark, false);
  /*if (is_benchmark == false) {
    std::cout << "TEST mode" << std::endl;
  } else if (is_benchmark == true) {
    std::cout << "BENCHMARK mode" << std::endl;
  }*/
  // std::cout << "is_benchmark " << is_benchmark << std::endl;
  node.param("/detect_grasps/hand_outer_diameter",
             hand_search_params.hand_outer_diameter_, 0.09);
  node.param("/detect_grasps/hand_depth", hand_search_params.hand_depth_, 0.06);
  node.param("/detect_grasps/hand_height", hand_search_params.hand_height_,
             0.02);
  node.param("/detect_grasps/init_bite", hand_search_params.init_bite_, 0.015);
  outer_diameter_ = hand_search_params.hand_outer_diameter_;

  // Read local hand search parameters.
  node.param("/detect_grasps/nn_radius", hand_search_params.nn_radius_frames_,
             0.01);
  node.param("/detect_grasps/num_orientations",
             hand_search_params.num_orientations_, 8);
  node.param("/detect_grasps/num_samples", hand_search_params.num_samples_,
             500);
  node.param("/detect_grasps/num_threads", hand_search_params.num_threads_, 1);
  node.param("/detect_grasps/rotation_axis", hand_search_params.rotation_axis_,
             2); // cannot be changed

  // Read plotting parameters.
  node.param("/detect_grasps/plot_samples", plot_samples_, false);
  node.param("/detect_grasps/plot_normals", plot_normals_, false);
  generator_params.plot_normals_ = plot_normals_;
  node.param("/detect_grasps/plot_filtered_grasps", plot_filtered_grasps_,
             false);
  node.param("/detect_grasps/plot_valid_grasps", plot_valid_grasps_, false);
  node.param("/detect_grasps/plot_clusters", plot_clusters_, false);
  node.param("/detect_grasps/plot_selected_grasps", plot_selected_grasps_,
             false);

  // Read general parameters.
  generator_params.num_samples_ = hand_search_params.num_samples_;
  generator_params.num_threads_ = hand_search_params.num_threads_;
  node.param("/detect_grasps/plot_candidates", generator_params.plot_grasps_,
             false);

  // Read preprocessing parameters.
  node.param("/detect_grasps/remove_outliers",
             generator_params.remove_statistical_outliers_, true);
  node.param("/detect_grasps/voxelize", generator_params.voxelize_, true);
  node.getParam("/detect_grasps/workspace", generator_params.workspace_);
  node.getParam("/detect_grasps/workspace_grasps", workspace_);

  // Create object to generate grasp candidates.
  candidates_generator_ =
      new CandidatesGenerator(generator_params, hand_search_params);

  // Read classification parameters and create classifier.
  std::string model_file, weights_file;
  bool use_gpu;
  node.param("/detect_grasps/model_file", model_file, std::string(""));
  node.param("/detect_grasps/trained_file", weights_file, std::string(""));
  node.param("/detect_grasps/min_score_diff", min_score_diff_, 500.0);
  node.param("/detect_grasps/create_image_batches", create_image_batches_,
             true);
  node.param("/detect_grasps/use_gpu", use_gpu, true);
  classifier_ = new CaffeClassifier(model_file, weights_file, use_gpu);

  // Read grasp image parameters.
  node.param("/detect_grasps/image_outer_diameter",
             image_params_.outer_diameter_,
             hand_search_params.hand_outer_diameter_);
  node.param("/detect_grasps/image_depth", image_params_.depth_,
             hand_search_params.hand_depth_);
  node.param("/detect_grasps/image_height", image_params_.height_,
             hand_search_params.hand_height_);
  node.param("/detect_grasps/image_size", image_params_.size_, 60);
  node.param("/detect_grasps/image_num_channels", image_params_.num_channels_,
             15);

  // Read learning parameters.
  bool remove_plane;
  node.param("/detect_grasps/remove_plane_before_image_calculation",
             remove_plane, false);

  // Create object to create grasp images from grasp candidates (used for
  // classification)
  learning_ =
      new Learning(image_params_, hand_search_params.num_threads_,
                   hand_search_params.num_orientations_, false, remove_plane);

  // Read grasp filtering parameters
  node.param("/detect_grasps/filter_grasps", filter_grasps_, false);
  node.param("/detect_grasps/filter_half_antipodal", filter_half_antipodal_,
             false);
  std::vector<double> gripper_width_range(2);
  gripper_width_range[0] = 0.03;
  gripper_width_range[1] = 0.07;
  node.getParam("/detect_grasps/gripper_width_range", gripper_width_range);
  min_aperture_ = gripper_width_range[0];
  max_aperture_ = gripper_width_range[1];

  // Read clustering parameters
  int min_inliers;
  node.param("/detect_grasps/min_inliers", min_inliers, 0);
  clustering_ = new Clustering(min_inliers);
  // cluster_grasps_ = min_inliers > 0 ? true : false;
  cluster_grasps_ = true;
  // Read grasp selection parameters
  node.param("/detect_grasps/num_selected", num_selected_, 100);
  cleaned_pointcloud_pub =
      node.advertise<sensor_msgs::PointCloud2>("cleaned_point_cloud", 1);
}

std::vector<Grasp> GraspDetector::detectGrasps(const CloudCamera &cloud_cam) {
  std::vector<Grasp> selected_grasps(0);
  // std::vector<Grasp> nan_grasps;
  // Grasp nan_grasp;

  // Check if the point cloud is empty.
  if (cloud_cam.getCloudOriginal()->size() == 0) {
    ROS_INFO("Point cloud is empty!");
    return selected_grasps;
    // return -1;
  }

  // 1. Generate grasp candidates.
  std::vector<GraspSet> candidates = generateGraspCandidates(cloud_cam);
  ROS_INFO_STREAM("Generated " << candidates.size()
                               << " grasp candidate sets.");
  if (candidates.size() == 0) {
    return selected_grasps;
  }

  // 2.1 Prune grasp candidates based on min. and max. robot hand aperture and
  // fingers below table surface.
  if (filter_grasps_) {
    candidates = filterGraspsWorkspace(candidates, workspace_);
  }
  if (candidates.size() == 0) {
    return selected_grasps;
  }

  // 2.2 Filter half grasps.
  if (filter_half_antipodal_) {
    candidates = filterHalfAntipodal(candidates);
  }
  if (candidates.size() == 0) {
    return selected_grasps;
  }

  // 3. Classify each grasp candidate. (Note: switch from a list of hypothesis
  // sets to a list of grasp hypotheses)
  valid_grasps = classifyGraspCandidates(cloud_cam, candidates);
  ROS_INFO_STREAM("Predicted " << valid_grasps.size() << " valid grasps.");

  if (valid_grasps.size() <= 2) {
    std::cout << "Not enough valid grasps predicted! Using all grasps from "
                 "previous step.\n";
    // return valid_grasps
    valid_grasps = extractHypotheses(candidates);
  }
  if (valid_grasps.size() <= 2) {
    std::cout
        << "Not enough valid grasps extracted! Using all grasps extracted\n";
    return valid_grasps;
  } else if (valid_grasps.size() == 0) {
    return selected_grasps;
  }
  // 4. Cluster the grasps.
  std::vector<Grasp> clustered_grasps;
  std::vector<cv::Mat> clustered_images;
  if (cluster_grasps_) {
    // clustered_grasps = findClusters(valid_grasps, screened_images);
    bool remove_inliers = false;
    std::tie(clustered_grasps, clustered_images) =
        findClusters(valid_grasps, screened_images, remove_inliers);

    ROS_INFO_STREAM("Found " << clustered_grasps.size() << " clusters.");
    if (clustered_grasps.size() <= 1) {
      std::cout << "Not enough clusters found! Using all grasps from previous "
                   "step.\n";
      clustered_grasps = valid_grasps;
      clustered_images = screened_images;
    }
  } else {
    clustered_grasps = valid_grasps;
    clustered_images = screened_images;
  }
  std::vector<std::pair<decltype(clustered_grasps)::reference,
                        decltype(clustered_images)::reference>>
      tmpvector;
  for (int i = 0; i < clustered_grasps.size(); i++) {
    auto &a = clustered_grasps.at(i);
    if (std::isnan(a.getScore())) {
      continue;
    }
    auto &b = clustered_images.at(i);
    tmpvector.push_back(std::make_pair(std::ref(a), std::ref(b)));
  }

  // 5. Select highest-scoring grasps.
  std::vector<cv::Mat> selected_images;
  std::cout << "Sorting the grasps based on their score ... \n";
  std::sort(tmpvector.begin(), tmpvector.end(),
            [](decltype(tmpvector)::const_reference p1,
               decltype(tmpvector)::const_reference p2) {
              return p1.first.getScore() > p2.first.getScore();
            });
  // push back the top-k highest grasps 
  for (int i = 0; i < std::min((int)tmpvector.size(), num_selected_); i++) {
    selected_grasps.push_back(tmpvector[i].first);
    selected_images.push_back(tmpvector[i].second);
  }

  ROS_INFO_STREAM("Selected the " << selected_grasps.size()
                                  << " highest scoring grasps.");

  if (is_benchmark) {
    ofstream fout;
    fout.open("/catkin_ws/src/gpd/annotation/tmp.bin",
              ios::out | ios::binary | ios::trunc);
    cv::Mat output;
    output = selected_images[0];
    for (int i = 0; i < 60; i++) {
      for (int j = 0; j < 60; j++) {
        for (int k = 0; k < 15; k++) {
          fout.write(
              (char *)&output.data[i * output.step + j * output.elemSize() + k],
              sizeof(uint8_t));
        }
      }
    }
    fout.close();
  }

  return selected_grasps;
}

std::vector<GraspSet>
GraspDetector::generateGraspCandidates(const CloudCamera &cloud_cam) {
  return candidates_generator_->generateGraspCandidateSets(cloud_cam);
}

void GraspDetector::preprocessPointCloud(CloudCamera &cloud_cam) {
  candidates_generator_->preprocessPointCloud(cloud_cam);
}

std::vector<Grasp>
GraspDetector::classifyGraspCandidates(const CloudCamera &cloud_cam,
                                       std::vector<GraspSet> &candidates) {
  // Create a grasp image for each grasp candidate.
  double t0 = omp_get_wtime();
  std::cout << "Creating grasp images for classifier input ...\n";
  std::vector<float> scores;
  std::vector<Grasp> grasp_list;
  int num_orientations = candidates[0].getHypotheses().size();
  // Create the grasp images.
  std::vector<cv::Mat> image_list =
      learning_->createImages(cloud_cam, candidates, cleaned_pointcloud_pub);

  std::cout << " Image creation time: " << omp_get_wtime() - t0 << std::endl;

  // std::vector<Grasp> valid_grasps;
  std::vector<cv::Mat> valid_images;
  extractGraspsAndImages(candidates, image_list, valid_grasps, valid_images);
  cout << "length of valid images" << valid_images.size() << std::endl;
  if (valid_images.size() == 0) {
    // Exception
    cout << "returning null grasp list" << endl;
    return grasp_list;
  }
  // Classify the grasp images.
  double t0_prediction = omp_get_wtime();
  scores = classifier_->classifyImages(valid_images);
  grasp_list.assign(valid_grasps.begin(), valid_grasps.end());
  std::cout << " Prediction time: " << omp_get_wtime() - t0 << std::endl;
  //}

  // Select grasps with a score of at least <min_score_diff_>.
  std::vector<Grasp> valid_grasps;

  for (int i = 0; i < grasp_list.size(); i++) {
    if (scores[i] >= min_score_diff_) {
      // std::cout << "grasp #" << i << ", score: " << scores[i] << "\n";
      valid_grasps.push_back(grasp_list[i]);
      screened_images.push_back(valid_images[i]);
      valid_grasps[valid_grasps.size() - 1].setScore(scores[i]);
      valid_grasps[valid_grasps.size() - 1].setFullAntipodal(true);
    }
  }

  std::cout << "Found " << valid_grasps.size()
            << " grasps with a score >= " << min_score_diff_ << "\n";
  std::cout << "Total classification time: " << omp_get_wtime() - t0
            << std::endl;
  return valid_grasps;
}

std::vector<GraspSet>
GraspDetector::filterGraspsWorkspace(const std::vector<GraspSet> &hand_set_list,
                                     const std::vector<double> &workspace) {
  int remaining = 0;
  std::vector<GraspSet> hand_set_list_out;

  for (int i = 0; i < hand_set_list.size(); i++) {
    const std::vector<Grasp> &hands = hand_set_list[i].getHypotheses();
    Eigen::Array<bool, 1, Eigen::Dynamic> is_valid =
        hand_set_list[i].getIsValid();

    for (int j = 0; j < hands.size(); j++) {
      if (is_valid(j)) {
        double half_width = 0.5 * outer_diameter_;
        Eigen::Vector3d left_bottom =
            hands[j].getGraspBottom() + half_width * hands[j].getBinormal();
        Eigen::Vector3d right_bottom =
            hands[j].getGraspBottom() - half_width * hands[j].getBinormal();
        Eigen::Vector3d left_top =
            hands[j].getGraspTop() + half_width * hands[j].getBinormal();
        Eigen::Vector3d right_top =
            hands[j].getGraspTop() - half_width * hands[j].getBinormal();
        Eigen::Vector3d approach =
            hands[j].getGraspBottom() - 0.05 * hands[j].getApproach();
        Eigen::VectorXd x(5), y(5), z(5);
        x << left_bottom(0), right_bottom(0), left_top(0), right_top(0),
            approach(0);
        y << left_bottom(1), right_bottom(1), left_top(1), right_top(1),
            approach(1);
        z << left_bottom(2), right_bottom(2), left_top(2), right_top(2),
            approach(2);
        double aperture = hands[j].getGraspWidth();

        if (aperture >= min_aperture_ &&
            aperture <= max_aperture_ // make sure the object fits into the hand
            && x.minCoeff() >= workspace[0] &&
            x.maxCoeff() <=
                workspace[1] // avoid grasping outside the x-workspace
            && y.minCoeff() >= workspace[2] &&
            y.maxCoeff() <=
                workspace[3] // avoid grasping outside the y-workspace
            && z.minCoeff() >= workspace[4] &&
            z.maxCoeff() <=
                workspace[5]) // avoid grasping outside the z-workspace
        {
          is_valid(j) = true;
          remaining++;
        } else {
          is_valid(j) = false;
        }
      }
    }

    if (is_valid.any()) {
      hand_set_list_out.push_back(hand_set_list[i]);
      hand_set_list_out[hand_set_list_out.size() - 1].setIsValid(is_valid);
    }
  }

  ROS_INFO_STREAM("# grasps within workspace and gripper width: " << remaining);

  return hand_set_list_out;
}

std::vector<GraspSet>
GraspDetector::filterHalfAntipodal(const std::vector<GraspSet> &hand_set_list) {
  int remaining = 0;
  std::vector<GraspSet> hand_set_list_out;

  for (int i = 0; i < hand_set_list.size(); i++) {
    const std::vector<Grasp> &hands = hand_set_list[i].getHypotheses();
    Eigen::Array<bool, 1, Eigen::Dynamic> is_valid =
        hand_set_list[i].getIsValid();

    for (int j = 0; j < hands.size(); j++) {
      if (is_valid(j)) {
        if (!hands[j].isHalfAntipodal() || hands[j].isFullAntipodal()) {
          is_valid(j) = true;
          remaining++;
        } else {
          is_valid(j) = false;
        }
      }
    }

    if (is_valid.any()) {
      hand_set_list_out.push_back(hand_set_list[i]);
      hand_set_list_out[hand_set_list_out.size() - 1].setIsValid(is_valid);
    }
  }

  ROS_INFO_STREAM("# grasps that are not half-antipodal: " << remaining);

  return hand_set_list_out;
}

std::vector<Grasp>
GraspDetector::extractHypotheses(const std::vector<GraspSet> &hand_set_list) {
  std::vector<Grasp> hands_out;
  hands_out.resize(0);

  for (int i = 0; i < hand_set_list.size(); i++) {
    const std::vector<Grasp> &hands = hand_set_list[i].getHypotheses();

    for (int j = 0; j < hands.size(); j++) {
      if (hand_set_list[i].getIsValid()(j)) {
        hands_out.push_back(hands[j]);
      }
    }
  }

  return hands_out;
}

void GraspDetector::extractGraspsAndImages(
    const std::vector<GraspSet> &hand_set_list,
    const std::vector<cv::Mat> &images, std::vector<Grasp> &grasps_out,
    std::vector<cv::Mat> &images_out) {
  grasps_out.resize(0);
  images_out.resize(0);
  int num_orientations = hand_set_list[0].getHypotheses().size();

  for (int i = 0; i < hand_set_list.size(); i++) {
    const std::vector<Grasp> &hands = hand_set_list[i].getHypotheses();

    for (int j = 0; j < hands.size(); j++) {
      if (hand_set_list[i].getIsValid()(j)) {
        grasps_out.push_back(hands[j]);
        images_out.push_back(images[i * num_orientations + j]);
      }
    }
  }
}

// std::vector<Grasp>
std::tuple<std::vector<Grasp>, std::vector<cv::Mat>>
GraspDetector::findClusters(const std::vector<Grasp> &grasps,
                            std::vector<cv::Mat> &screened_images, bool remove_inliers) {
  return clustering_->findClusters(grasps, screened_images, remove_inliers);
}

std::vector<Grasp>
GraspDetector::findClusters(const std::vector<Grasp> &grasps) {
  return clustering_->findClusters(grasps);
}
