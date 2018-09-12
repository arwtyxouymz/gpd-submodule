#include "../../include/gpd/clustering.h"
#include <tuple>

// std::vector<Grasp> Clustering::findClusters
std::tuple<std::vector<Grasp>, std::vector<cv::Mat>> Clustering::findClusters(const std::vector<Grasp>& hand_list, std::vector<cv::Mat>& screened_images, bool remove_inliers)
{
  // const double AXIS_ALIGN_ANGLE_THRESH = 15.0 * M_PI/180.0;
  const double AXIS_ALIGN_ANGLE_THRESH = 12.0 * M_PI/180.0;
  const double AXIS_ALIGN_DIST_THRESH = 0.005;
  // const double MAX_DIST_THRESH = 0.07;
  const double MAX_DIST_THRESH = 0.05;
  //  const int max_inliers = 50;

  std::vector<Grasp> hands_out;
  std::vector<bool> has_used;
  if (remove_inliers)
  {
    has_used.resize(hand_list.size());
    for (int i = 0; i < hand_list.size(); i++)
    {
      has_used[i] = false;
    }
  }
  std::vector<cv::Mat> images_out;
  std::vector<int> inliers;

  for (int i = 0; i < hand_list.size(); i++)
  {
    int num_inliers = 0;
    Eigen::Vector3d position_delta = Eigen::Vector3d::Zero();
    Eigen::Matrix3d axis_outer_prod = hand_list[i].getAxis() * hand_list[i].getAxis().transpose();
    inliers.resize(0);
    double mean = 0.0;
    double standard_deviation = 0.0;
    std::vector<double> scores(0);

    for (int j = 0; j < hand_list.size(); j++)
    {
      if (i == j || (remove_inliers && has_used[j]))
        continue;

      // Which hands have an axis within <AXIS_ALIGN_ANGLE_THRESH> of this one?
      double axis_aligned = hand_list[i].getAxis().transpose() * hand_list[j].getAxis();
      bool axis_aligned_binary = fabs(axis_aligned) > cos(AXIS_ALIGN_ANGLE_THRESH);

      // Which hands are within <MAX_DIST_THRESH> of this one?
      Eigen::Vector3d delta_pos = hand_list[i].getGraspBottom() - hand_list[j].getGraspBottom();
      double delta_pos_mag = delta_pos.norm();
      bool delta_pos_mag_binary = delta_pos_mag <= MAX_DIST_THRESH;

      // Which hands are within <AXIS_ALIGN_DIST_THRESH> of this one when projected onto the plane orthognal to this
      // one's axis?
      Eigen::Matrix3d axis_orth_proj = Eigen::Matrix3d::Identity() - axis_outer_prod;
      Eigen::Vector3d delta_pos_proj = axis_orth_proj * delta_pos;
      double delta_pos_proj_mag = delta_pos_proj.norm();
      bool delta_pos_proj_mag_binary = delta_pos_proj_mag <= AXIS_ALIGN_DIST_THRESH;

      bool inlier_binary = axis_aligned_binary && delta_pos_mag_binary && delta_pos_proj_mag_binary;
      if (inlier_binary)
      {
        inliers.push_back(i);
        scores.push_back(hand_list[j].getScore());
        num_inliers++;
        position_delta += hand_list[j].getGraspBottom();
        mean += hand_list[j].getScore();
        standard_deviation += hand_list[j].getScore() * hand_list[j].getScore();
        if (remove_inliers)
        {
          has_used[j] = true;
        }
      }
    }
    //std::cout << "num_inliers" << num_inliers << std::endl;
    if (num_inliers >= min_inliers_)
    {
      position_delta = position_delta / (double) num_inliers - hand_list[i].getGraspBottom();
      mean = std::max(mean / (double) num_inliers, (double) num_inliers);
      standard_deviation = standard_deviation == 0.0 ? 0.0 : sqrt(standard_deviation/(double) num_inliers - mean*mean);
      std::nth_element(scores.begin(), scores.begin() + scores.size()/2, scores.end());
      double median = scores[scores.size()/2];
      double conf_lb = mean - 2.576*standard_deviation/sqrt((double)num_inliers);
      double conf_ub = mean + 2.576*standard_deviation/sqrt((double)num_inliers);
//      std::cout << "grasp " << i << ", num_inliers: " << num_inliers << ", ||pos_delta||: " << position_delta.norm()
//        << ", mean: " << mean << ", std: " << standard_deviation << ", median: " << median << ", conf: " << conf_lb
//        << ", " << conf_ub << "\n";
      Grasp hand = hand_list[i];
      hand.setGraspSurface(hand.getGraspSurface() + position_delta);
      hand.setGraspBottom(hand.getGraspBottom() + position_delta);
      hand.setGraspTop(hand.getGraspTop() + position_delta);
      // hand.setScore(avg_score);
      hand.setScore(conf_lb);
      hand.setFullAntipodal(hand_list[i].isFullAntipodal());
      hands_out.push_back(hand);
      images_out.push_back(screened_images[i]);
    }
  }

  return std::forward_as_tuple(hands_out, images_out);
}

std::vector<Grasp> Clustering::findClusters(const std::vector<Grasp>& hand_list, bool remove_inliers)
{
  // const double AXIS_ALIGN_ANGLE_THRESH = 15.0 * M_PI/180.0;
  const double AXIS_ALIGN_ANGLE_THRESH = 12.0 * M_PI/180.0;
  const double AXIS_ALIGN_DIST_THRESH = 0.005;
  // const double MAX_DIST_THRESH = 0.07;
  const double MAX_DIST_THRESH = 0.05;
  //  const int max_inliers = 50;

  std::vector<Grasp> hands_out;
  std::vector<bool> has_used;
  if (remove_inliers)
  {
    has_used.resize(hand_list.size());
    for (int i = 0; i < hand_list.size(); i++)
    {
      has_used[i] = false;
    }
  }

  std::vector<int> inliers;

  for (int i = 0; i < hand_list.size(); i++)
  {
    int num_inliers = 0;
    Eigen::Vector3d position_delta = Eigen::Vector3d::Zero();
    Eigen::Matrix3d axis_outer_prod = hand_list[i].getAxis() * hand_list[i].getAxis().transpose();
    inliers.resize(0);
    double mean = 0.0;
    double standard_deviation = 0.0;
    std::vector<double> scores(0);

    for (int j = 0; j < hand_list.size(); j++)
    {
      if (i == j || (remove_inliers && has_used[j]))
        continue;

      // Which hands have an axis within <AXIS_ALIGN_ANGLE_THRESH> of this one?
      double axis_aligned = hand_list[i].getAxis().transpose() * hand_list[j].getAxis();
      bool axis_aligned_binary = fabs(axis_aligned) > cos(AXIS_ALIGN_ANGLE_THRESH);

      // Which hands are within <MAX_DIST_THRESH> of this one?
      Eigen::Vector3d delta_pos = hand_list[i].getGraspBottom() - hand_list[j].getGraspBottom();
      double delta_pos_mag = delta_pos.norm();
      bool delta_pos_mag_binary = delta_pos_mag <= MAX_DIST_THRESH;

      // Which hands are within <AXIS_ALIGN_DIST_THRESH> of this one when projected onto the plane orthognal to this
      // one's axis?
      Eigen::Matrix3d axis_orth_proj = Eigen::Matrix3d::Identity() - axis_outer_prod;
      Eigen::Vector3d delta_pos_proj = axis_orth_proj * delta_pos;
      double delta_pos_proj_mag = delta_pos_proj.norm();
      bool delta_pos_proj_mag_binary = delta_pos_proj_mag <= AXIS_ALIGN_DIST_THRESH;

      bool inlier_binary = axis_aligned_binary && delta_pos_mag_binary && delta_pos_proj_mag_binary;
      if (inlier_binary)
      {
        inliers.push_back(i);
        scores.push_back(hand_list[j].getScore());
        num_inliers++;
        position_delta += hand_list[j].getGraspBottom();
        mean += hand_list[j].getScore();
        standard_deviation += hand_list[j].getScore() * hand_list[j].getScore();
        if (remove_inliers)
        {
          has_used[j] = true;
        }
      }
    }

    if (num_inliers != 0 && num_inliers >= min_inliers_)
    {
      position_delta = position_delta / (double) num_inliers - hand_list[i].getGraspBottom();
      mean = std::max(mean / (double) num_inliers, (double) num_inliers);
      standard_deviation = standard_deviation == 0.0 ? 0.0 : sqrt(standard_deviation/(double) num_inliers - mean*mean);
      std::nth_element(scores.begin(), scores.begin() + scores.size()/2, scores.end());
      double median = scores[scores.size()/2];
      double conf_lb = mean - 2.576*standard_deviation/sqrt((double)num_inliers);
      double conf_ub = mean + 2.576*standard_deviation/sqrt((double)num_inliers);
     // std::cout << "grasp " << i << ", num_inliers: " << num_inliers << ", ||pos_delta||: " << position_delta.norm()
     //   << ", mean: " << mean << ", std: " << standard_deviation << ", median: " << median << ", conf: " << conf_lb
     //   << ", " << conf_ub << "\n";
      Grasp hand = hand_list[i];
      hand.setGraspSurface(hand.getGraspSurface() + position_delta);
      hand.setGraspBottom(hand.getGraspBottom() + position_delta);
      hand.setGraspTop(hand.getGraspTop() + position_delta);
      // hand.setScore(avg_score);
      hand.setScore(conf_lb);
      hand.setFullAntipodal(hand_list[i].isFullAntipodal());
      hands_out.push_back(hand);
    }
  }

  return hands_out;
}
