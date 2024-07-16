#include <iostream>

#include "adaptive_clustering/ClusterArray.h"
#include <geometry_msgs/PoseArray.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>

using namespace std;

ros::Publisher cluster_array_pub_;
ros::Publisher cloud_filtered_pub_;
ros::Publisher pose_array_pub_;
ros::Publisher marker_array_pub_;

bool print_fps_;
int leaf_;
float z_axis_min_;
float z_axis_max_;
float radius_min_;
float radius_max_;
int cluster_size_min_;
int cluster_size_max_;
float k_merging_threshold_;
float z_merging_threshold_;

const int region_max_ = 10; // how far to detect
int regions_[100];

int frames;
clock_t start_time;
bool reset = true;

void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& ros_pc2_in) {
  if (print_fps_) {
      if (reset) {
          frames = 0;
          start_time = clock();
          reset = false;
      }
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_pc_in(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*ros_pc2_in, *pcl_pc_in);
  cerr << "Received point with " << pcl_pc_in->points.size() << "points" << endl; 

  // Downsampling + ground & ceiling removal
  pcl::IndicesPtr pc_indices(new std::vector<int>);
  for (int i = 0; i < pcl_pc_in->size(); ++i) {
    if (i % leaf_ == 0) {
      if (pcl_pc_in->points[i].z >= z_axis_min_ && pcl_pc_in->points[i].z <= z_axis_max_) {
        pc_indices->push_back(i);
      }
    }
  }
  cerr << "Filtered " << pc_indices->size() << " points afer ground & ceiling removal" << endl;

  // Divide point cloud into nested circular regions
  boost::array<std::vector<int>, region_max_> indices_array;
  for (int i = 0; i < pc_indices->size(); ++i) {
    float range = 0.0;
    for (int j = 0; j < region_max_; ++j) {
      float d2 = pcl_pc_in->points[(*pc_indices)[i]].x * pcl_pc_in->points[(*pc_indices)[i]].x +
                 pcl_pc_in->points[(*pc_indices)[i]].y * pcl_pc_in->points[(*pc_indices)[i]].y +
                 pcl_pc_in->points[(*pc_indices)[i]].z * pcl_pc_in->points[(*pc_indices)[i]].z;
      if (d2 > radius_min_ * radius_min_ &&
        d2 < radius_max_ * radius_max_ &&
        d2 > range * range && d2 <= (range + regions_[j]) * (range + regions_[j])) { // 修正了这里的 regions_[i] 为 regions_[j]
        indices_array[j].push_back((*pc_indices)[i]);
        break;
      }
      range += regions_[j];
    }
  }

  float tolerance = 0.0;
  vector<pcl::PointCloud<pcl::PointXYZ>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<pcl::PointXYZ>::Ptr>> clusters;
  int last_clusters_begin = 0, last_clusters_end = 0;

  for (int i = 0; i < region_max_; ++i) {
    tolerance += 0.1;
    if (indices_array[i].size() > cluster_size_min_) {
      boost::shared_ptr<std::vector<int>> indices_array_ptr(new std::vector<int>(indices_array[i]));
      pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
      tree->setInputCloud(pcl_pc_in, indices_array_ptr);

      vector<pcl::PointIndices> cluster_indices;
      pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
      ec.setClusterTolerance(tolerance);
      ec.setMinClusterSize(cluster_size_min_);
      ec.setMaxClusterSize(cluster_size_max_);
      ec.setSearchMethod(tree);
      ec.setInputCloud(pcl_pc_in);
      ec.setIndices(indices_array_ptr);
      ec.extract(cluster_indices);

      for (vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
          cluster->points.push_back(pcl_pc_in->points[*pit]);
        }
        for (int j = last_clusters_begin; j < last_clusters_end; ++j) {
          pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
          int K = 1;
          vector<int> k_indices(K);
          vector<float> k_sqr_distances(K);
          kdtree.setInputCloud(cluster);
          if (clusters[j]->points.size() >= 1) {
            if (kdtree.nearestKSearch(*clusters[j], clusters[j]->points.size() - 1, K, k_indices, k_sqr_distances) > 0) {
              if (k_sqr_distances[0] < k_merging_threshold_) { // 修正了这里的 逗号 为小于号
                *cluster += *clusters[j];
                clusters.erase(clusters.begin() + j);
                last_clusters_end--;
              }
            }
          }
        }
        cluster->width = cluster->size();
        cluster->height = 1;
        cluster->is_dense = true;
        clusters.push_back(cluster);
      }
      for (int j = last_clusters_end; j < clusters.size(); ++j) {
        Eigen::Vector4f j_min, j_max;
        pcl::getMinMax3D(*clusters[j], j_min, j_max);
        for (int k = j + 1; k < clusters.size(); ++k) {
          Eigen::Vector4f k_min, k_max;
          pcl::getMinMax3D(*clusters[k], k_min, k_max);
          if (std::max(std::min((double)j_max[0], (double)k_max[0]) - std::max((double)j_min[0], (double)k_min[0]), 0.0) * 
            std::max(std::min((double)j_max[1], (double)k_max[1]) - std::max((double)j_min[1], (double)k_min[1]), 0.0)
            > z_merging_threshold_) {
            *clusters[j] += *clusters[k];
            clusters.erase(clusters.begin() + k);   
          }
        }
      }
      last_clusters_begin = last_clusters_end;
      last_clusters_end = clusters.size();
    }
  }

  if (cloud_filtered_pub_.getNumSubscribers() > 0) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_pc_out(new pcl::PointCloud<pcl::PointXYZ>);
    sensor_msgs::PointCloud2 ros_pc2_out;
    pcl::copyPointCloud(*pcl_pc_in, *pc_indices, *pcl_pc_out);
    pcl::toROSMsg(*pcl_pc_out, ros_pc2_out);
    cloud_filtered_pub_.publish(ros_pc2_out);
  }

  adaptive_clustering::ClusterArray cluster_array;
  geometry_msgs::PoseArray pose_array;
  visualization_msgs::MarkerArray marker_array;

  for (int i = 0; i < clusters.size(); ++i) {
    if (cluster_array_pub_.getNumSubscribers() > 0) {
      sensor_msgs::PointCloud2 ros_pc2_out;
      pcl::toROSMsg(*clusters[i], ros_pc2_out);
      cluster_array.clusters.push_back(ros_pc2_out);
    }

    if (pose_array_pub_.getNumSubscribers() > 0) {
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*clusters[i], centroid);

      geometry_msgs::Pose pose;
      pose.position.x = centroid[0];
      pose.position.y = centroid[1];
      pose.position.z = centroid[2];
      pose.orientation.w = 1;
      pose_array.poses.push_back(pose);
    }

    if (marker_array_pub_.getNumSubscribers() > 0) {
        Eigen::Vector4f min, max;
        pcl::getMinMax3D(*clusters[i], min, max);

        visualization_msgs::Marker marker;
        marker.header = ros_pc2_in->header;
        marker.ns = "adaptive_clustering";
        marker.id = i;
        marker.type = visualization_msgs::Marker::LINE_LIST;

        geometry_msgs::Point p[24];
        p[0].x = max[0];  p[0].y = max[1];  p[0].z = max[2];
        p[1].x = min[0];  p[1].y = max[1];  p[1].z = max[2];
        p[2].x = max[0];  p[2].y = max[1];  p[2].z = max[2];
        p[3].x = max[0];  p[3].y = min[1];  p[3].z = max[2];
        p[4].x = max[0];  p[4].y = max[1];  p[4].z = max[2];
        p[5].x = max[0];  p[5].y = max[1];  p[5].z = min[2];
        p[6].x = min[0];  p[6].y = min[1];  p[6].z = min[2];
        p[7].x = max[0];  p[7].y = min[1];  p[7].z = min[2];
        p[8].x = min[0];  p[8].y = min[1];  p[8].z = min[2];
        p[9].x = min[0];  p[9].y = max[1];  p[9].z = min[2];
        p[10].x = min[0]; p[10].y = min[1]; p[10].z = min[2];
        p[11].x = min[0]; p[11].y = min[1]; p[11].z = max[2];
        p[12].x = min[0]; p[12].y = max[1]; p[12].z = max[2];
        p[13].x = min[0]; p[13].y = max[1]; p[13].z = min[2];
        p[14].x = min[0]; p[14].y = max[1]; p[14].z = max[2];
        p[15].x = min[0]; p[15].y = min[1]; p[15].z = max[2];
        p[16].x = max[0]; p[16].y = min[1]; p[16].z = max[2];
        p[17].x = max[0]; p[17].y = min[1]; p[17].z = min[2];
        p[18].x = max[0]; p[18].y = min[1]; p[18].z = max[2];
        p[19].x = min[0]; p[19].y = min[1]; p[19].z = max[2];
        p[20].x = max[0]; p[20].y = max[1]; p[20].z = min[2];
        p[21].x = min[0]; p[21].y = max[1]; p[21].z = min[2];
        p[22].x = max[0]; p[22].y = max[1]; p[22].z = min[2];
        p[23].x = max[0]; p[23].y = min[1]; p[23].z = min[2];

        for (int k = 0; k < 24; ++k) {
          marker.points.push_back(p[k]);
        }
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.02;
        marker.color.a = 1.0;
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.5;
        marker.lifetime = ros::Duration(0.1);
        marker_array.markers.push_back(marker);
    }
  }

  if (cluster_array.clusters.size()) {
    cluster_array.header = ros_pc2_in->header;
    cluster_array_pub_.publish(cluster_array);
  }

  if (pose_array.poses.size()) {
    pose_array.header = ros_pc2_in->header;
    pose_array_pub_.publish(pose_array);
  }

  if (marker_array.markers.size()) {
    marker_array_pub_.publish(marker_array);
  }

  if (print_fps_) {
    if (++frames > 10) {
      cerr << "[adaptive_clustering] fps = " << float(frames) / (float(clock() - start_time) / CLOCKS_PER_SEC)
           << ", timestamp = " << clock() / CLOCKS_PER_SEC << endl;
      reset = true;
    }
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "adaptive_clustering");
  ros::NodeHandle nh_("~");
  ros::Subscriber point_cloud_sub_ = nh_.subscribe<sensor_msgs::PointCloud2>("/livox/lidar", 1, pointCloudCallback);
  cluster_array_pub_ = nh_.advertise<adaptive_clustering::ClusterArray>("clusters", 100);
  cloud_filtered_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("cloud_filtered", 100);
  pose_array_pub_ = nh_.advertise<geometry_msgs::PoseArray>("poses", 100);
  marker_array_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("markers", 100);

  string sensor_model;

  nh_.param<string>("sensor_model", sensor_model, "Livox-HAP");
  nh_.param<bool>("print_fps", print_fps_, false);
  nh_.param<int>("leaf", leaf_, 3);
  nh_.param<float>("z_axis_min", z_axis_min_, -0.8);
  nh_.param<float>("z_axis_max", z_axis_max_, 2.0);
  nh_.param<float>("radius_min", radius_min_, 0);
  nh_.param<float>("radius_max", radius_max_, 30);
  nh_.param<int>("cluster_size_min", cluster_size_min_, 3);
  nh_.param<int>("cluster_size_max", cluster_size_max_, 2200000);
  nh_.param<float>("k_merging_threshold", k_merging_threshold_, 0.1);
  nh_.param<float>("z_merging_threshold", z_merging_threshold_, 0.0);

  if (sensor_model.compare("Livox-HAP") == 0) {
    ROS_INFO("Sensor model is matching to Livox HAP Lidar");
    regions_[0] = 2; regions_[1] = 3; regions_[2] = 3; regions_[3] = 3; regions_[4] = 3;
    regions_[5] = 3; regions_[6] = 3; regions_[7] = 2; regions_[8] = 3; regions_[9] = 3;
    regions_[10]= 3; regions_[11]= 3; regions_[12]= 3; regions_[13]= 3;
  } else if (sensor_model.compare("VLP-16") == 0) {
    ROS_INFO("Sensor model is matching to VLP 16 Lidar");
    regions_[0] = 2; regions_[1] = 3; regions_[2] = 3; regions_[3] = 3; regions_[4] = 3;
    regions_[5] = 3; regions_[6] = 3; regions_[7] = 2; regions_[8] = 3; regions_[9] = 3;
    regions_[10]= 3; regions_[11]= 3; regions_[12]= 3; regions_[13]= 3;
  } else if (sensor_model.compare("HDL-32E") == 0) {
    regions_[0] = 4; regions_[1] = 5; regions_[2] = 4; regions_[3] = 5; regions_[4] = 4;
    regions_[5] = 5; regions_[6] = 5; regions_[7] = 4; regions_[8] = 5; regions_[9] = 4;
    regions_[10]= 5; regions_[11]= 5; regions_[12]= 4; regions_[13]= 5;
  } else if (sensor_model.compare("HDL-64E") == 0) {
    regions_[0] = 14; regions_[1] = 14; regions_[2] = 14; regions_[3] = 15; regions_[4] = 14;
  } else {
    ROS_FATAL("Unknown sensor model!");
  }
  ros::spin();
  
  return 0;
}