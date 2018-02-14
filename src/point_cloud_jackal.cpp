#include <iostream>
#include <math.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_parallel_plane.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>

#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <point_cloud_jackal/PlanarSegmentation.h>
#include <point_cloud_jackal/Plane.h>

class PointCloudProc
{
  public:
    PointCloudProc() : cloud_transformed_(new pcl::PointCloud<pcl::PointXYZ>), cloud_filtered_(new pcl::PointCloud<pcl::PointXYZ>),
                       cloud_hull(new pcl::PointCloud<pcl::PointXYZ>), cloud_raw_(new pcl::PointCloud<pcl::PointXYZ>)
    {
        filter_range_ = 1.0;

        planar_segment_src_ = nh_.advertiseService("planer_segment", &PointCloudProc::planarSegmentationCB, this);
        pc_sub_ = nh_.subscribe("/kinect2/qhd/points", 1, &PointCloudProc::pointcloudcb, this);
        point_cloud_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZ>>("segmented_plane_point_cloud", 1000);
        fixed_frame_ = "/base_link";
        //listener_ = new tf::TransformListener();
    }

    void pointcloudcb(const pcl::PointCloud<pcl::PointXYZ>::Ptr &msg)
    {
        cloud_raw_ = msg;
        cloud_transformed_ = cloud_raw_;
        //listener_->waitForTransform(fixed_frame_, (*msg).header.frame_id, (*msg).header.stamp , ros::Duration(5.0));
    }

    bool transformPointCloud()
    {
        bool transform_success = pcl_ros::transformPointCloud(fixed_frame_, *cloud_raw_, *cloud_transformed_, listener_);
        return transform_success;
    }

    bool filterPointCloud(geometry_msgs::Point center)
    {
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud_transformed_);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(center.x - filter_range_, center.x + filter_range_);
        pass.filter(*cloud_filtered_);
        pass.setInputCloud(cloud_filtered_);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(center.y - filter_range_, center.y + filter_range_);
        pass.filter(*cloud_filtered_);
        pass.setInputCloud(cloud_filtered_);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(center.z - filter_range_, center.z + filter_range_);
        pass.filter(*cloud_filtered_);

        ROS_INFO("Point cloud is filtered!");
        if (cloud_filtered_->points.size() == 0)
        {
            ROS_WARN("Point cloud is empty after filtering!");
            return false;
        }

        return true;
    }

    bool planarSegmentation(bool create_srv_res)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        Eigen::Vector3f axis = Eigen::Vector3f(0.0, 0.0, 1.0); //z axis
        // Create the segmentation object
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        // Optional
        seg.setOptimizeCoefficients(true);
        // Mandatory set plane to be parallel to Z axis within a 15 degrees tolerance
        seg.setModelType(pcl::SACMODEL_PARALLEL_PLANE);
        seg.setMaxIterations(500); // iteration limits decides segmentation goodness
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setAxis(axis);
        seg.setEpsAngle(pcl::deg2rad(15.0f));
        seg.setDistanceThreshold(0.01);
        seg.setInputCloud(cloud_filtered_);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.size() == 0)
        {
            PCL_ERROR("Could not estimate a planar model for the given dataset.");
            return false;
        }

        // extract inline points original point coulds
        extract_.setInputCloud(cloud_filtered_);
        extract_.setNegative(false);
        extract_.setIndices(inliers);
        extract_.filter(*cloud_plane);
        ROS_INFO_STREAM("# of points in plane: " << cloud_plane->points.size());

        // publish segmented point cloud
        point_cloud_pub_.publish(cloud_filtered_);

        // Create a Convex Hull representation of the plane
        chull.setInputCloud(cloud_plane);
        chull.setDimension(2);
        chull.reconstruct(*cloud_hull);

        if (create_srv_res)
        {
            // Construct plane object msg
            point_cloud_jackal::Plane plane_object_msg;

            pcl_conversions::fromPCL(cloud_plane->header, plane_object_msg.header);

            // Get plane center
            Eigen::Vector4f center;
            pcl::compute3DCentroid(*cloud_plane, center);
            plane_object_msg.center.x = center[0];
            plane_object_msg.center.y = center[1];
            plane_object_msg.center.z = center[2];

            // Get plane min and max values
            Eigen::Vector4f min_vals, max_vals;
            pcl::getMinMax3D(*cloud_plane, min_vals, max_vals);

            plane_object_msg.min.x = min_vals[0];
            plane_object_msg.min.y = min_vals[1];
            plane_object_msg.min.z = min_vals[2];

            plane_object_msg.max.x = max_vals[0];
            plane_object_msg.max.y = max_vals[1];
            plane_object_msg.max.z = max_vals[2];

            // Get plane polygon
            for (int i = 0; i < cloud_hull->points.size(); i++)
            {
                geometry_msgs::Point32 p;
                p.x = cloud_hull->points[i].x;
                p.y = cloud_hull->points[i].y;
                p.z = cloud_hull->points[i].z;

                plane_object_msg.polygon.push_back(p);
            }

            // Get plane coefficients
            plane_object_msg.coef[0] = coefficients->values[0];
            plane_object_msg.coef[1] = coefficients->values[1];
            plane_object_msg.coef[2] = coefficients->values[2];
            plane_object_msg.coef[3] = coefficients->values[3];

            // Get plane normal
            float length = sqrt(coefficients->values[0] * coefficients->values[0] +
                                coefficients->values[1] * coefficients->values[1] +
                                coefficients->values[2] * coefficients->values[2]);
            plane_object_msg.normal[0] = coefficients->values[0] / length;
            plane_object_msg.normal[1] = coefficients->values[1] / length;
            plane_object_msg.normal[2] = coefficients->values[2] / length;

            plane_object_msg.size.data = cloud_plane->points.size();
            plane_object_msg.is_vertical = true;
            plane_object_ = plane_object_msg;
        }

        return true;
    }

    bool planarSegmentationCB(point_cloud_jackal::PlanarSegmentation::Request &req,
                              point_cloud_jackal::PlanarSegmentation::Response &res)
    {
        ROS_INFO("stage 0");
        if (!this->transformPointCloud())
        {
            ROS_INFO("failed transform point cloud");
            res.success = false;
            return true;
        }

        if (!this->filterPointCloud(req.center))
        {
            ROS_INFO("failed filter point cloud");
            res.success = false;
            return true;
        }

        ROS_INFO("stage 1");
        if (this->planarSegmentation(true))
        {
            res.success = true;
            res.plane_object = plane_object_;
            return true;
        }
        else
        {
            res.success = false;
            return true;
        }
    }

  private:
    ros::NodeHandle nh_;
    ros::Subscriber pc_sub_;
    ros::Publisher plane_pub_;
    ros::Publisher point_cloud_pub_;
    ros::ServiceServer planar_segment_src_;

    tf::TransformListener listener_;
    std::string fixed_frame_;

    point_cloud_jackal::Plane plane_object_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_raw_, cloud_transformed_, cloud_filtered_;
    pcl::ExtractIndices<pcl::PointXYZ> extract_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull;
    pcl::ConvexHull<pcl::PointXYZ> chull;
    float filter_range_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "point_cloud_jackal");

    PointCloudProc pc_tools;
    ROS_INFO("service initialized");
    ros::spin();

    return 0;
}
