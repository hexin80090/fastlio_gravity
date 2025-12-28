#pragma once
#include <mutex>
#include <string>
#include <queue>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <livox_ros_driver2/CustomMsg.h>
#include "IKFoM_toolkit/esekfom/esekfom.hpp"

#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TransformStamped.h>
#include <pcl_conversions/pcl_conversions.h>

namespace fastlio
{
#define NUM_MATCH_POINTS (5)

#define SKEW_SYM_MATRX(v) 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0

#define NUM_MAX_POINTS (10000)
    const double G_m_s2 = 9.81;
    typedef pcl::PointXYZINormal PointType;
    typedef pcl::PointCloud<PointType> PointCloudXYZI;
    typedef std::vector<PointType, Eigen::aligned_allocator<PointType>> PointVector;
    // typedef esekfom::esekf<state_ikfom, 12, input_ikfom> ESEKF;
    struct IMU
    {
        IMU() : acc(Eigen::Vector3d::Zero()), gyro(Eigen::Vector3d::Zero()) {}
        IMU(double t, Eigen::Vector3d a, Eigen::Vector3d g)
            : timestamp(t), acc(a), gyro(g) {}
        IMU(double t, double a1, double a2, double a3, double g1, double g2, double g3)
            : timestamp(t), acc(a1, a2, a3), gyro(g1, g2, g3) {}
        double timestamp;
        Eigen::Vector3d acc;
        Eigen::Vector3d gyro;
    };

    bool esti_plane(Eigen::Vector4d &out, const PointVector &points, const double &thresh);

    float sq_dist(const PointType &p1, const PointType &p2);

    typedef MTK::vect<3, double> vect3;
    typedef MTK::SO3<double> SO3;
    typedef MTK::S2<double, 98090, 10000, 1> S2;
    typedef MTK::vect<1, double> vect1;
    typedef MTK::vect<2, double> vect2;
    MTK_BUILD_MANIFOLD(state_ikfom,
                       ((vect3, pos))((SO3, rot))((SO3, offset_R_L_I))((vect3, offset_T_L_I))((vect3, vel))((vect3, bg))((vect3, ba))((S2, grav)));

    MTK_BUILD_MANIFOLD(input_ikfom,
                       ((vect3, acc))((vect3, gyro)));

    MTK_BUILD_MANIFOLD(process_noise_ikfom,
                       ((vect3, ng))((vect3, na))((vect3, nbg))((vect3, nba)));

    MTK::get_cov<process_noise_ikfom>::type process_noise_cov();
    Eigen::Matrix<double, 24, 1> get_f(state_ikfom &s, const input_ikfom &in);
    Eigen::Matrix<double, 24, 23> df_dx(state_ikfom &s, const input_ikfom &in);
    Eigen::Matrix<double, 24, 12> df_dw(state_ikfom &s, const input_ikfom &in);

    template <typename T, typename Ts>
    Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1> &ang_vel, const Ts &dt)
    {
        T ang_vel_norm = ang_vel.norm();
        Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();

        if (ang_vel_norm > 0.0000001)
        {
            Eigen::Matrix<T, 3, 1> r_axis = ang_vel / ang_vel_norm;
            Eigen::Matrix<T, 3, 3> K;

            K << SKEW_SYM_MATRX(r_axis);

            T r_ang = ang_vel_norm * dt;

            /// Roderigous Tranformation
            return Eye3 + std::sin(r_ang) * K + (1.0 - std::cos(r_ang)) * K * K;
        }
        else
        {
            return Eye3;
        }
    }
}
struct ImuData
{
    std::string topic;
    std::mutex mutex;
    std::deque<fastlio::IMU> buffer;
    double last_timestamp = 0;
    void callback(const sensor_msgs::Imu::ConstPtr &msg);
};

struct LivoxData
{
    std::string topic;
    std::mutex mutex;
    std::deque<fastlio::PointCloudXYZI::Ptr> buffer;
    std::deque<double> time_buffer;
    double blind = 0.5;
    int filter_num = 3;
    double last_timestamp = 0;
    void callback(const livox_ros_driver2::CustomMsg::ConstPtr &msg);
    void livox2pcl(const livox_ros_driver2::CustomMsg::ConstPtr &msg, fastlio::PointCloudXYZI::Ptr &out);
};

struct MeasureGroup
{
    double lidar_time_begin = 0.0;
    double lidar_time_end = 0.0;
    bool lidar_pushed = false;
    fastlio::PointCloudXYZI::Ptr lidar;
    std::deque<fastlio::IMU> imus;
    bool syncPackage(ImuData &imu_data, LivoxData &livox_data);
};

nav_msgs::Odometry eigen2Odometry(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp);

// 生成完整的Odometry（包含速度、角速度、协方差），格式与faster-lio/fast_lio_localization一致
template<typename MatrixType>
nav_msgs::Odometry eigen2OdometryFull(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, 
                                      const Eigen::Vector3d &vel, const Eigen::Vector3d &angular_vel,
                                      const MatrixType &covariance,
                                      const std::string &frame_id, const std::string &child_frame_id, 
                                      const double &timestamp)
{
    nav_msgs::Odometry odom;
    odom.header.frame_id = frame_id;
    odom.header.stamp = ros::Time().fromSec(timestamp);
    odom.child_frame_id = child_frame_id;
    
    // 位置和姿态
    Eigen::Quaterniond q = Eigen::Quaterniond(rot);
    odom.pose.pose.position.x = pos(0);
    odom.pose.pose.position.y = pos(1);
    odom.pose.pose.position.z = pos(2);
    odom.pose.pose.orientation.w = q.w();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    
    // 速度（线性）
    odom.twist.twist.linear.x = vel(0);
    odom.twist.twist.linear.y = vel(1);
    odom.twist.twist.linear.z = vel(2);
    
    // 角速度
    odom.twist.twist.angular.x = angular_vel(0);
    odom.twist.twist.angular.y = angular_vel(1);
    odom.twist.twist.angular.z = angular_vel(2);
    
    // 协方差矩阵（6x6，位置+姿态）
    // 按照 faster-lio/fast_lio_localization 的格式填充协方差
    // 顺序：x, y, z, roll, pitch, yaw
    // 协方差矩阵是 23x23，我们需要提取位置(3,4,5)和姿态(0,1,2)的协方差
    if (covariance.rows() >= 6 && covariance.cols() >= 6)
    {
        for (int i = 0; i < 6; i++)
        {
            int k = i < 3 ? i + 3 : i - 3;  // 位置索引：3,4,5，姿态索引：0,1,2
            odom.pose.covariance[i * 6 + 0] = covariance(k, 3);
            odom.pose.covariance[i * 6 + 1] = covariance(k, 4);
            odom.pose.covariance[i * 6 + 2] = covariance(k, 5);
            odom.pose.covariance[i * 6 + 3] = covariance(k, 0);
            odom.pose.covariance[i * 6 + 4] = covariance(k, 1);
            odom.pose.covariance[i * 6 + 5] = covariance(k, 2);
        }
    }
    
    return odom;
}

geometry_msgs::TransformStamped eigen2Transform(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp);

sensor_msgs::PointCloud2 pcl2msg(fastlio::PointCloudXYZI::Ptr inp, std::string &frame_id, const double &timestamp);

Eigen::Vector3d rotate2rpy(Eigen::Matrix3d &rot);

// 重定位统计信息结构体
struct RelocStatistics
{
    std::vector<Eigen::Matrix4d> poses;          // 多次匹配的位姿
    std::vector<double> fitnesses;               // 多次匹配的fitness
    Eigen::Matrix4d mean_pose;                   // 平均位姿
    Eigen::Vector3d mean_translation;             // 平均平移
    Eigen::Matrix3d mean_rotation;                // 平均旋转
    Eigen::Vector3d std_translation;             // 平移标准差
    Eigen::Vector3d std_rotation_euler;          // 旋转标准差（欧拉角）
    double mean_fitness;                          // 平均fitness
    double std_fitness;                           // fitness标准差
    double min_fitness;                           // 最小fitness
    double max_fitness;                           // 最大fitness
    Eigen::Vector3d min_translation;              // 最小平移
    Eigen::Vector3d max_translation;              // 最大平移
    Eigen::Vector3d max_dev_translation;         // 最大偏差（平移）
    Eigen::Vector3d max_dev_rotation_euler;       // 最大偏差（旋转，欧拉角）
};

// 计算旋转矩阵的平均值（通过四元数）
Eigen::Matrix3d averageRotations(const std::vector<Eigen::Matrix3d>& rotations);

// 计算统计信息
RelocStatistics calculateRelocStatistics(const std::vector<Eigen::Matrix4d>& poses, 
                                          const std::vector<double>& fitnesses);

// 打印统计报告
void printRelocStatisticsReport(const RelocStatistics& stats);