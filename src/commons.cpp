#include "commons.h"
#include <cmath>
#include <algorithm>
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fastlio
{
    bool esti_plane(Eigen::Vector4d &out, const PointVector &points, const double &thresh)
    {
        Eigen::Matrix<double, NUM_MATCH_POINTS, 3> A;
        Eigen::Matrix<double, NUM_MATCH_POINTS, 1> b;
        A.setZero();
        b.setOnes();
        b *= -1.0;
        for (int i = 0; i < NUM_MATCH_POINTS; i++)
        {
            A(i, 0) = points[i].x;
            A(i, 1) = points[i].y;
            A(i, 2) = points[i].z;
        }

        Eigen::Vector3d normvec = A.colPivHouseholderQr().solve(b);

        double norm = normvec.norm();
        out[0] = normvec(0) / norm;
        out[1] = normvec(1) / norm;
        out[2] = normvec(2) / norm;
        out[3] = 1.0 / norm;

        for (int j = 0; j < NUM_MATCH_POINTS; j++)
        {
            if (std::fabs(out(0) * points[j].x + out(1) * points[j].y + out(2) * points[j].z + out(3)) > thresh)
            {
                return false;
            }
        }
        return true;
    }

    float sq_dist(const PointType &p1, const PointType &p2)
    {
        return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    }

    MTK::get_cov<process_noise_ikfom>::type process_noise_cov()
    {
        MTK::get_cov<process_noise_ikfom>::type cov = MTK::get_cov<process_noise_ikfom>::type::Zero();
        MTK::setDiagonal<process_noise_ikfom, vect3, 0>(cov, &process_noise_ikfom::ng, 0.0001);   // 0.03
        MTK::setDiagonal<process_noise_ikfom, vect3, 3>(cov, &process_noise_ikfom::na, 0.0001);   // *dt 0.01 0.01 * dt * dt 0.05
        MTK::setDiagonal<process_noise_ikfom, vect3, 6>(cov, &process_noise_ikfom::nbg, 0.00001); // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
        MTK::setDiagonal<process_noise_ikfom, vect3, 9>(cov, &process_noise_ikfom::nba, 0.00001); // 0.001 0.05 0.0001/out 0.01
        return cov;
    }

    Eigen::Matrix<double, 24, 1> get_f(state_ikfom &s, const input_ikfom &in)
    {
        Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
        vect3 omega;
        in.gyro.boxminus(omega, s.bg);
        vect3 a_inertial = s.rot * (in.acc - s.ba);
        for (int i = 0; i < 3; i++)
        {
            res(i) = s.vel[i];
            res(i + 3) = omega[i];
            res(i + 12) = a_inertial[i] + s.grav[i];
        }
        return res;
    }

    Eigen::Matrix<double, 24, 23> df_dx(state_ikfom &s, const input_ikfom &in)
    {
        Eigen::Matrix<double, 24, 23> cov = Eigen::Matrix<double, 24, 23>::Zero();
        cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
        vect3 acc_;
        in.acc.boxminus(acc_, s.ba);
        vect3 omega;
        in.gyro.boxminus(omega, s.bg);
        cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix() * MTK::hat(acc_);
        cov.template block<3, 3>(12, 18) = -s.rot.toRotationMatrix();
        Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
        Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
        s.S2_Mx(grav_matrix, vec, 21);
        cov.template block<3, 2>(12, 21) = grav_matrix;
        cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity();
        return cov;
    }

    Eigen::Matrix<double, 24, 12> df_dw(state_ikfom &s, const input_ikfom &in)
    {
        Eigen::Matrix<double, 24, 12> cov = Eigen::Matrix<double, 24, 12>::Zero();
        cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix();
        cov.template block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
        cov.template block<3, 3>(15, 6) = Eigen::Matrix3d::Identity();
        cov.template block<3, 3>(18, 9) = Eigen::Matrix3d::Identity();
        return cov;
    }

} // namespace name

void ImuData::callback(const sensor_msgs::Imu::ConstPtr &msg)
{
    std::lock_guard<std::mutex> lock(mutex);
    double timestamp = msg->header.stamp.toSec();
    if (timestamp < last_timestamp)
    {
        ROS_WARN("imu loop back, clear buffer, last_timestamp: %f  current_timestamp: %f", last_timestamp, timestamp);
        buffer.clear();
    }
    last_timestamp = timestamp;
    buffer.emplace_back(timestamp,
                        msg->linear_acceleration.x,
                        msg->linear_acceleration.y,
                        msg->linear_acceleration.z,
                        msg->angular_velocity.x,
                        msg->angular_velocity.y,
                        msg->angular_velocity.z);
}

void LivoxData::callback(const livox_ros_driver2::CustomMsg::ConstPtr &msg)
{
    std::lock_guard<std::mutex> lock(mutex);
    double timestamp = msg->header.stamp.toSec();
    if (timestamp < last_timestamp)
    {
        ROS_WARN("livox loop back, clear buffer, last_timestamp: %f  current_timestamp: %f", last_timestamp, timestamp);
        buffer.clear();
        time_buffer.clear();
    }
    last_timestamp = timestamp;
    fastlio::PointCloudXYZI::Ptr ptr(new fastlio::PointCloudXYZI());
    livox2pcl(msg, ptr);
    // ROS_INFO("size: %lu",ptr->size());
    buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp);
}

void LivoxData::livox2pcl(const livox_ros_driver2::CustomMsg::ConstPtr &msg, fastlio::PointCloudXYZI::Ptr &out)
{
    int point_num = msg->point_num;
    out->clear();
    out->reserve(point_num / filter_num + 1);
    uint valid_num = 0;
    for (uint i = 0; i < point_num; i++)
    {
        if ((msg->points[i].line < 4) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
        {
            valid_num++;
            if (valid_num % filter_num != 0)
                continue;
            fastlio::PointType p;
            p.x = msg->points[i].x;
            p.y = msg->points[i].y;
            p.z = msg->points[i].z;
            p.intensity = msg->points[i].reflectivity;
            p.curvature = msg->points[i].offset_time / float(1000000); // 纳秒->毫秒
            if ((p.x * p.x + p.y * p.y + p.z * p.z > (blind * blind)))
            {
                out->push_back(p);
            }
        }
    }
}

bool MeasureGroup::syncPackage(ImuData &imu_data, LivoxData &livox_data)
{
    if (imu_data.buffer.empty() || livox_data.buffer.empty())
        return false;

    if (!lidar_pushed)
    {
        lidar = livox_data.buffer.front();
        lidar_time_begin = livox_data.time_buffer.front();
        lidar_time_end = lidar_time_begin + lidar->points.back().curvature / double(1000);
        lidar_pushed = true;
    }

    if (imu_data.last_timestamp < lidar_time_end)
        return false;
    double imu_time = imu_data.buffer.front().timestamp;
    imus.clear();
    while (!imu_data.buffer.empty() && (imu_time < lidar_time_end))
    {
        imu_time = imu_data.buffer.front().timestamp;
        if (imu_time > lidar_time_end)
            break;
        imus.push_back(imu_data.buffer.front());
        imu_data.buffer.pop_front();
    }
    livox_data.buffer.pop_front();
    livox_data.time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

sensor_msgs::PointCloud2 pcl2msg(fastlio::PointCloudXYZI::Ptr inp, std::string &frame_id, const double &timestamp)
{
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(*inp, msg);
    if (timestamp < 0)
        msg.header.stamp = ros::Time().now();
    else
        msg.header.stamp = ros::Time().fromSec(timestamp);
    msg.header.frame_id = frame_id;
    return msg;
}

nav_msgs::Odometry eigen2Odometry(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp)
{
    nav_msgs::Odometry odom;
    odom.header.frame_id = frame_id;
    odom.header.stamp = ros::Time().fromSec(timestamp);
    odom.child_frame_id = child_frame_id;
    Eigen::Quaterniond q = Eigen::Quaterniond(rot);
    odom.pose.pose.position.x = pos(0);
    odom.pose.pose.position.y = pos(1);
    odom.pose.pose.position.z = pos(2);

    odom.pose.pose.orientation.w = q.w();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    return odom;
}

geometry_msgs::TransformStamped eigen2Transform(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp)
{
    geometry_msgs::TransformStamped transform;
    transform.header.frame_id = frame_id;
    transform.header.stamp = ros::Time().fromSec(timestamp);
    transform.child_frame_id = child_frame_id;
    transform.transform.translation.x = pos(0);
    transform.transform.translation.y = pos(1);
    transform.transform.translation.z = pos(2);
    Eigen::Quaterniond q = Eigen::Quaterniond(rot);
    // std::cout << rot << std::endl;
    // std::cout << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << std::endl;
    transform.transform.rotation.w = q.w();
    transform.transform.rotation.x = q.x();
    transform.transform.rotation.y = q.y();
    transform.transform.rotation.z = q.z();
    return transform;
}

Eigen::Vector3d rotate2rpy(Eigen::Matrix3d &rot)
{
    double roll = std::atan2(rot(2, 1), rot(2, 2));
    double pitch = asin(-rot(2, 0));
    double yaw = std::atan2(rot(1, 0), rot(0, 0));
    return Eigen::Vector3d(roll, pitch, yaw);
}

Eigen::Matrix3d averageRotations(const std::vector<Eigen::Matrix3d>& rotations)
{
    if (rotations.empty())
        return Eigen::Matrix3d::Identity();
    
    // 转换为四元数后平均
    std::vector<Eigen::Quaterniond> quaternions;
    for (const auto& rot : rotations)
    {
        Eigen::Quaterniond q(rot);
        // 确保四元数在同一个半球（避免 q 和 -q 的问题）
        if (quaternions.size() > 0 && quaternions[0].dot(q) < 0)
            q.coeffs() = -q.coeffs();
        quaternions.push_back(q);
    }
    
    // 计算平均四元数（简单平均，然后归一化）
    Eigen::Vector4d mean_coeffs = Eigen::Vector4d::Zero();
    for (const auto& q : quaternions)
    {
        mean_coeffs += q.coeffs();
    }
    mean_coeffs /= quaternions.size();
    
    Eigen::Quaterniond mean_quat(mean_coeffs);
    mean_quat.normalize();
    
    return mean_quat.toRotationMatrix();
}

RelocStatistics calculateRelocStatistics(const std::vector<Eigen::Matrix4d>& poses, 
                                          const std::vector<double>& fitnesses)
{
    RelocStatistics stats;
    stats.poses = poses;
    stats.fitnesses = fitnesses;
    
    if (poses.empty() || fitnesses.empty())
        return stats;
    
    // 提取平移和旋转
    std::vector<Eigen::Vector3d> translations;
    std::vector<Eigen::Matrix3d> rotations;
    
    for (const auto& pose : poses)
    {
        translations.push_back(pose.block<3, 1>(0, 3));
        rotations.push_back(pose.block<3, 3>(0, 0));
    }
    
    // 计算平均平移
    stats.mean_translation = Eigen::Vector3d::Zero();
    for (const auto& t : translations)
        stats.mean_translation += t;
    stats.mean_translation /= translations.size();
    
    // 计算平均旋转
    stats.mean_rotation = averageRotations(rotations);
    
    // 计算平均位姿
    stats.mean_pose.setIdentity();
    stats.mean_pose.block<3, 3>(0, 0) = stats.mean_rotation;
    stats.mean_pose.block<3, 1>(0, 3) = stats.mean_translation;
    
    // 计算平移标准差
    Eigen::Vector3d translation_variance = Eigen::Vector3d::Zero();
    for (const auto& t : translations)
    {
        Eigen::Vector3d diff = t - stats.mean_translation;
        translation_variance += diff.cwiseProduct(diff);
    }
    translation_variance /= translations.size();
    stats.std_translation = translation_variance.cwiseSqrt();
    
    // 计算最大偏差（平移）
    stats.max_dev_translation = Eigen::Vector3d::Zero();
    for (const auto& t : translations)
    {
        Eigen::Vector3d diff = (t - stats.mean_translation).cwiseAbs();
        stats.max_dev_translation = stats.max_dev_translation.cwiseMax(diff);
    }
    
    // 计算最小/最大平移
    stats.min_translation = translations[0];
    stats.max_translation = translations[0];
    for (const auto& t : translations)
    {
        stats.min_translation = stats.min_translation.cwiseMin(t);
        stats.max_translation = stats.max_translation.cwiseMax(t);
    }
    
    // 计算旋转统计（转换为欧拉角）
    std::vector<Eigen::Vector3d> eulers;
    for (const auto& rot : rotations)
    {
        Eigen::Matrix3d rot_copy = rot;
        eulers.push_back(rotate2rpy(rot_copy));
    }
    
    Eigen::Vector3d mean_euler = Eigen::Vector3d::Zero();
    for (const auto& e : eulers)
        mean_euler += e;
    mean_euler /= eulers.size();
    
    Eigen::Vector3d euler_variance = Eigen::Vector3d::Zero();
    for (const auto& e : eulers)
    {
        Eigen::Vector3d diff = e - mean_euler;
        euler_variance += diff.cwiseProduct(diff);
    }
    euler_variance /= eulers.size();
    stats.std_rotation_euler = euler_variance.cwiseSqrt();
    
    // 计算最大偏差（旋转）
    stats.max_dev_rotation_euler = Eigen::Vector3d::Zero();
    for (const auto& e : eulers)
    {
        Eigen::Vector3d diff = (e - mean_euler).cwiseAbs();
        stats.max_dev_rotation_euler = stats.max_dev_rotation_euler.cwiseMax(diff);
    }
    
    // 计算fitness统计
    stats.mean_fitness = 0.0;
    for (double f : fitnesses)
        stats.mean_fitness += f;
    stats.mean_fitness /= fitnesses.size();
    
    double fitness_variance = 0.0;
    for (double f : fitnesses)
    {
        double diff = f - stats.mean_fitness;
        fitness_variance += diff * diff;
    }
    fitness_variance /= fitnesses.size();
    stats.std_fitness = std::sqrt(fitness_variance);
    
    stats.min_fitness = *std::min_element(fitnesses.begin(), fitnesses.end());
    stats.max_fitness = *std::max_element(fitnesses.begin(), fitnesses.end());
    
    return stats;
}

void printRelocStatisticsReport(const RelocStatistics& stats)
{
    if (stats.poses.empty())
    {
        ROS_WARN("No statistics to report (empty poses)");
        return;
    }
    
    std::string separator(60, '=');
    ROS_INFO("\n%s", separator.c_str());
    ROS_INFO("Relocalization Consistency Analysis Report");
    ROS_INFO("%s", separator.c_str());
    
    // 位置统计
    ROS_INFO("\n--- Position Statistics (m) ---");
    ROS_INFO("Mean:    [%.6f, %.6f, %.6f]", 
             stats.mean_translation(0), stats.mean_translation(1), stats.mean_translation(2));
    ROS_INFO("Std:     [%.6f, %.6f, %.6f]", 
             stats.std_translation(0), stats.std_translation(1), stats.std_translation(2));
    ROS_INFO("Max Dev: [%.6f, %.6f, %.6f]", 
             stats.max_dev_translation(0), stats.max_dev_translation(1), stats.max_dev_translation(2));
    ROS_INFO("Range:   [%.6f ~ %.6f, %.6f ~ %.6f, %.6f ~ %.6f]",
             stats.min_translation(0), stats.max_translation(0),
             stats.min_translation(1), stats.max_translation(1),
             stats.min_translation(2), stats.max_translation(2));
    
    // 旋转统计（转换为度）
    Eigen::Matrix3d mean_rot_copy = stats.mean_rotation;  // 创建副本，因为rotate2rpy需要非const引用
    Eigen::Vector3d mean_euler = rotate2rpy(mean_rot_copy);
    Eigen::Vector3d mean_euler_deg = mean_euler * 180.0 / M_PI;
    Eigen::Vector3d std_euler_deg = stats.std_rotation_euler * 180.0 / M_PI;
    Eigen::Vector3d max_dev_euler_deg = stats.max_dev_rotation_euler * 180.0 / M_PI;
    
    ROS_INFO("\n--- Rotation Statistics (deg) ---");
    ROS_INFO("Mean Euler:    [%.4f, %.4f, %.4f]", 
             mean_euler_deg(0), mean_euler_deg(1), mean_euler_deg(2));
    ROS_INFO("Std Euler:     [%.4f, %.4f, %.4f]", 
             std_euler_deg(0), std_euler_deg(1), std_euler_deg(2));
    ROS_INFO("Max Dev Euler: [%.4f, %.4f, %.4f]", 
             max_dev_euler_deg(0), max_dev_euler_deg(1), max_dev_euler_deg(2));
    
    // Fitness统计
    ROS_INFO("\n--- Fitness Statistics ---");
    ROS_INFO("Mean: %.6f", stats.mean_fitness);
    ROS_INFO("Std:  %.6f", stats.std_fitness);
    ROS_INFO("Min:  %.6f", stats.min_fitness);
    ROS_INFO("Max:  %.6f", stats.max_fitness);
    
    // 最终平均位姿
    Eigen::Quaterniond mean_quat(stats.mean_rotation);
    Eigen::Matrix3d final_rot_copy = stats.mean_rotation;  // 创建副本
    Eigen::Vector3d final_euler = rotate2rpy(final_rot_copy);
    Eigen::Vector3d final_euler_deg = final_euler * 180.0 / M_PI;
    
    ROS_INFO("\n--- Final Offset (Mean) ---");
    ROS_INFO("Translation: [%.6f, %.6f, %.6f]", 
             stats.mean_translation(0), stats.mean_translation(1), stats.mean_translation(2));
    ROS_INFO("Quaternion:  [%.6f, %.6f, %.6f, %.6f]", 
             mean_quat.x(), mean_quat.y(), mean_quat.z(), mean_quat.w());
    ROS_INFO("Euler (RPY): [%.4f, %.4f, %.4f] deg", 
             final_euler_deg(0), final_euler_deg(1), final_euler_deg(2));
    ROS_INFO("%s\n", separator.c_str());
}