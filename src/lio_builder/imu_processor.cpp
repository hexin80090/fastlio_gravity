#include "lio_builder/imu_processor.h"
#include <cmath>
#include <ros/ros.h>

namespace fastlio
{
    IMUProcessor::IMUProcessor(std::shared_ptr<esekfom::esekf<state_ikfom, 12, input_ikfom>> kf)
        : kf_(kf), init_count_(0), last_lidar_time_end_(0.0),
          mean_acc_(Eigen::Vector3d::Zero()), mean_gyro_(Eigen::Vector3d::Zero()),
          last_acc_(Eigen::Vector3d::Zero()), last_gyro_(Eigen::Vector3d::Zero()),
          rot_ext_(Eigen::Matrix3d::Identity()), pos_ext_(Eigen::Vector3d::Zero())
    {
        Q_ = process_noise_cov();
    }

    void IMUProcessor::setExtParams(Eigen::Matrix3d &rot_ext, Eigen::Vector3d &pos_ext)
    {
        rot_ext_ = rot_ext;
        pos_ext_ = pos_ext;
    }

    void IMUProcessor::setCov(Eigen::Vector3d gyro_cov, Eigen::Vector3d acc_cov, Eigen::Vector3d gyro_bias_cov, Eigen::Vector3d acc_bias_cov)
    {
        gyro_cov_ = gyro_cov;
        acc_cov_ = acc_cov;
        gyro_bias_cov_ = gyro_bias_cov;
        acc_bias_cov_ = acc_bias_cov;
    }

    void IMUProcessor::setCov(double gyro_cov, double acc_cov, double gyro_bias_cov, double acc_bias_cov)
    {
        gyro_cov_ = Eigen::Vector3d(gyro_cov, gyro_cov, gyro_cov);
        acc_cov_ = Eigen::Vector3d(acc_cov, acc_cov, acc_cov);
        gyro_bias_cov_ = Eigen::Vector3d(gyro_bias_cov, gyro_bias_cov, gyro_bias_cov);
        acc_bias_cov_ = Eigen::Vector3d(acc_bias_cov, acc_bias_cov, acc_bias_cov);
    }

    void IMUProcessor::init(const MeasureGroup &meas)
    {
        if (meas.imus.empty())
            return;

        // 静态初始化，估计重力和角速度偏置

        for (const auto &imu : meas.imus)
        {
            init_count_++;
            mean_acc_ += (imu.acc - mean_acc_) / init_count_;
            mean_gyro_ += (imu.gyro - mean_gyro_) / init_count_;
        }
        if (init_count_ < max_init_count_)
            return;
        init_flag_ = true;

        // 设置初始化状态
        state_ikfom state = kf_->get_x();
        state.offset_R_L_I = rot_ext_;
        state.offset_T_L_I = pos_ext_;
        state.bg = mean_gyro_;
        
        // Print gravity direction info to diagnose if aircraft is level
        ROS_WARN("========== IMU Initialization Info ==========");
        ROS_WARN("Mean acceleration (mean_acc_): [%.6f, %.6f, %.6f] m/s^2", mean_acc_.x(), mean_acc_.y(), mean_acc_.z());
        ROS_WARN("Mean acceleration magnitude: %.6f m/s^2", mean_acc_.norm());
        ROS_WARN("Normalized gravity direction (-mean_acc_): [%.6f, %.6f, %.6f]", (-mean_acc_).normalized().x(), (-mean_acc_).normalized().y(), (-mean_acc_).normalized().z());
        
        // Calculate angle between gravity direction and vertical
        Eigen::Vector3d gravity_dir = (-mean_acc_).normalized();
        Eigen::Vector3d vertical_dir(0.0, 0.0, 1.0);  // Vertical upward
        double gravity_angle = std::acos(gravity_dir.dot(vertical_dir)) * 180.0 / M_PI;
        ROS_WARN("Angle between gravity and vertical: %.4f degrees", gravity_angle);
        
        // Calculate X, Y, Z axis tilt angles
        double x_tilt = std::asin(std::abs(gravity_dir.x())) * 180.0 / M_PI;
        double y_tilt = std::asin(std::abs(gravity_dir.y())) * 180.0 / M_PI;
        double z_tilt = std::acos(std::abs(gravity_dir.z())) * 180.0 / M_PI;
        ROS_WARN("X-axis tilt angle: %.4f degrees", x_tilt);
        ROS_WARN("Y-axis tilt angle: %.4f degrees", y_tilt);
        ROS_WARN("Z-axis tilt angle: %.4f degrees", z_tilt);
        ROS_WARN("==============================================");
        
        // For non-horizontal initial placement, perform gravity alignment
        // But compensate for the 15-degree rotation already applied by the driver
        if (align_gravity_)
        {
            // Calculate gravity alignment rotation (IMU frame to world frame)
            Eigen::Quaterniond gravity_align_rot = Eigen::Quaterniond::FromTwoVectors(
                (-mean_acc_).normalized(), 
                Eigen::Vector3d(0.0, 0.0, -1.0)
            );
            
            // Driver has already applied 15-degree pitch rotation (pitch = 15.0 degrees)
            // Compensate for this by rotating back by -15 degrees around Y-axis
            const double driver_pitch_deg = 15.0;
            const double driver_pitch_rad = driver_pitch_deg * M_PI / 180.0;
            Eigen::AngleAxisd driver_compensation(-driver_pitch_rad, Eigen::Vector3d::UnitY());
            Eigen::Quaterniond driver_comp_rot(driver_compensation);
            
            // Calculate extrinsic rotation angles (if needed for compensation)
            Eigen::Matrix3d ext_rot_mat = rot_ext_;
            Eigen::Vector3d ext_euler = ext_rot_mat.eulerAngles(2, 1, 0) * 180.0 / M_PI;  // ZYX order
            double ext_pitch_deg = ext_euler(1);
            double ext_roll_deg = ext_euler(2);
            double ext_yaw_deg = ext_euler(0);
            
            // Compensate for extrinsic pitch rotation (LiDAR->IMU)
            // Note: Pitch compensation (-0.448 deg) has been added to extrinsic parameters in config file
            // So we only need to compensate for the original extrinsic pitch here
            const double ext_pitch_rad = ext_pitch_deg * M_PI / 180.0;
            Eigen::AngleAxisd ext_pitch_compensation(-ext_pitch_rad, Eigen::Vector3d::UnitY());
            Eigen::Quaterniond ext_pitch_comp_rot(ext_pitch_compensation);
            
            // Apply compensation: gravity_align - driver_rotation - extrinsic_pitch
            // Note: Additional pitch compensation is now in extrinsic parameters
            state.rot = ext_pitch_comp_rot * driver_comp_rot * gravity_align_rot;
            state.grav = S2(Eigen::Vector3d(0, 0, -G_m_s2));
            
            // Print detailed rotation info
            Eigen::Matrix3d gravity_rot_mat = gravity_align_rot.toRotationMatrix();
            Eigen::Vector3d gravity_euler = gravity_rot_mat.eulerAngles(2, 1, 0) * 180.0 / M_PI;
            
            Eigen::Matrix3d final_rot_mat = state.rot.toRotationMatrix();
            Eigen::Vector3d final_euler = final_rot_mat.eulerAngles(2, 1, 0) * 180.0 / M_PI;
            
            ROS_WARN("========== Gravity Alignment with Compensation ==========");
            ROS_WARN("Original gravity alignment rotation:");
            ROS_WARN("  Euler angles: roll=%.4f deg, pitch=%.4f deg, yaw=%.4f deg", 
                     gravity_euler(2), gravity_euler(1), gravity_euler(0));
            
            ROS_WARN("Driver compensation (pitch = -%.4f deg):", driver_pitch_deg);
            Eigen::Matrix3d driver_comp_mat = driver_comp_rot.toRotationMatrix();
            Eigen::Vector3d driver_comp_euler = driver_comp_mat.eulerAngles(2, 1, 0) * 180.0 / M_PI;
            ROS_WARN("  Euler angles: roll=%.4f deg, pitch=%.4f deg, yaw=%.4f deg", 
                     driver_comp_euler(2), driver_comp_euler(1), driver_comp_euler(0));
            
            ROS_WARN("Extrinsic rotation (LiDAR->IMU):");
            ROS_WARN("  Euler angles: roll=%.4f deg, pitch=%.4f deg, yaw=%.4f deg", 
                     ext_roll_deg, ext_pitch_deg, ext_yaw_deg);
            
            ROS_WARN("Extrinsic pitch compensation (pitch = -%.4f deg):", ext_pitch_deg);
            Eigen::Matrix3d ext_pitch_comp_mat = ext_pitch_comp_rot.toRotationMatrix();
            Eigen::Vector3d ext_pitch_comp_euler = ext_pitch_comp_mat.eulerAngles(2, 1, 0) * 180.0 / M_PI;
            ROS_WARN("  Euler angles: roll=%.4f deg, pitch=%.4f deg, yaw=%.4f deg", 
                     ext_pitch_comp_euler(2), ext_pitch_comp_euler(1), ext_pitch_comp_euler(0));
            ROS_WARN("  Note: Additional pitch compensation (-0.448 deg) is now in extrinsic parameters");
            
            ROS_WARN("Final compensated rotation (state.rot):");
            ROS_WARN("  Rotation matrix:");
            ROS_WARN("    [%.6f, %.6f, %.6f]", final_rot_mat(0,0), final_rot_mat(0,1), final_rot_mat(0,2));
            ROS_WARN("    [%.6f, %.6f, %.6f]", final_rot_mat(1,0), final_rot_mat(1,1), final_rot_mat(1,2));
            ROS_WARN("    [%.6f, %.6f, %.6f]", final_rot_mat(2,0), final_rot_mat(2,1), final_rot_mat(2,2));
            ROS_WARN("  Euler angles (ZYX order): roll=%.4f deg, pitch=%.4f deg, yaw=%.4f deg", 
                     final_euler(2), final_euler(1), final_euler(0));
            
            // Calculate net compensation
            double net_pitch = final_euler(1) - gravity_euler(1);
            double net_roll = final_euler(2) - gravity_euler(2);
            double net_yaw = final_euler(0) - gravity_euler(0);
            ROS_WARN("Net compensation: roll=%.4f deg, pitch=%.4f deg, yaw=%.4f deg", 
                     net_roll, net_pitch, net_yaw);
            
            // Calculate total rotation applied to point cloud
            // Final point cloud angle = driver(15deg) + extrinsic(b deg) + gravity_align(a deg) + fixed_tilt(+0.27deg)
            // But driver compensation and fixed tilt are already applied in state.rot, so:
            // Final = extrinsic(b deg) + compensated_gravity_align(a - 15 deg + 0.27 deg)
            Eigen::Matrix3d total_rot = final_rot_mat * ext_rot_mat;
            Eigen::Vector3d total_euler = total_rot.eulerAngles(2, 1, 0) * 180.0 / M_PI;
            ROS_WARN("========== Total Point Cloud Rotation ==========");
            ROS_WARN("Total rotation (gravity_align * extrinsic):");
            ROS_WARN("  Euler angles: roll=%.4f deg, pitch=%.4f deg, yaw=%.4f deg", 
                     total_euler(2), total_euler(1), total_euler(0));
            ROS_WARN("Note: Driver 15deg is already applied to point cloud data");
            ROS_WARN("      Pitch compensation (-0.448deg) is included in extrinsic parameters");
            ROS_WARN("      Extrinsic rotation is LiDAR->IMU (physical installation + pitch compensation)");
            ROS_WARN("      Gravity alignment is IMU->World (with driver compensation)");
            ROS_WARN("================================================");
        }
        else
        {
            state.grav = S2(-mean_acc_ / mean_acc_.norm() * G_m_s2);
        }

        kf_->change_x(state);

        // 初始化噪声的协方差矩阵
        esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_->get_P();
        init_P.setIdentity();
        init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001;
        init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001;
        init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001;
        init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = 0.001;
        init_P(21, 21) = init_P(22, 22) = 0.00001;
        kf_->change_P(init_P);

        last_imu_ = meas.imus.back();
    }

    void IMUProcessor::undistortPointcloud(const MeasureGroup &meas, PointCloudXYZI::Ptr &out)
    {

        std::deque<IMU> v_imus(meas.imus.begin(), meas.imus.end());
        v_imus.push_front(last_imu_);
        const double imu_time_begin = v_imus.front().timestamp;
        const double imu_time_end = v_imus.back().timestamp;
        const double lidar_time_begin = meas.lidar_time_begin;
        const double lidar_time_end = meas.lidar_time_end;

        out = meas.lidar;

        std::sort(out->points.begin(), out->points.end(), [](PointType &p1, PointType &p2) -> bool
                  { return p1.curvature < p2.curvature; });

        state_ikfom state = kf_->get_x();
        imu_poses_.clear();
        imu_poses_.emplace_back(0.0, last_acc_, last_gyro_, state.vel, state.pos, state.rot.toRotationMatrix());

        Eigen::Vector3d acc_val, gyro_val;
        double dt = 0.0;
        Q_.setIdentity();

        input_ikfom inp;
        // 计算每一帧IMU的位姿
        for (auto it_imu = v_imus.begin(); it_imu < (v_imus.end() - 1); it_imu++)
        {
            IMU &head = *it_imu;
            IMU &tail = *(it_imu + 1);
            if (tail.timestamp < last_lidar_time_end_)
                continue;
            gyro_val = 0.5 * (head.gyro + tail.gyro);
            acc_val = 0.5 * (head.acc + head.acc);
            // normalize acc
            acc_val = acc_val * G_m_s2 / mean_acc_.norm();

            if (head.timestamp < last_lidar_time_end_)
                dt = tail.timestamp - last_lidar_time_end_;
            else
                dt = tail.timestamp - head.timestamp;

            Q_.block<3, 3>(0, 0).diagonal() = gyro_cov_;
            Q_.block<3, 3>(3, 3).diagonal() = acc_cov_;
            Q_.block<3, 3>(6, 6).diagonal() = gyro_bias_cov_;
            Q_.block<3, 3>(9, 9).diagonal() = acc_bias_cov_;
            inp.acc = acc_val;
            inp.gyro = gyro_val;
            kf_->predict(dt, Q_, inp);

            state = kf_->get_x();

            last_gyro_ = gyro_val - state.bg;
            last_acc_ = state.rot.toRotationMatrix() * (acc_val - state.ba);
            last_acc_ += state.grav.get_vect();

            double offset = tail.timestamp - lidar_time_begin;
            imu_poses_.emplace_back(offset, last_acc_, last_gyro_, state.vel, state.pos, state.rot.toRotationMatrix());
        }

        // 计算最后一个点云的位姿
        // double sign = lidar_time_end > imu_time_end ? 1.0 : -1.0;
        dt = lidar_time_end - imu_time_end;
        kf_->predict(dt, Q_, inp);

        last_imu_ = v_imus.back();
        last_lidar_time_end_ = lidar_time_end;

        state = kf_->get_x();
        state.rot;
        Eigen::Matrix3d cur_rot = state.rot.toRotationMatrix();
        Eigen::Vector3d cur_pos = state.pos;
        Eigen::Matrix3d cur_rot_ext = state.offset_R_L_I.toRotationMatrix();
        Eigen::Vector3d cur_pos_ext = state.offset_T_L_I;

        // 畸变矫正
        auto it_pcl = out->points.end() - 1;
        for (auto it_kp = imu_poses_.end() - 1; it_kp != imu_poses_.begin(); it_kp--)
        {
            auto head = it_kp - 1;
            auto tail = it_kp;

            Eigen::Matrix3d imu_rot = head->rot;
            Eigen::Vector3d imu_pos = head->pos;
            Eigen::Vector3d imu_vel = head->vel;
            Eigen::Vector3d imu_acc = tail->acc;
            Eigen::Vector3d imu_gyro = tail->gyro;

            for (; it_pcl->curvature / double(1000) > head->offset; it_pcl--)
            {
                dt = it_pcl->curvature / double(1000) - head->offset;
                Eigen::Vector3d point(it_pcl->x, it_pcl->y, it_pcl->z);
                Eigen::Matrix3d point_rot = imu_rot * Exp(imu_gyro, dt);
                Eigen::Vector3d point_pos = imu_pos + imu_vel * dt + 0.5 * imu_acc * dt * dt;
                // T_l_b * T_j_w * T_w_i * T_b_l * p
                Eigen::Vector3d p_compensate = cur_rot_ext.transpose() * (cur_rot.transpose() * (point_rot * (cur_rot_ext * point + cur_pos_ext) + point_pos - cur_pos) - cur_pos_ext);
                it_pcl->x = p_compensate(0);
                it_pcl->y = p_compensate(1);
                it_pcl->z = p_compensate(2);

                if (it_pcl == out->points.begin())
                    break;
            }
        }
    }

    bool IMUProcessor::operator()(const MeasureGroup &meas, PointCloudXYZI::Ptr &out)
    {
        if (!init_flag_)
        {
            init(meas);
            return false;
        }
        undistortPointcloud(meas, out);
        return true;
    }

    void IMUProcessor::reset()
    {
        init_count_ = 0;
        init_flag_ = false;  
        mean_acc_ = Eigen::Vector3d::Zero();
        mean_gyro_ = Eigen::Vector3d::Zero();
    }

} // namespace fastlio
