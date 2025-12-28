#include <thread>
#include <csignal>
#include <fstream>
#include <ros/ros.h>
#include <ros/package.h>
#include "commons.h"
#include "localizer/icp_localizer.h"
#include "lio_builder/lio_builder.h"
#include <tf2_ros/transform_broadcaster.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>

#include "fastlio/SlamReLoc.h"
#include "fastlio/MapConvert.h"
#include "fastlio/SlamHold.h"
#include "fastlio/SlamStart.h"
#include "fastlio/SlamRelocCheck.h"

bool terminate_flag = false;

void signalHandler(int signum)
{
    std::cout << "SHUTTING DOWN LOCALIZER NODE!" << std::endl;
    terminate_flag = true;
}

struct SharedData
{
    std::mutex service_mutex;
    std::mutex main_mutex;
    bool pose_updated = false;
    bool localizer_activate = false;
    bool service_called = false;
    bool service_success = false;

    std::string map_path;
    Eigen::Matrix3d offset_rot = Eigen::Matrix3d::Identity();
    Eigen::Vector3d offset_pos = Eigen::Vector3d::Zero();
    Eigen::Matrix3d local_rot;
    Eigen::Vector3d local_pos;
    Eigen::Matrix4d initial_guess;
    fastlio::PointCloudXYZI::Ptr cloud;

    bool reset_flag = false;
    bool halt_flag = false;
};

class LocalizerThread
{
public:
    LocalizerThread() : verification_mode_(false), verification_count_(20), 
                       verification_rate_(1.0), fitness_threshold_(0.95), 
                       offset_calculated_(false) {}

    void setSharedDate(std::shared_ptr<SharedData> shared_data)
    {
        shared_data_ = shared_data;
    }

    void setRate(double rate)
    {
        rate_ = std::make_shared<ros::Rate>(rate);
    }
    void setRate(std::shared_ptr<ros::Rate> rate)
    {
        rate_ = rate;
    }
    void setLocalizer(std::shared_ptr<fastlio::IcpLocalizer> localizer)
    {
        icp_localizer_ = localizer;
    }
    
    void setVerificationParams(bool verification_mode, int verification_count, 
                               double verification_rate, double fitness_threshold)
    {
        verification_mode_ = verification_mode;
        verification_count_ = verification_count;
        verification_rate_ = verification_rate;
        fitness_threshold_ = fitness_threshold;
    }
    
    bool calculateOffsetVerificationMode(const Eigen::Matrix4d& initial_guess)
    {
        ROS_INFO("=== Starting Offset Verification Mode (%d iterations) ===", verification_count_);
        
        std::vector<Eigen::Matrix4d> poses;
        std::vector<double> fitnesses;
        
        ros::Rate verification_rate(verification_rate_);
        
        for (int i = 0; i < verification_count_; i++)
        {
            ROS_INFO("Iteration %d/%d...", i + 1, verification_count_);
            
            // 等待新的点云数据
            while (ros::ok() && !shared_data_->pose_updated)
            {
                verification_rate.sleep();
                ros::spinOnce();
            }
            
            if (terminate_flag)
                break;
            
            // 获取当前点云和位姿
            Eigen::Matrix3d local_rot;
            Eigen::Vector3d local_pos;
            pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZI>);
            {
                std::lock_guard<std::mutex> lock(shared_data_->main_mutex);
                shared_data_->pose_updated = false;
                local_rot = shared_data_->local_rot;
                local_pos = shared_data_->local_pos;
                pcl::copyPointCloud(*shared_data_->cloud, *current_cloud);
            }
            
            // 计算初始猜测（使用当前的offset）
            Eigen::Matrix4d init_guess;
            init_guess.setIdentity();
            init_guess.block<3, 3>(0, 0) = shared_data_->offset_rot * local_rot;
            init_guess.block<3, 1>(0, 3) = shared_data_->offset_rot * local_pos + shared_data_->offset_pos;
            
            // 执行ICP配准
            Eigen::Matrix4d gloabl_pose = icp_localizer_->multi_align_sync(current_cloud, init_guess);
            
            if (icp_localizer_->isSuccess())
            {
                double fitness = icp_localizer_->getFitnessScore();
                poses.push_back(gloabl_pose);
                fitnesses.push_back(fitness);
                ROS_INFO("  Fitness: %.6f", fitness);
            }
            else
            {
                ROS_WARN("  ICP alignment failed in iteration %d", i + 1);
            }
            
            verification_rate.sleep();
        }
        
        if (poses.empty())
        {
            ROS_ERROR("No successful alignments in verification mode!");
            return false;
        }
        
        // 计算统计信息
        RelocStatistics stats = calculateRelocStatistics(poses, fitnesses);
        
        // 检查平均fitness是否满足阈值
        // 注意：PCL的ICP getFitnessScore()返回的是误差值（越小越好），不是匹配度
        // 所以应该检查是否小于阈值，而不是大于
        // 使用localizer的thresh作为参考，fitness应该小于thresh才认为成功
        // 这里使用fitness_threshold_作为最大允许误差（应该设置为较小的值，如0.1）
        if (stats.mean_fitness > fitness_threshold_)
        {
            ROS_ERROR("Mean fitness (error) %.6f above threshold %.6f", stats.mean_fitness, fitness_threshold_);
            ROS_ERROR("Offset calculation failed!");
            return false;
        }
        
        // 打印统计报告
        printRelocStatisticsReport(stats);
        
        // 使用平均位姿计算offset
        {
            std::lock_guard<std::mutex> lock(shared_data_->main_mutex);
            // 获取最后一次的local_rot和local_pos（用于计算offset）
            Eigen::Matrix3d final_local_rot = shared_data_->local_rot;
            Eigen::Vector3d final_local_pos = shared_data_->local_pos;
            
            // 使用平均位姿计算offset
            shared_data_->offset_rot = stats.mean_rotation * final_local_rot.transpose();
            shared_data_->offset_pos = -stats.mean_rotation * final_local_rot.transpose() * final_local_pos + stats.mean_translation;
        }
        
        ROS_INFO("Offset calculation completed successfully using mean pose!");
        return true;
    }

    void operator()()
    {
        current_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>);

        while (ros::ok())
        {
            rate_->sleep();
            if (terminate_flag)
                break;
            if (shared_data_->halt_flag)
                continue;
            if (!shared_data_->localizer_activate)
                continue;
            
            // 如果已经计算过offset（验证模式），不再更新
            if (offset_calculated_)
            {
                continue;  // 保持线程运行，但不再更新offset
            }
            
            // 只在 service_called 时执行验证模式或单次计算
            if (shared_data_->service_called)
            {
                std::lock_guard<std::mutex> lock(shared_data_->service_mutex);
                shared_data_->service_called = false;
                icp_localizer_->init(shared_data_->map_path, false);
                
                bool success = false;
                if (verification_mode_)
                {
                    // 验证模式：20次匹配取均值
                    success = calculateOffsetVerificationMode(shared_data_->initial_guess);
                    if (success)
                    {
                        offset_calculated_ = true;
                        shared_data_->localizer_activate = true;
                        shared_data_->service_success = true;
                    }
                    else
                    {
                        shared_data_->localizer_activate = false;
                        shared_data_->service_success = false;
                    }
                }
                else
                {
                    // 正常模式：单次计算
                    while (ros::ok() && !shared_data_->pose_updated)
                    {
                        rate_->sleep();
                        ros::spinOnce();
                    }
                    
                    if (terminate_flag)
                        break;
                    
                    Eigen::Matrix3d local_rot;
                    Eigen::Vector3d local_pos;
                    {
                        std::lock_guard<std::mutex> lock_main(shared_data_->main_mutex);
                        shared_data_->pose_updated = false;
                        local_rot = shared_data_->local_rot;
                        local_pos = shared_data_->local_pos;
                        pcl::copyPointCloud(*shared_data_->cloud, *current_cloud_);
                    }
                    
                    Eigen::Matrix4d gloabl_pose = icp_localizer_->multi_align_sync(current_cloud_, shared_data_->initial_guess);
                    
                    // 注意：PCL的fitness是误差值（越小越好），应该检查是否小于阈值
                    if (icp_localizer_->isSuccess() && icp_localizer_->getFitnessScore() <= fitness_threshold_)
                    {
                        {
                            std::lock_guard<std::mutex> lock_main(shared_data_->main_mutex);
                            shared_data_->offset_rot = gloabl_pose.block<3, 3>(0, 0) * local_rot.transpose();
                            shared_data_->offset_pos = -gloabl_pose.block<3, 3>(0, 0) * local_rot.transpose() * local_pos + gloabl_pose.block<3, 1>(0, 3);
                        }
                        offset_calculated_ = true;
                        shared_data_->localizer_activate = true;
                        shared_data_->service_success = true;
                        success = true;
                    }
                    else
                    {
                        shared_data_->localizer_activate = false;
                        shared_data_->service_success = false;
                        success = false;
                    }
                }
            }
        }
    }

private:
    std::shared_ptr<SharedData> shared_data_;
    std::shared_ptr<fastlio::IcpLocalizer> icp_localizer_;
    std::shared_ptr<ros::Rate> rate_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud_;
    Eigen::Matrix4d gloabl_pose_;
    Eigen::Matrix3d local_rot_;
    Eigen::Vector3d local_pos_;
    
    // 验证模式参数
    bool verification_mode_;
    int verification_count_;
    double verification_rate_;
    double fitness_threshold_;
    bool offset_calculated_;  // 标志：offset是否已计算（固定后不再更新）
};

class LocalizerROS
{
public:
    LocalizerROS(tf2_ros::TransformBroadcaster &br, std::shared_ptr<SharedData> shared_data) : shared_date_(shared_data), br_(br)
    {
        initParams();
        initSubscribers();
        initPublishers();
        initServices();
        lio_builder_ = std::make_shared<fastlio::LIOBuilder>(lio_params_);
        icp_localizer_ = std::make_shared<fastlio::IcpLocalizer>(localizer_params_.refine_resolution,
                                                                 localizer_params_.rough_resolution,
                                                                 localizer_params_.refine_iter,
                                                                 localizer_params_.rough_iter,
                                                                 localizer_params_.thresh);
        icp_localizer_->setSearchParams(localizer_params_.xy_offset, localizer_params_.yaw_offset, localizer_params_.yaw_resolution);
        localizer_loop_.setRate(loop_rate_);
        localizer_loop_.setSharedDate(shared_data);
        localizer_loop_.setLocalizer(icp_localizer_);
        localizer_thread_ = std::make_shared<std::thread>(std::ref(localizer_loop_));
        
        // 自动加载点云地图
        if (auto_load_enable_ && !auto_load_map_path_.empty())
        {
            std::string map_path = auto_load_map_path_;
            // 如果路径不是绝对路径，则相对于包目录
            if (map_path[0] != '/')
            {
                std::string package_path = ros::package::getPath("fastlio");
                if (!package_path.empty())
                {
                    map_path = package_path + "/" + map_path;
                }
            }
            
            // 检查文件是否存在
            std::ifstream file_check(map_path);
            if (file_check.good())
            {
                file_check.close();
                ROS_INFO("Auto-loading map from: %s", map_path.c_str());
                
                // 设置地图路径并初始化定位器
                {
                    std::lock_guard<std::mutex> lock(shared_data->service_mutex);
                    shared_data->map_path = map_path;
                    shared_data->localizer_activate = true;
                    
                    // 如果启用自动重定位，设置初始位姿
                    if (auto_load_auto_reloc_)
                    {
                        Eigen::AngleAxisf rollAngle(auto_load_initial_roll_, Eigen::Vector3f::UnitX());
                        Eigen::AngleAxisf pitchAngle(auto_load_initial_pitch_, Eigen::Vector3f::UnitY());
                        Eigen::AngleAxisf yawAngle(auto_load_initial_yaw_, Eigen::Vector3f::UnitZ());
                        Eigen::Quaternionf q = rollAngle * pitchAngle * yawAngle;
                        shared_data->initial_guess.block<3, 3>(0, 0) = q.toRotationMatrix().cast<double>();
                        shared_data->initial_guess.block<3, 1>(0, 3) = Eigen::Vector3d(auto_load_initial_x_, auto_load_initial_y_, auto_load_initial_z_);
                        shared_data->service_called = true;
                        ROS_INFO("Auto-relocalization enabled with initial pose: x=%.2f, y=%.2f, z=%.2f, roll=%.2f, pitch=%.2f, yaw=%.2f",
                                 auto_load_initial_x_, auto_load_initial_y_, auto_load_initial_z_,
                                 auto_load_initial_roll_, auto_load_initial_pitch_, auto_load_initial_yaw_);
                    }
                    else
                    {
                        // 只加载地图，不进行重定位
                        icp_localizer_->init(map_path, false);
                        ROS_INFO("Map loaded successfully (without auto-relocalization)");
                    }
                }
            }
            else
            {
                ROS_WARN("Map file not found at: %s. Please check the path or disable auto_load.", map_path.c_str());
            }
        }
    }

    void initParams()
    {
        nh_.param<std::string>("map_frame", global_frame_, "map");
        nh_.param<std::string>("local_frame", local_frame_, "local");
        nh_.param<std::string>("body_frame", body_frame_, "body");
        nh_.param<std::string>("imu_topic", imu_data_.topic, "/livox/imu");
        nh_.param<std::string>("livox_topic", livox_data_.topic, "/livox/lidar");
        nh_.param<bool>("publish_map_cloud", publish_map_cloud_, false);
        double local_rate, loop_rate;
        nh_.param<double>("local_rate", local_rate, 20.0);
        nh_.param<double>("loop_rate", loop_rate, 1.0);
        local_rate_ = std::make_shared<ros::Rate>(local_rate);
        loop_rate_ = std::make_shared<ros::Rate>(loop_rate);

        nh_.param<double>("lio_builder/det_range", lio_params_.det_range, 100.0);
        nh_.param<double>("lio_builder/cube_len", lio_params_.cube_len, 500.0);
        nh_.param<double>("lio_builder/resolution", lio_params_.resolution, 0.1);
        nh_.param<double>("lio_builder/move_thresh", lio_params_.move_thresh, 1.5);
        nh_.param<bool>("lio_builder/align_gravity", lio_params_.align_gravity, true);
        nh_.param<std::vector<double>>("lio_builder/imu_ext_rot", lio_params_.imu_ext_rot, std::vector<double>());
        nh_.param<std::vector<double>>("lio_builder/imu_ext_pos", lio_params_.imu_ext_pos, std::vector<double>());
        
        nh_.param<double>("localizer/refine_resolution", localizer_params_.refine_resolution, 0.2);
        nh_.param<double>("localizer/rough_resolution", localizer_params_.rough_resolution, 0.5);
        nh_.param<double>("localizer/refine_iter", localizer_params_.refine_iter, 5);
        nh_.param<double>("localizer/rough_iter", localizer_params_.rough_iter, 10);
        nh_.param<double>("localizer/thresh", localizer_params_.thresh, 0.15);

        nh_.param<double>("localizer/xy_offset", localizer_params_.xy_offset, 2.0);
        nh_.param<double>("localizer/yaw_resolution", localizer_params_.yaw_resolution, 0.5);
        nh_.param<int>("localizer/yaw_offset", localizer_params_.yaw_offset, 1);
        
        // 验证模式参数
        bool verification_mode;
        int verification_count;
        double verification_rate;
        double fitness_threshold;
        nh_.param<bool>("localizer/verification_mode", verification_mode, true);
        nh_.param<int>("localizer/verification_count", verification_count, 20);
        nh_.param<double>("localizer/verification_rate", verification_rate, 1.0);
        nh_.param<double>("localizer/fitness_threshold", fitness_threshold, 0.95);
        localizer_loop_.setVerificationParams(verification_mode, verification_count, verification_rate, fitness_threshold);

        // 自动加载点云地图配置
        nh_.param<bool>("auto_load/enable", auto_load_enable_, false);
        nh_.param<std::string>("auto_load/map_path", auto_load_map_path_, "");
        nh_.param<bool>("auto_load/auto_reloc", auto_load_auto_reloc_, false);
        nh_.param<double>("auto_load/initial_pose/x", auto_load_initial_x_, 0.0);
        nh_.param<double>("auto_load/initial_pose/y", auto_load_initial_y_, 0.0);
        nh_.param<double>("auto_load/initial_pose/z", auto_load_initial_z_, 0.0);
        nh_.param<double>("auto_load/initial_pose/roll", auto_load_initial_roll_, 0.0);
        nh_.param<double>("auto_load/initial_pose/pitch", auto_load_initial_pitch_, 0.0);
        nh_.param<double>("auto_load/initial_pose/yaw", auto_load_initial_yaw_, 0.0);
    }

    void initSubscribers()
    {
        imu_sub_ = nh_.subscribe(imu_data_.topic, 1000, &ImuData::callback, &imu_data_);
        livox_sub_ = nh_.subscribe(livox_data_.topic, 1000, &LivoxData::callback, &livox_data_);
    }

    void initPublishers()
    {
        local_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("local_cloud", 1000);
        body_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("body_cloud", 1000);
        odom_pub_ = nh_.advertise<nav_msgs::Odometry>("slam_odom", 1000);
        // 定位模式：发布到 /mavros/local_position/odom（格式与 faster-lio/fast_lio_localization 一致）
        odom_mavros_pub_ = nh_.advertise<nav_msgs::Odometry>("/mavros/local_position/odom", 100000);
        map_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("map_cloud", 1000);
    }

    bool relocCallback(fastlio::SlamReLoc::Request &req, fastlio::SlamReLoc::Response &res)
    {
        std::string map_path = req.pcd_path;
        float x = req.x;
        float y = req.y;
        float z = req.z;
        float roll = req.roll;
        float pitch = req.pitch;
        float yaw = req.yaw;
        Eigen::AngleAxisf rollAngle(roll, Eigen::Vector3f::UnitX());
        Eigen::AngleAxisf pitchAngle(pitch, Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf yawAngle(yaw, Eigen::Vector3f::UnitZ());
        Eigen::Quaternionf q = rollAngle * pitchAngle * yawAngle;
        {
            std::lock_guard<std::mutex> lock(shared_date_->service_mutex);
            shared_date_->halt_flag = false;
            shared_date_->service_called = true;
            shared_date_->localizer_activate = true;
            shared_date_->map_path = map_path;
            shared_date_->initial_guess.block<3, 3>(0, 0) = q.toRotationMatrix().cast<double>();
            shared_date_->initial_guess.block<3, 1>(0, 3) = Eigen::Vector3d(x, y, z);
        }
        res.status = 1;
        res.message = "RELOCALIZE CALLED!";

        return true;
    }

    bool mapConvertCallback(fastlio::MapConvert::Request &req, fastlio::MapConvert::Response &res)
    {
        pcl::PCDReader reader;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        reader.read(req.map_path, *cloud);
        pcl::VoxelGrid<pcl::PointXYZI> down_sample_filter;
        down_sample_filter.setLeafSize(req.resolution, req.resolution, req.resolution);
        down_sample_filter.setInputCloud(cloud);
        down_sample_filter.filter(*cloud);

        fastlio::PointCloudXYZI::Ptr cloud_with_norm = fastlio::IcpLocalizer::addNorm(cloud);
        pcl::PCDWriter writer;
        writer.writeBinaryCompressed(req.save_path, *cloud_with_norm);
        res.message = "CONVERT SUCCESS!";
        res.status = 1;

        return true;
    }

    bool slamHoldCallback(fastlio::SlamHold::Request &req, fastlio::SlamHold::Response &res)
    {
        shared_date_->service_mutex.lock();
        shared_date_->halt_flag = true;
        shared_date_->reset_flag = true;
        shared_date_->service_mutex.unlock();
        res.message = "SLAM HALT!";
        res.status = 1;
        return true;
    }

    bool slamStartCallback(fastlio::SlamStart::Request &req, fastlio::SlamStart::Response &res)
    {
        shared_date_->service_mutex.lock();
        shared_date_->halt_flag = false;
        shared_date_->service_mutex.unlock();
        res.message = "SLAM START!";
        res.status = 1;
        return true;
    }

    bool slamRelocCheckCallback(fastlio::SlamRelocCheck::Request &req, fastlio::SlamRelocCheck::Response &res)
    {
        res.status = shared_date_->service_success;
        return true;
    }

    void initServices()
    {
        reloc_server_ = nh_.advertiseService("slam_reloc", &LocalizerROS::relocCallback, this);
        map_convert_server_ = nh_.advertiseService("map_convert", &LocalizerROS::mapConvertCallback, this);
        hold_server_ = nh_.advertiseService("slam_hold", &LocalizerROS::slamHoldCallback, this);
        start_server_ = nh_.advertiseService("slam_start", &LocalizerROS::slamStartCallback, this);
        reloc_check_server_ = nh_.advertiseService("slam_reloc_check", &LocalizerROS::slamRelocCheckCallback, this);
    }

    void publishCloud(ros::Publisher &publisher, const sensor_msgs::PointCloud2 &cloud_to_pub)
    {
        if (publisher.getNumSubscribers() == 0)
            return;
        publisher.publish(cloud_to_pub);
    }

    void publishOdom(const nav_msgs::Odometry &odom_to_pub)
    {
        if (odom_pub_.getNumSubscribers() > 0)
            odom_pub_.publish(odom_to_pub);
        // 定位模式：发布到 /mavros/local_position/odom（格式与 faster-lio/fast_lio_localization 一致）
        odom_mavros_pub_.publish(odom_to_pub);
    }

    void systemReset()
    {
        offset_rot_ = Eigen::Matrix3d::Identity();
        offset_pos_ = Eigen::Vector3d::Zero();
        {
            std::lock_guard<std::mutex> lock(shared_date_->main_mutex);
            shared_date_->offset_rot = Eigen::Matrix3d::Identity();
            shared_date_->offset_pos = Eigen::Vector3d::Zero();
            shared_date_->service_success = false;
        }
        lio_builder_->reset();
    }

    void run()
    {
        while (ros::ok())
        {
            local_rate_->sleep();
            ros::spinOnce();
            if (terminate_flag)
                break;
            if (!measure_group_.syncPackage(imu_data_, livox_data_))
                continue;
            if (shared_date_->halt_flag)
                continue;

            if (shared_date_->reset_flag)
            {
                // ROS_INFO("SLAM RESET!");
                systemReset();
                shared_date_->service_mutex.lock();
                shared_date_->reset_flag = false;
                shared_date_->service_mutex.unlock();
            }

            lio_builder_->mapping(measure_group_);
            if (lio_builder_->currentStatus() == fastlio::Status::INITIALIZE)
                continue;
            current_time_ = measure_group_.lidar_time_end;
            current_state_ = lio_builder_->currentState();
            current_cloud_body_ = lio_builder_->cloudUndistortedBody();
            {
                std::lock_guard<std::mutex> lock(shared_date_->main_mutex);
                shared_date_->local_rot = current_state_.rot.toRotationMatrix();
                shared_date_->local_pos = current_state_.pos;
                shared_date_->cloud = current_cloud_body_;
                offset_rot_ = shared_date_->offset_rot;
                offset_pos_ = shared_date_->offset_pos;
                shared_date_->pose_updated = true;
            }
            br_.sendTransform(eigen2Transform(
                current_state_.rot.toRotationMatrix(),
                current_state_.pos,
                local_frame_,
                body_frame_,
                current_time_));
            br_.sendTransform(eigen2Transform(
                offset_rot_,
                offset_pos_,
                global_frame_,
                local_frame_,
                current_time_));
            // 获取速度和角速度
            Eigen::Vector3d vel = current_state_.vel;
            Eigen::Vector3d angular_vel(0, 0, 0);
            if (!measure_group_.imus.empty())
            {
                // 使用最新的IMU数据获取角速度
                angular_vel(0) = measure_group_.imus.back().gyro(0);
                angular_vel(1) = measure_group_.imus.back().gyro(1);
                angular_vel(2) = measure_group_.imus.back().gyro(2);
            }
            
            // 获取协方差矩阵
            auto kf = lio_builder_->getKF();
            auto P = kf->get_P();
            
            // 计算全局坐标系下的位姿（camera_init -> body）
            // 全局位姿 = offset * local位姿
            Eigen::Matrix3d global_rot = offset_rot_ * current_state_.rot.toRotationMatrix();
            Eigen::Vector3d global_pos = offset_rot_ * current_state_.pos + offset_pos_;
            
            // 生成完整的Odometry（格式与 faster-lio/fast_lio_localization 一致）
            nav_msgs::Odometry odom_full = eigen2OdometryFull(
                global_rot, global_pos,
                vel, angular_vel, P,
                "camera_init",  // 使用 camera_init 作为 frame_id（与 faster-lio/fast_lio_localization 一致）
                "body",
                current_time_
            );
            
            publishOdom(odom_full);
            publishCloud(body_cloud_pub_,
                         pcl2msg(current_cloud_body_,
                                 body_frame_,
                                 current_time_));
            publishCloud(local_cloud_pub_,
                         pcl2msg(lio_builder_->cloudWorld(),
                                 local_frame_,
                                 current_time_));
            if (publish_map_cloud_)
            {
                if (icp_localizer_->isInitialized())
                {
                    publishCloud(map_cloud_pub_,
                                 pcl2msg(icp_localizer_->getRoughMap(),
                                         global_frame_,
                                         current_time_));
                }
            }
        }

        localizer_thread_->join();
        std::cout << "LOCALIZER NODE IS DOWN!" << std::endl;
    }

private:
    ros::NodeHandle nh_;
    std::string body_frame_;
    std::string local_frame_;
    std::string global_frame_;

    double current_time_;
    bool publish_map_cloud_;
    fastlio::state_ikfom current_state_;

    ImuData imu_data_;
    LivoxData livox_data_;
    MeasureGroup measure_group_;
    std::shared_ptr<SharedData> shared_date_;
    std::shared_ptr<ros::Rate> local_rate_;
    std::shared_ptr<ros::Rate> loop_rate_;
    tf2_ros::TransformBroadcaster &br_;
    fastlio::LioParams lio_params_;
    fastlio::LocalizerParams localizer_params_;
    std::shared_ptr<fastlio::LIOBuilder> lio_builder_;
    std::shared_ptr<fastlio::IcpLocalizer> icp_localizer_;
    LocalizerThread localizer_loop_;
    std::shared_ptr<std::thread> localizer_thread_;

    ros::Subscriber imu_sub_;

    ros::Subscriber livox_sub_;

    // 自动加载相关
    bool auto_load_enable_;
    std::string auto_load_map_path_;
    bool auto_load_auto_reloc_;
    double auto_load_initial_x_, auto_load_initial_y_, auto_load_initial_z_;
    double auto_load_initial_roll_, auto_load_initial_pitch_, auto_load_initial_yaw_;

    ros::Publisher odom_pub_;
    ros::Publisher odom_mavros_pub_;  // 定位模式：发布到 /mavros/local_position/odom

    ros::Publisher body_cloud_pub_;

    ros::Publisher local_cloud_pub_;

    ros::Publisher map_cloud_pub_;

    ros::ServiceServer reloc_server_;

    ros::ServiceServer map_convert_server_;

    ros::ServiceServer reloc_check_server_;

    ros::ServiceServer hold_server_;

    ros::ServiceServer start_server_;

    Eigen::Matrix3d offset_rot_ = Eigen::Matrix3d::Identity();

    Eigen::Vector3d offset_pos_ = Eigen::Vector3d::Zero();

    fastlio::PointCloudXYZI::Ptr current_cloud_body_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "localizer_node");
    tf2_ros::TransformBroadcaster br;
    signal(SIGINT, signalHandler);
    std::shared_ptr<SharedData> shared_date = std::make_shared<SharedData>();
    LocalizerROS localizer_ros(br, shared_date);
    localizer_ros.run();
    return 0;
}