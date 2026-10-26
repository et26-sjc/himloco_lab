/*****************************************************************
 Copyright (c) 2020, Unitree Robotics.Co.Ltd. All rights reserved.
 HIMLoco deployment adaptation of the WTW SDK/LCM bridge.
******************************************************************/
#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include "unitree_legged_sdk/joystick.h"
#include "bridge_guard.hpp"
#include <lcm/lcm-cpp.hpp>
#include "state_estimator_lcmt.hpp"
#include "leg_control_data_lcmt.hpp"
#include "pd_tau_targets_lcmt.hpp"
#include "rc_command_lcmt.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#ifndef HIM_BUILD_SOURCE_SHA256
#define HIM_BUILD_SOURCE_SHA256 "untracked-build"
#endif

using namespace UNITREE_LEGGED_SDK;
using SteadyClock = std::chrono::steady_clock;
static_assert(ATOMIC_BOOL_LOCK_FREE == 2, "Signal shutdown requires lock-free atomic bool");
static std::atomic<bool> shutdown_requested(false);
static void requestShutdown(int) { shutdown_requested.store(true, std::memory_order_relaxed); }

class Custom {
public:
    static constexpr double dt = 0.002;  // SDK 500 Hz; policy remains 50 Hz.
    explicit Custom(const std::string& lcm_url, bool observe_only = false)
             : safe_(LeggedType::Go1), udp_(LOWLEVEL, 8090, "192.168.123.10", 8007),
               lcm_(lcm_url),
               observe_only_(observe_only),
               steady_origin_(SteadyClock::now()),
               epoch_origin_us_(std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::system_clock::now().time_since_epoch()).count()) {
        udp_.InitCmdData(cmd_);
        setInactive();
        udp_.SetSend(cmd_);  // Never initialize an active nominal pose.
    }
    void init() {
        if (!lcm_.good()) throw std::runtime_error("LCM initialization failed");
        if (!observe_only_) {
            auto* subscription = lcm_.subscribe("pd_plustau_targets", &Custom::handleActionLCM, this);
            if (!subscription) throw std::runtime_error("LCM command subscription failed");
            subscription->setQueueCapacity(1);
        }
        // All fields and guard are initialized before the callback thread exists.
        lcm_thread_ = std::thread([this]() {
            while (true) {
                if (lcm_.handleTimeout(100) < 0) {
                    guard_.trip(him_bridge::Fault::Shutdown);
                    return;
                }
            }
        });
        lcm_thread_.detach();  // Main uses _Exit after sending inactive frames.
    }
    void UDPRecv() { udp_.Recv(); }
    void UDPSend() { udp_.Send(); }
    void stop() { guard_.trip(him_bridge::Fault::Shutdown); }

    void RobotControl() {
        udp_.GetRecv(state_);
        const double now = monotonicSeconds();
        xRockerBtnDataStruct keys{};
        static_assert(sizeof(keys) == 40, "Unsupported Unitree joystick layout");
        std::memcpy(&keys, state_.wirelessRemote.data(), sizeof(keys));
        const bool valid = validState(keys);
        const std::int64_t source_stamp = epoch_origin_us_ +
            static_cast<std::int64_t>(now * 1e6);
        const bool new_state = guard_.observeState(state_.tick, valid, source_stamp);
        if (valid) guard_.emergencyButton(keys.btn.components.L2 != 0);
        if (shutdown_requested.load(std::memory_order_relaxed)) stop();
        auto snapshot = guard_.snapshot();
        if (snapshot.active && !observe_only_) {
            std::array<float, 12> requested_targets{};
            for (std::size_t i = 0; i < 12; ++i) {
                cmd_.motorCmd[i].mode = 0x0A;  // Go1 SDK: servo mode.
                cmd_.motorCmd[i].q = snapshot.command.q[i];
                cmd_.motorCmd[i].dq = 0.0f;
                cmd_.motorCmd[i].Kp = snapshot.command.kp[i];
                cmd_.motorCmd[i].Kd = snapshot.command.kd[i];
                cmd_.motorCmd[i].tau = 0.0f;
                requested_targets[i] = cmd_.motorCmd[i].q;
            }
            safe_.PositionLimit(cmd_);
            bool target_changed = false;
            for (std::size_t i = 0; i < 12; ++i)
                if (cmd_.motorCmd[i].q != requested_targets[i]) target_changed = true;
            if (target_changed) {
                // SDK hard limits may differ from training URDF. Never silently
                // execute a clipped target while policy history records another.
                guard_.trip(him_bridge::Fault::PositionLimit);
                snapshot = guard_.snapshot();
                setInactive();
            } else if (safe_.PowerProtect(cmd_, state_, 9) < 0) {
                guard_.trip(him_bridge::Fault::PowerProtection);
                snapshot = guard_.snapshot();
                setInactive();
            }
        } else {
            setInactive();
        }
        udp_.SetSend(cmd_);
        if (snapshot.fault != reported_fault_) {
            reported_fault_ = snapshot.fault;
            std::cerr << "HIM bridge latched: " << him_bridge::faultName(reported_fault_)
                      << "; command_age_ms=" << snapshot.command_age_s * 1000.0
                      << "; state_age_ms=" << snapshot.state_age_s * 1000.0
                      << "; last_command_id=" << snapshot.last_command_id
                      << "; restart bridge to rearm." << std::endl;
        }
        // Duplicate GetRecv values are never published as new measurements.
        if (new_state) publishState(keys, source_stamp, snapshot.fault);
    }
private:
    double monotonicSeconds() const {
        return std::chrono::duration<double>(SteadyClock::now() - steady_origin_).count();
    }
    bool validState(const xRockerBtnDataStruct& keys) const {
        for (std::size_t i = 0; i < 12; ++i)
            if (!std::isfinite(state_.motorState[i].q) || !std::isfinite(state_.motorState[i].dq) ||
                !std::isfinite(state_.motorState[i].tauEst)) return false;
        for (std::size_t i = 0; i < 3; ++i)
            if (!std::isfinite(state_.imu.rpy[i]) || !std::isfinite(state_.imu.gyroscope[i]) ||
                !std::isfinite(state_.imu.accelerometer[i])) return false;
        double norm2 = 0.0;
        for (float value : state_.imu.quaternion) {
            if (!std::isfinite(value)) return false;
            norm2 += static_cast<double>(value) * value;
        }
        // SDK declares a normalized wxyz quaternion; reject uninitialized/zero data.
        if (norm2 < 0.81 || norm2 > 1.21) return false;
        return std::isfinite(keys.lx) && std::isfinite(keys.ly) &&
               std::isfinite(keys.rx) && std::isfinite(keys.ry) &&
               std::abs(keys.lx) <= 1.01f && std::abs(keys.ly) <= 1.01f &&
               std::abs(keys.rx) <= 1.01f && std::abs(keys.ry) <= 1.01f;
    }
    void setInactive() {
        for (auto& motor : cmd_.motorCmd) {
            motor.mode = 0x00;  // SDK damping mode, NOT electrical power-off.
            motor.q = PosStopF;
            motor.dq = VelStopF;
            motor.Kp = 0.0f;
            motor.Kd = 0.0f;
            motor.tau = 0.0f;
        }
    }
    void handleActionLCM(const lcm::ReceiveBuffer*, const std::string&,
                         const pd_tau_targets_lcmt* message) {
        him_bridge::Command command;
        command.id = message->id;
        command.protocol = message->robot_id;
        // Echo bridge sensor stamp; sender and bridge clocks are never compared.
        command.state_timestamp_us = message->timestamp_us;
        for (std::size_t i = 0; i < 12; ++i) {
            command.q[i] = message->q_des[i];
            command.dq[i] = message->qd_des[i];
            command.kp[i] = message->kp[i];
            command.kd[i] = message->kd[i];
            command.tau[i] = message->tau_ff[i];
        }
        guard_.submit(command);
    }
    void publishState(const xRockerBtnDataStruct& keys, std::int64_t stamp,
                      him_bridge::Fault fault) {
        state_estimator_lcmt body{};
        leg_control_data_lcmt joints{};
        rc_command_lcmt remote{};
        body.id = joints.id = static_cast<std::int64_t>(state_.tick);
        body.timestamp_us = joints.timestamp_us = stamp;
        body.robot_id = joints.robot_id = fault == him_bridge::Fault::None ?
            him_bridge::kProtocol : -him_bridge::kProtocol;
        for (std::size_t i = 0; i < 12; ++i) {
            joints.q[i] = state_.motorState[i].q;
            joints.qd[i] = state_.motorState[i].dq;
            joints.tau_est[i] = state_.motorState[i].tauEst;
        }
        for (std::size_t i = 0; i < 4; ++i) {
            body.quat[i] = state_.imu.quaternion[i];
            body.contact_estimate[i] = state_.footForce[i];
        }
        for (std::size_t i = 0; i < 3; ++i) {
            body.rpy[i] = state_.imu.rpy[i];
            body.omegaBody[i] = state_.imu.gyroscope[i];
            body.aBody[i] = state_.imu.accelerometer[i];
        }
        remote.left_stick[0] = keys.lx;
        remote.left_stick[1] = keys.ly;
        remote.right_stick[0] = keys.rx;
        remote.right_stick[1] = keys.ry;
        remote.right_lower_right_switch = keys.btn.components.R2;
        remote.right_upper_switch = keys.btn.components.R1;
        remote.left_lower_left_switch = keys.btn.components.L2;
        remote.left_upper_switch = keys.btn.components.L1;
        lcm_.publish("state_estimator_data", &body);
        lcm_.publish("leg_control_data", &joints);
        lcm_.publish("rc_command", &remote);
    }
    Safety safe_;
    UDP udp_;
    LowCmd cmd_{};
    LowState state_{};
    lcm::LCM lcm_;
    const bool observe_only_;
    std::thread lcm_thread_;
    him_bridge::Guard guard_;
    him_bridge::Fault reported_fault_ = him_bridge::Fault::None;
    SteadyClock::time_point steady_origin_;
    std::int64_t epoch_origin_us_;
};

constexpr double Custom::dt;

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << "HIM_BRIDGE protocol=" << him_bridge::kProtocol
                  << " source_sha256=" << HIM_BUILD_SOURCE_SHA256 << std::endl;
        return 0;  // Before SDK construction, sockets, or motor commands.
    }
    std::string lcm_url = "udpm://239.255.76.67:7667?ttl=255";
    bool observe_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--observe-only") observe_only = true;
        else if (arg == "--lcm-url" && i + 1 < argc) lcm_url = argv[++i];
        else {
            std::cerr << "Usage: lcm_position_him [--version | --observe-only] [--lcm-url URL]" << std::endl;
            return 2;
        }
    }
    if (observe_only)
        std::cout << "OBSERVE ONLY: never subscribes to motor targets; mode 0 / zero gains.\n"
                  << "The body must be supported; R2 cannot enable actuation in this mode.\n";
    std::cout << "HIM Go1 LOW-level bridge; 500 Hz. Support the robot before starting.\n"
              << "No servo target before a valid HIM command. Stop/fault is latched.\n"
              << "Press Enter to start bridge communication..." << std::endl;
    std::string confirmation;
    if (!std::getline(std::cin, confirmation)) return 2;
    std::signal(SIGINT, requestShutdown);
    std::signal(SIGTERM, requestShutdown);
    Custom custom(lcm_url, observe_only);
    custom.init();
    LoopFunc control("control_loop", Custom::dt, boost::bind(&Custom::RobotControl, &custom));
    LoopFunc send("udp_send", Custom::dt, 3, boost::bind(&Custom::UDPSend, &custom));
    LoopFunc receive("udp_recv", Custom::dt, 3, boost::bind(&Custom::UDPRecv, &custom));
    send.start();
    receive.start();
    control.start();
    while (!shutdown_requested.load(std::memory_order_relaxed))
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    custom.stop();
    // Transmit inactive frames before exit. SIGKILL/hang/power loss cannot be handled here.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::cout << "HIM bridge stopped." << std::endl;
    std::_Exit(EXIT_SUCCESS);
}
