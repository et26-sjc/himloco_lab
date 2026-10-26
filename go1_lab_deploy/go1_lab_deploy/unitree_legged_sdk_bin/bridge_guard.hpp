#ifndef HIMLOCO_BRIDGE_GUARD_HPP
#define HIMLOCO_BRIDGE_GUARD_HPP

// SDK-independent gate. Sample the monotonic clock INSIDE the mutex: timestamps
// captured by different callers before locking can arrive in reversed order.
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <mutex>
#include <utility>

namespace him_bridge {
constexpr std::int64_t kProtocol = 20261023;
constexpr double kStateTimeout = 0.050;  // Deployment limits, not training parameters.
constexpr double kCommandTimeout = 0.100;
constexpr std::int64_t kEchoMaxAgeUs = 100000;

enum class Fault { None, StopCommand, InvalidState, StateTimeout, InvalidCommand,
                   CommandTimeout, EmergencyButton, PositionLimit, PowerProtection, Shutdown };
inline const char* faultName(Fault value) {
    switch (value) {
    case Fault::None: return "none";
    case Fault::StopCommand: return "stop command";
    case Fault::InvalidState: return "invalid SDK state";
    case Fault::StateTimeout: return "SDK state timeout";
    case Fault::InvalidCommand: return "invalid HIM command";
    case Fault::CommandTimeout: return "policy command timeout";
    case Fault::EmergencyButton: return "remote L2 stop";
    case Fault::PositionLimit: return "SDK PositionLimit changed target";
    case Fault::PowerProtection: return "SDK PowerProtect";
    case Fault::Shutdown: return "bridge shutdown";
    }
    return "unknown";
}
struct Command {
    std::array<double, 12> q{}, dq{}, kp{}, kd{}, tau{};
    std::int64_t id = 0;
    std::int64_t protocol = 0;
    std::int64_t state_timestamp_us = 0;
};
struct Snapshot {
    Command command;
    bool active = false;
    bool state_ready = false;
    Fault fault = Fault::None;
    double state_age_s = -1.0;
    double command_age_s = -1.0;
    std::int64_t last_command_id = 0;
};
inline bool validCommand(const Command& command) {
    // SDK: FR, FL, RR, RL; each block hip/thigh/calf. Official HIM Go1 URDF bounds.
    // q_des is float32 on the wire. Compare the same rounded endpoints used
    // by Python, rather than rejecting an endpoint due to float/double rounding.
    const float lower[3] = {-0.802851455917f, -1.0471975512f, -2.69653369433f};
    const float upper[3] = { 0.802851455917f,  4.18879020479f, -0.916297857297f};
    for (std::size_t i = 0; i < 12; ++i) {
        if (!std::isfinite(command.q[i]) || !std::isfinite(command.dq[i]) ||
            !std::isfinite(command.kp[i]) || !std::isfinite(command.kd[i]) ||
            !std::isfinite(command.tau[i])) return false;
        if (command.q[i] < lower[i % 3] || command.q[i] > upper[i % 3] ||
            command.kp[i] < 0.0 || command.kp[i] > 40.0 ||
            command.kd[i] < 0.0 || command.kd[i] > 1.0 ||
            command.dq[i] != 0.0 || command.tau[i] != 0.0) return false;
    }
    return true;
}
class Guard {
public:
    using Clock = std::function<double()>;
    explicit Guard(Clock clock = steadySeconds) : clock_(std::move(clock)) {}
    // A second forward tick establishes readiness. Duplicate ticks never refresh age.
    bool observeState(std::uint32_t tick, bool valid, std::int64_t source_timestamp_us) {
        std::lock_guard<std::mutex> lock(mutex_);
        const double now = clock_();
        expireLocked(now);
        if (!valid || !std::isfinite(now) || source_timestamp_us <= 0) {
            ready_ = false;
            if (armed_) tripLocked(Fault::InvalidState);
            return false;
        }
        if (!seen_tick_) {
            seen_tick_ = true;
            last_tick_ = tick;
            last_state_time_ = now;
            last_state_stamp_ = source_timestamp_us;
            return false;
        }
        const std::uint32_t advance = tick - last_tick_;
        // Modular comparison handles normal uint32 wrap and rejects old packets.
        if (advance == 0 || advance >= 0x80000000u) return false;
        if (source_timestamp_us <= last_state_stamp_) return false;
        last_tick_ = tick;
        last_state_time_ = now;
        last_state_stamp_ = source_timestamp_us;
        ready_ = true;
        published_stamps_[next_stamp_slot_] = source_timestamp_us;
        next_stamp_slot_ = (next_stamp_slot_ + 1) % published_stamps_.size();
        return true;
    }
    bool submit(const Command& command) {
        std::lock_guard<std::mutex> lock(mutex_);
        const double now = clock_();
        expireLocked(now);
        if (command.protocol != kProtocol) return false;
        // Stop ignores payload, stale sensors and sequence; the marker is still required.
        if (command.id == -1) { tripLocked(Fault::StopCommand); return true; }
        if (fault_ != Fault::None || command.id <= 0 || command.id <= last_command_id_)
            return false;
        if (!validCommand(command)) { tripLocked(Fault::InvalidCommand); return false; }
        if (!freshLocked(now) || emergency_pressed_) return false;
        if (command.state_timestamp_us <= 0 || command.state_timestamp_us > last_state_stamp_ ||
            std::find(published_stamps_.begin(), published_stamps_.end(),
                      command.state_timestamp_us) == published_stamps_.end() ||
            static_cast<double>(last_state_stamp_ - command.state_timestamp_us) +
                (now - last_state_time_) * 1e6 > kEchoMaxAgeUs) {
            if (armed_) tripLocked(Fault::InvalidCommand);
            return false;
        }
        command_ = command;
        last_command_id_ = command.id;
        last_command_time_ = now;
        armed_ = true;
        return true;
    }
    Snapshot snapshot() {
        std::lock_guard<std::mutex> lock(mutex_);
        const double now = clock_();
        expireLocked(now);
        Snapshot result;
        result.command = command_;
        result.state_ready = freshLocked(now);
        result.fault = fault_;
        result.active = armed_ && fault_ == Fault::None && result.state_ready;
        result.state_age_s = seen_tick_ ? now - last_state_time_ : -1.0;
        result.command_age_s = armed_ ? now - last_command_time_ : -1.0;
        result.last_command_id = last_command_id_;
        return result;
    }
    void trip(Fault reason) {
        std::lock_guard<std::mutex> lock(mutex_);
        tripLocked(reason);
    }
    void emergencyButton(bool pressed) {
        std::lock_guard<std::mutex> lock(mutex_);
        emergency_pressed_ = pressed;
        if (pressed && armed_) tripLocked(Fault::EmergencyButton);
    }
private:
    static double steadySeconds() {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    bool freshLocked(double now) const {
        return ready_ && std::isfinite(now) && now >= last_state_time_ &&
               now - last_state_time_ <= kStateTimeout;
    }
    void tripLocked(Fault reason) {
        if (fault_ == Fault::None) fault_ = reason;  // Restart required to rearm.
    }
    void expireLocked(double now) {
        if (!armed_ || fault_ != Fault::None) return;
        if (!freshLocked(now)) tripLocked(Fault::StateTimeout);
        else if (!std::isfinite(now) || now < last_command_time_ ||
                 now - last_command_time_ > kCommandTimeout)
            tripLocked(Fault::CommandTimeout);
    }
    std::mutex mutex_;
    const Clock clock_;
    bool seen_tick_ = false, ready_ = false, armed_ = false, emergency_pressed_ = false;
    std::uint32_t last_tick_ = 0;
    double last_state_time_ = 0.0, last_command_time_ = 0.0;
    std::int64_t last_state_stamp_ = 0, last_command_id_ = 0;
    // A timestamp must belong to this bridge instance, not merely appear recent.
    // 128 stamps cover >100 ms at the nominal 500 Hz publication rate.
    std::array<std::int64_t, 128> published_stamps_{};
    std::size_t next_stamp_slot_ = 0;
    Fault fault_ = Fault::None;
    Command command_;
};
}  // namespace him_bridge
#endif
