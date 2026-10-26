#include "../../unitree_legged_sdk_bin/bridge_guard.hpp"
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>
using namespace him_bridge;
// The production guard samples its clock only after locking. This adapter keeps
// the existing deterministic test timeline without exposing caller timestamps in
// the production API. Concurrent legacy checks all use the same fixed time.
class TestGuard : public him_bridge::Guard {
public:
    TestGuard() : him_bridge::Guard([this]() { return now_.load(); }) {}
    bool observeState(std::uint32_t tick, bool valid, double now, std::int64_t stamp) {
        now_.store(now);
        return him_bridge::Guard::observeState(tick, valid, stamp);
    }
    bool submit(const Command& command, double now) {
        now_.store(now);
        return him_bridge::Guard::submit(command);
    }
    Snapshot snapshot(double now) {
        now_.store(now);
        return him_bridge::Guard::snapshot();
    }
private:
    std::atomic<double> now_{0.0};
};
#define CHECK(expression) do { if (!(expression)) { std::cerr << "FAIL " << __LINE__ << ": " << #expression << '\n'; std::exit(1); } } while (0)

Command command(std::int64_t id = 1, std::int64_t stamp = 1002000) {
    Command c;
    c.id = id; c.protocol = kProtocol; c.state_timestamp_us = stamp;
    for (std::size_t i = 0; i < 12; ++i) {
        c.q[i] = i % 3 == 0 ? 0.0 : (i % 3 == 1 ? 0.8 : -1.5);
        c.kp[i] = 40.0; c.kd[i] = 1.0;
    }
    return c;
}
void ready(TestGuard& g) {
    CHECK(!g.observeState(100, true, 0.0, 1000000));
    CHECK(g.observeState(102, true, 0.002, 1002000));
}
int main() {
    {
        // Regression: a control thread may capture its source timestamp before
        // a command thread acquires the guard. Receipt time must instead come
        // from the clock under the guard lock, never that older source stamp.
        double clock = 0.0;
        him_bridge::Guard g([&]() { return clock; });
        CHECK(!g.observeState(100, true, 1000000));
        clock = .002;
        CHECK(g.observeState(102, true, 1002000));
        const std::int64_t earlier_control_stamp = 1003990;
        clock = .004;
        CHECK(g.submit(command()));
        clock = .004010;
        CHECK(g.observeState(104, true, earlier_control_stamp));
        auto s = g.snapshot();
        CHECK(s.active);
        CHECK(s.fault == Fault::None);
        CHECK(s.command_age_s >= 0 && s.command_age_s < .000011);
        CHECK(s.last_command_id == 1);
        // The genuine 100 ms command watchdog still trips with fresh state.
        clock = .040;
        CHECK(g.observeState(140, true, 1040000));
        clock = .080;
        CHECK(g.observeState(180, true, 1080000));
        clock = .105;
        CHECK(g.observeState(205, true, 1105000));
        CHECK(g.snapshot().fault == Fault::CommandTimeout);
    }
    {
        TestGuard g;
        CHECK(!g.snapshot(0).active);
        CHECK(!g.submit(command(), 0));
        CHECK(!g.observeState(0, false, 0, 1000000));
        CHECK(!g.observeState(0, true, 0, 1000000));
        CHECK(!g.submit(command(), 0));
        CHECK(g.observeState(2, true, .002, 1002000));
        CHECK(g.submit(command(), .002));
        CHECK(g.snapshot(.002).active);
        CHECK(!g.submit(command(), .003));  // Replay must not refresh command timer.
        CHECK(!g.observeState(2, true, .040, 1040000));
        CHECK(!g.snapshot(.053).active);
        CHECK(g.snapshot(.053).fault == Fault::StateTimeout);
        CHECK(g.observeState(100, true, .054, 1054000));
        CHECK(!g.submit(command(2,1054000), .054));
    }
    {
        TestGuard g; ready(g);
        CHECK(g.submit(command(), .002));
        for (int n=1;n<=5;++n) CHECK(g.observeState(102+n, true, .002+.02*n, 1002000+20000*n));
        CHECK(!g.snapshot(.103).active);
        CHECK(g.snapshot(.103).fault == Fault::CommandTimeout);
    }
    {
        TestGuard g; ready(g);
        auto wrong = command(); wrong.protocol = 0;
        CHECK(!g.submit(wrong, .002));
        CHECK(g.snapshot(.002).fault == Fault::None);
        wrong = command(); wrong.state_timestamp_us = 1002001;
        CHECK(!g.submit(wrong, .002));  // Future echo is rejected without arming.
        wrong.state_timestamp_us = 1000000;
        CHECK(!g.submit(wrong, .002));  // First candidate was never published.
        wrong.state_timestamp_us = 1001000;
        CHECK(!g.submit(wrong, .002));  // Invented recent stamp must not arm.
        wrong.state_timestamp_us = 992000;
        CHECK(!g.submit(wrong, .002));  // Old bridge session stamp, only 10 ms old.
        wrong.state_timestamp_us = std::numeric_limits<std::int64_t>::min();
        CHECK(!g.submit(wrong, .002));  // No signed-overflow in age arithmetic.
        CHECK(g.submit(command(), .002));
        wrong = command(2); wrong.state_timestamp_us = 1;
        CHECK(!g.submit(wrong, .003));
        CHECK(g.snapshot(.003).fault == Fault::InvalidCommand);
    }
    {
        TestGuard g;
        auto stop = command(-1); stop.q[0] = std::numeric_limits<double>::quiet_NaN();
        stop.state_timestamp_us = 0;
        CHECK(g.submit(stop,0));
        ready(g);
        CHECK(!g.submit(command(),.002));
        CHECK(g.snapshot(.002).fault == Fault::StopCommand);
    }
    {
        TestGuard g;
        CHECK(!g.observeState(0xfffffffeu,true,0,1000000));
        CHECK(g.observeState(1u,true,.002,1002000));
        CHECK(!g.observeState(0xfffffffeu,true,.003,1003000));
        CHECK(g.submit(command(),.003));
        CHECK(!g.observeState(2u,false,.004,1004000));
        CHECK(g.snapshot(.004).fault == Fault::InvalidState);
    }
    {
        TestGuard g; ready(g); g.emergencyButton(true);
        CHECK(!g.submit(command(),.002));
        g.emergencyButton(false); CHECK(g.submit(command(),.002));
        g.emergencyButton(true);
        CHECK(g.snapshot(.003).fault == Fault::EmergencyButton);
        g.emergencyButton(false); CHECK(!g.submit(command(2),.003));
    }
    {
        auto c = command(); CHECK(validCommand(c));
        c.q[2] = static_cast<float>(-2.69653369433);
        c.q[5] = static_cast<float>(-0.916297857297);
        CHECK(validCommand(c));  // Python/LCM float32 endpoint roundoff.
        for (int kind=0;kind<7;++kind) {
            c=command();
            if(kind==0)c.q[0]=.803;
            if(kind==1)c.q[2]=0;
            if(kind==2)c.kp[1]=40.01;
            if(kind==3)c.kd[2]=1.01;
            if(kind==4)c.dq[0]=.01;
            if(kind==5)c.tau[11]=.01;
            if(kind==6)c.q[3]=std::numeric_limits<double>::infinity();
            CHECK(!validCommand(c));
            TestGuard g;ready(g);CHECK(!g.submit(c,.002));
            CHECK(g.snapshot(.002).fault==Fault::InvalidCommand);
        }
    }
    {
        TestGuard g;ready(g);CHECK(g.submit(command(),.002));
        g.trip(Fault::PowerProtection);g.trip(Fault::Shutdown);
        CHECK(g.snapshot(.002).fault==Fault::PowerProtection);
    }
    {
        // Contending callbacks and snapshot readers must never see mixed fields.
        TestGuard g;ready(g);std::atomic<bool> done(false);
        std::thread writer([&](){
            for(int id=1;id<=10000;++id){
                auto c=command(id);const double value=id%2?.1:.2;
                c.kp.fill(value);c.kd.fill(value);CHECK(g.submit(c,.002));
            }
            done.store(true);
        });
        std::vector<std::thread> readers;
        for(int r=0;r<3;++r)readers.emplace_back([&](){
            do{auto s=g.snapshot(.002);if(s.active){
                for(int i=0;i<12;++i){CHECK(s.command.kp[i]==s.command.kp[0]);
                    CHECK(s.command.kd[i]==s.command.kp[0]);}
            }}while(!done.load());
        });
        writer.join();for(auto& reader:readers)reader.join();
    }
    std::cout << "PASS: startup, first-command gate, timestamps, sequence replay, tick wrap,\n"
                 "state/command watchdogs, latched stop/fault/L2, bounds, concurrent snapshots,\n"
                 "clock sampled under lock (no false timeout from caller interleaving)\n";
}
