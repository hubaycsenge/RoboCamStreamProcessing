# link — everything crossing the robot ↔ cluster boundary

Two directions, one directory: the **control path** (give and receive ROS 2
commands from the cluster) and the **data path** (stream frames to the server,
get results back). They were separate — `mecanumbot-link` for the first, the
client and its notes for the second — which was tolerable only while the data
path was assumed to work without a tunnel. It does not. Both now depend on the
same SSH plumbing, so both live here.

| file | direction | what it is |
| --- | --- | --- |
| `robocam_client.py` | data | standalone client, deploy to the Orin. No ROS, one file. |
| `robot` | control | run `ros2` on the robot from nipg1, over the reverse tunnel |
| `ssh_config` | both | host aliases; `robot` reads this file directly |
| `ros-env.sh` | control | ROS env sourced *on the robot* by the wrapper |
| `mecanumbot-tunnel.service` | control | robot:22 → nipg1:8200, robot-side systemd |
| `mecanumbot-deep3r-tunnel.service` | data | robot:5555 → nipg1:5555, robot-side systemd |
| `netcheck.sh` | data | can the robot actually reach the server? |

The cluster-side half of the data path lives in `../scripts/run_deep3r_bridged.sh`
— it runs inside the Slurm job and forwards the server back to nipg1.

## The network, as measured

Not as previously documented. These were tested on 2026-08-28 and 2026-09-01,
from the hosts named:

| from | to | result | when |
| --- | --- | --- | --- |
| nipg1 | robot | **no route** | 08-28 |
| robot | `nipg1.inf.elte.hu:22` | reachable | 08-28 |
| robot | `nipg36` `10.128.17.196:5555` | **unreachable** | 08-28 |
| robot | `nipg36.inf.elte.hu:10113` | reachable | 08-28 |
| nipg1 | `nipg36:10113` | reachable | 08-28 |
| nipg1 | `10.128.17.196` (ping) | reachable, 0.7 ms, via `157.181.160.254` | 09-01 |
| nipg1 | `nipg36:22` | **connection refused** | 09-01 |
| nipg3 (Slurm job) | `nipg1:22` | reachable | 09-01 |

The robot is on the lab WiFi at **192.168.1.240** (earlier notes said
`192.168.0.240`; that subnet is stale). nipg1 is on the public university
subnet at `157.181.160.161`; nipg36's `10.128.17.x` address is private to the
cluster, though *routed* from nipg1 rather than isolated from it — the last
three rows are why the data path no longer goes anywhere near nipg36.

Four facts follow, and they shape everything here:

1. **The robot always initiates.** Nothing outside can dial in through the
   router's NAT. Both tunnels are therefore opened *by the robot*, which is
   also what makes them permanent: the robot is the always-on machine on that
   WiFi, whereas a laptop tunnel lives only while the laptop does.
2. **No cluster address is reachable from the robot.** The main README's "the
   robot connects to `10.128.17.196:5555`" describes a situation the robot is
   not in. `mecanumbot-deep3r-tunnel.service` is the fix: dial out over SSH and
   forward 5555 back. The client then talks to `tcp://127.0.0.1:5555`.
3. **The rendezvous is nipg1, and the compute is anywhere.** This is the one
   thing that changed on 2026-09-01. The deep3r tunnel used to target nipg36's
   public SSH port, because that was the port measured to work — but nipg1 is
   equally reachable from the robot (fact 1 has been relying on it all along),
   and nipg1 is the better endpoint precisely *because* it has no GPU. It is
   the login node, absent from `sinfo -N`, so it cannot host the server and the
   two concerns are forced apart:

   - the tunnel endpoint is fixed, and the robot never has to be reconfigured;
   - the server is a Slurm job, on whatever node has a card free.

   Targeting nipg36 pinned the *compute* to nipg36 as a side effect, and nipg36
   is 2× TITAN RTX — Turing, sm_75, no bf16, the slowest thing you would pick
   for deep3r deliberately. (`deep3r-live` migrated off Turing for exactly this
   reason.) The job's half of the bridge is `../scripts/run_deep3r_bridged.sh`.

   It has to be the *job* that dials nipg1, not nipg1 that dials the job: you
   cannot ssh into a compute node — `nipg36:22` is connection-refused — so a
   `-L` from nipg1 has nothing to connect to.
4. **`ros2` runs ON the robot, not on nipg1.** DDS discovery needs a real
   network path that a single TCP tunnel cannot provide, so a ROS node on nipg1
   would see an empty graph. `robot` runs each command over SSH on the robot and
   streams the result back — local in feel, remote in mechanism.

```
  you @ nipg1                              Mecanumbot (192.168.1.240)
  ───────────                              ──────────────────────────
  robot topic list ──ssh nipg1:8200──▶ sshd ──▶ ros2 topic list
                   ◀───── stdout ──────────────────  (DOMAIN_ID 19, cyclonedds)
        ▲                                                  │
        └────── reverse tunnel held open by ───────────────┘
                mecanumbot-tunnel.service

  compute node (Slurm)          nipg1                    Mecanumbot
  ────────────────────          ─────                    ──────────
  server :5555 ──ssh -R──▶  :5555 (loopback)  ◀──ssh -L── robocam_client.py
  run_deep3r_bridged.sh      the rendezvous     mecanumbot-deep3r-tunnel.service
                                                       → tcp://127.0.0.1:5555
```

Both halves bind nipg1's loopback (sshd's `GatewayPorts` defaults to `no`), so
they meet on that one port and neither end needs the other's address.

## Everyday use

    robot                      # interactive shell on the robot, ROS 2 sourced
    robot topic list           # ros2 is implied
    robot topic echo /mecanumbot/scan
    robot status               # is the tunnel up? prints the topic count
    robot grippers close       # helper: close grippers (neck held at 6.5)
    robot grippers open        # helper: open grippers (neutral 5.12)
    robot pub-accessory 6.5 6.83 3.36     # raw {n_pos, gl_pos, gr_pos}
    robot raw 'ros2 param list /mecanumbot/mecanumbot_joy_node'

Put it on your PATH if you like:

    ln -sf "$PWD/robot" ~/.local/bin/robot

`robot` finds `ssh_config` next to itself, so moving this directory does not
break it. That is deliberate: the old arrangement needed
`Include ~/mecanumbot-link/ssh_config` in `~/.ssh/config`, and moving the
directory left that line dangling — the wrapper then failed with
`Could not resolve hostname mecanumbot`, which reads like a dead tunnel rather
than a stale path. If you still have that Include line, it is now unnecessary
and can go.

## One-time setup

Steps 1–2 are on nipg1; step 3 is on the robot and needs its sudo. The robot
must be reachable to do step 3 the first time — easiest via a laptop ad-hoc
tunnel (`ssh -R 8222:192.168.1.240:22 csengehubay@nipg1...`), which the
`mecanumbot-laptop` host alias targets.

**1. Authorize the robot's key on nipg1** (forwarding-only, no shell):

    cat >> ~/.ssh/authorized_keys <<'KEY'
    restrict,port-forwarding ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAID0KxNZEJjc9VV8VsAEu37zr3t2Q0zEa298uB+A0k8P4 csengehubay@gmail.com
    KEY
    chmod 600 ~/.ssh/authorized_keys

**2. Authorize nipg1's own key on nipg1** (forwarding-only, no shell), so that a
Slurm job on a compute node can open the deep3r reverse tunnel back here. Home is
shared over the NAS, so the private half is already on every node — only the
public half is missing:

    sed 's/^/restrict,port-forwarding /' ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys

Check it took — this must print `nipg1` and not prompt:

    ssh -o BatchMode=yes nipg1.inf.elte.hu true && echo ok

(The wrapper reads `ssh_config` from this directory, so there is no ssh config to
wire beyond this.)

**3. Install both tunnels on the robot:**

    scp -P 8222 *.service ubuntu@127.0.0.1:~/
    ssh -p 8222 ubuntu@127.0.0.1 \
      'sudo cp ~/mecanumbot-tunnel.service ~/mecanumbot-deep3r-tunnel.service /etc/systemd/system/ && \
       sudo systemctl daemon-reload && \
       sudo systemctl enable --now mecanumbot-tunnel.service mecanumbot-deep3r-tunnel.service && \
       systemctl status mecanumbot-tunnel mecanumbot-deep3r-tunnel --no-pager | head -20'

Then, from nipg1:

    ./robot status        # -> tunnel UP — NN topics visible on the robot

Enable only the reverse tunnel if you are not running the server; the forward
one will restart every 10 s against a closed port otherwise, which is harmless
but noisy in the journal.

### No root on the robot

Run them as user services with linger (one sudo, for linger only):

    ssh -p 8222 ubuntu@127.0.0.1 'mkdir -p ~/.config/systemd/user && \
      cp ~/mecanumbot-*.service ~/.config/systemd/user/ && \
      sed -i "/^User=/d" ~/.config/systemd/user/mecanumbot-*.service && \
      systemctl --user daemon-reload && \
      systemctl --user enable --now mecanumbot-tunnel mecanumbot-deep3r-tunnel'
    ssh -p 8222 ubuntu@127.0.0.1 'sudo loginctl enable-linger ubuntu'

## Security notes

- The robot's key on nipg1 is `restrict,port-forwarding`: no pty, no agent or
  X11 forwarding, no `~/.ssh/rc`. **Correction to an earlier claim here:** that
  is not "no shell". `restrict` disables pty allocation, not command execution,
  so a holder of that key can still run `ssh nipg1 <command>` non-interactively
  on the cluster account. To actually close that off, add a forced command —
  `restrict,port-forwarding,command="/bin/false"` — which leaves forwarding
  working and makes execution a no-op. Not yet applied; it is a deliberate
  decision to make, not a typo to fix.
- The same applies to the nipg1 self-key added in step 2 of the setup, which
  carries the same options for the same reason.
- Your own logins to nipg1 are unaffected (separate key).
- Anyone with the robot can, through this, reach the forwarded port — i.e. get
  an SSH prompt to *the robot*, not to the cluster. Revoke by deleting that line
  from `~/.ssh/authorized_keys` and disabling the robot-side service.
- Ports 8200 and 5555 bind nipg1's localhost only, so they are not exposed to
  other cluster users unless they share this account. 5555 in particular is an
  unauthenticated ZeroMQ endpoint, so keep it off `0.0.0.0` on nipg1 — do not
  add `GatewayPorts` or a `*:5555` bind to make it "easier to test".
- The deep3r tunnel no longer targets nipg36, so it no longer needs your
  unrestricted cluster key: pointed at nipg1, it reuses the robot's existing
  restricted key. That removes the shell-on-nipg36 exposure the previous version
  of this note warned about.

## Robot facts (captured 2026-08-27, addresses re-checked 2026-08-28)

- ROS 2 Humble; `ROS_DOMAIN_ID=19`; `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`.
- Workspace overlay: `~/mecanumbot_ws/install/setup.bash`.
- WiFi address `192.168.1.240` on `wlP1p1s0`.
- Accessory command: `/cmd_accessory_pos`, type `mecanumbot_msgs/msg/AccessMotorCmd`
  = `{float32 n_pos, float32 gl_pos, float32 gr_pos}`.
- Neck range 2.0–8.6; gripper range 1.6–8.54; neutral/"front" 5.12.
  Close = `gl 6.83 / gr 3.36` (mirrored); open = `gl 5.12 / gr 5.12`.
  Servo readback is on `/mecanumbot/opencr_state` (pos_* ×100 of the command).
