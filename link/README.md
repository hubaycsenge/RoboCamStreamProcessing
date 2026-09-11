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
| `robot` | control | run `ros2` on the robot from nipg1, over the reverse tunnel; `robot web` also forwards its web GUI to your browser |
| `ssh_config` | both | host aliases; `robot` reads this file directly. `mecanumbot-jump` is the same robot via `ProxyJump`, for a laptop rather than nipg1 |
| `ros-env.sh` | control | ROS env sourced *on the robot* by the wrapper |
| `mecanumbot-tunnel.service` | control | robot:22 → nipg1:8200, robot-side systemd |
| `mecanumbot-deep3r-tunnel.service` | data | robot:5555 → nipg1:5555, robot-side systemd |
| `netcheck.sh` | data | can the robot actually reach the server? |
| `robot-addr.sh` | setup | which of the two lab subnets the robot is on today |

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

The robot is on the lab WiFi, at **`192.168.0.240` or `192.168.1.240`
depending on which router it associated with** — the lab has two, and they hand
out different subnets:

| router | robot |
| --- | --- |
| `192.168.0.1` | `192.168.0.240` |
| `192.168.1.1` | `192.168.1.240` |

The last octet is fixed by a DHCP reservation, so only the third changes; the
robot is always `.240`. An earlier version of this file called `192.168.0.240`
"stale", which was wrong — it is the address on the other router, not a
historical one, and treating it as dead is what makes a working robot look
unreachable after someone moves it between rooms. `link/robot-addr.sh` probes
both and prints whichever answers.

Nothing else in this directory depends on which it is: both tunnels are opened
*by the robot*, so the robot's own address never has to be known from outside.
The one place it does matter is the laptop ad-hoc tunnel in step 3 of the setup
below, which has to name the robot to forward to it.

nipg1 is on the public university subnet at `157.181.160.161`; nipg36's
`10.128.17.x` address is private to the cluster, though *routed* from nipg1
rather than isolated from it — the last three rows are why the data path no
longer goes anywhere near nipg36.

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
  you @ nipg1                          Mecanumbot (192.168.{0,1}.240)
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

  your browser                 nipg1                    Mecanumbot
  ────────────                 ─────                    ──────────
  localhost:8080 ──ssh -L over the control tunnel──▶ web GUI :8080
  `robot web`                (nothing permanent; open while the command runs)
```

Both halves bind nipg1's loopback (sshd's `GatewayPorts` defaults to `no`), so
they meet on that one port and neither end needs the other's address.

## Everyday use

    robot                      # interactive shell on the robot, ROS 2 sourced
    robot topic list           # ros2 is implied
    robot topic echo /mecanumbot/scan
    robot status               # is the tunnel up? prints the topic count
    robot web                  # the robot's web GUI in your own browser
    robot grippers close       # helper: close grippers (neck held at 6.5)
    robot grippers open        # helper: open grippers (neutral 5.12)
    robot pub-accessory 6.5 6.83 3.36     # raw {n_pos, gl_pos, gr_pos}
    robot raw 'ros2 param list /mecanumbot/mecanumbot_joy_node'

### The web GUI from outside the lab

`robot web` forwards the robot's GUI to `http://localhost:8080` and holds it
open until you Ctrl-C. From a laptop rather than nipg1:

    MECANUMBOT_SSH_HOST=mecanumbot-jump robot web

**The node was always running; there was just no way to reach it.** The GUI
starts with the base launch — `use_web` defaults true and nothing in the T1 path
turns it off — and serves on the robot's `0.0.0.0:8080`. On the lab WiFi you
open `http://192.168.{0,1}.240:8080` and it is there. Through this directory's
tunnels it was not, because they carry exactly two ports across the boundary,
22 and 5555, and 8080 is neither. nipg1 has no route to the robot's LAN at all,
so the page simply did not answer — which looks exactly like the node having
failed to start, and is why it was read that way.

`robot web` adds nothing permanent. It opens an `ssh -L` over the control tunnel
that already exists, for as long as the command runs. That is on purpose, and
the alternative — a third systemd unit beside the control and data ones — was
rejected for two reasons:

- The control unit sets `ExitOnForwardFailure=yes`. Adding a second `-R` to it
  means a stale GUI forward left on nipg1 takes **SSH to the robot** down with
  it, and that tunnel is the lifeline you would need to fix it.
- The GUI has no login, and it can drive the robot and start behaviour trees. A
  permanent forward leaves that standing open on a shared login node's loopback
  for as long as the robot is powered. On demand, bound to your own loopback, it
  is open while you are looking at it and gone afterwards.

Pass a different local port if 8080 is taken: `robot web 8081`.

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
tunnel from a machine on the same WiFi, which the `mecanumbot-laptop` host alias
targets:

    ssh -R 8222:"$(./robot-addr.sh)":22 csengehubay@nipg1.inf.elte.hu

`robot-addr.sh` prints whichever of `192.168.0.240` / `192.168.1.240` answers,
so the command is the same in both rooms. Hard-coding one of them is the
mistake this script exists to prevent: the failure is `channel 2: open failed:
administratively prohibited`, which reads like a permissions problem on nipg1
and is actually a laptop on the other router.

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

If `~/.ssh/id_ed25519.pub` does not exist, make the pair first — there is
nothing to authorize otherwise:

    ssh-keygen -t ed25519 -N ""

Check it took — this must print `ok` and not prompt:

    ssh -o BatchMode=yes nipg1.inf.elte.hu true && echo ok

Two lines like these may appear alongside the `ok`, and are **not** a failure:

    hostfile_replace_entries: link .../known_hosts to .../known_hosts.old: No such file or directory
    update_known_hosts: hostfile_replace_entries failed for .../known_hosts: ...

Home is on the NAS, which does not support the hardlink ssh uses to rotate
`known_hosts`. It is the bookkeeping complaining, not the authentication, and it
only fires on the connection that adds a new host entry. `ok` is the part that
matters.

**Skipping this step is the most common way `run_deep3r_bridged.sh` fails**, and
it fails after the server has already started and bound, so the run looks
half-alive:

    server is listening on :5555
    bridged: robot -> nipg1.inf.elte.hu:5555 -> nipg4:5555
    csengehubay@nipg1.inf.elte.hu: Permission denied (publickey).
    --- reverse tunnel to nipg1.inf.elte.hu:5555 dropped (exit 255) ---

`Permission denied (publickey)` is this step and nothing else. The script names
the cause it read out of ssh's own output, so take that over guessing — in
particular it is *not* the stale-port case, and going to hunt a held forward on
nipg1 is a wasted trip. The Slurm allocation survives the failure; fix the key
and re-run the `srun` against the same job id.

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
- The robot's web GUI is **not** forwarded permanently, and should not be: it
  has no login, and anything that reaches port 8080 can drive the robot and
  start a behaviour tree. `robot web` opens it over the existing control tunnel
  on demand and binds your own loopback (`ssh -L` does that unless you add
  `-g` — do not). Close it when you are done by stopping the command.
- The deep3r tunnel no longer targets nipg36, so it no longer needs your
  unrestricted cluster key: pointed at nipg1, it reuses the robot's existing
  restricted key. That removes the shell-on-nipg36 exposure the previous version
  of this note warned about.

## Robot facts (captured 2026-08-27, addresses re-checked 2026-08-28)

- ROS 2 Humble; `ROS_DOMAIN_ID=19`; `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`.
- Workspace overlay: `~/mecanumbot_ws/install/setup.bash`.
- WiFi address `192.168.0.240` **or** `192.168.1.240` on `wlP1p1s0`, depending
  on which lab router it associated with (`.0.1` and `.1.1` respectively). Last
  octet reserved, so only the third octet moves.
- Accessory command: `/cmd_accessory_pos`, type `mecanumbot_msgs/msg/AccessMotorCmd`
  = `{float32 n_pos, float32 gl_pos, float32 gr_pos}`.
- Neck range 2.0–8.6; gripper range 1.6–8.54; neutral/"front" 5.12.
  Close = `gl 6.83 / gr 3.36` (mirrored); open = `gl 5.12 / gr 5.12`.
  Servo readback is on `/mecanumbot/opencr_state` (pos_* ×100 of the command).
