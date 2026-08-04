# RoboCamStreamProcessing

Video link between the robot's Jetson Orin Nano and the GPU server `nipg36`.
The Orin streams its webcam, the server decodes each frame and replies over the
same connection with whatever the current processor produced.

**Status: transport layer complete and measured.** The only processor today
reports image metadata — it answers "are frames actually arriving, and what
shape are they?". YOLO / MASt3R / VGGT plug in as additional processors without
the wire protocol changing. See [Adding a model](#adding-a-model).

```text
  Jetson Orin Nano                         nipg36 (10.128.17.196)
  ────────────────                         ──────────────────────
  webcam                                        ZeroMQ ROUTER :5555
    │                                                 │
    ▼                                                 ▼
  capture (V4L2 / GStreamer)                  IO thread — never blocks
    │                                          decode JPEG → numpy
    ▼                                                 │
  JPEG encode ──────► DEALER ───── LAN ──────►  FrameQueue (depth 2,
    │                    ▲                       drops oldest when full)
    │                    │                             │
  in-flight window ≤ 3   │                             ▼
    │                    │                       worker thread(s)
    ▼                    │                        processor.process()
  on_result(dict) ◄──────┴──────────────────────────── result
```

Two ideas run through the design:

**A stale frame is worse than no frame.** Buffers are deliberately tiny — depth
2 on the server, 3 in-flight on the client. When the server falls behind, frames
are dropped at the source rather than queued, so the robot always acts on
something recent. Latency degrades gracefully instead of the link appearing to
freeze.

**Every frame gets exactly one reply**, including dropped ones (`reason:
"dropped"`). The client sizes its in-flight window from outstanding replies, so
a silently discarded frame would slowly starve the sender.

---

## Server setup (nipg36)

Already done in this checkout, but re-runnable and idempotent:

```bash
cd ~/RoboCamStreamProcessing
./scripts/setup_server.sh
```

The system Python on nipg36 is 3.8, which is too old for current torch, and
there is no root access or usable Docker. So `uv` installs into `~/.local/bin`
and manages its own Python 3.11 in `.venv/`. Nothing touches the system.

Run it:

```bash
./scripts/run_server.sh                              # config/server.yaml
./scripts/run_server.sh --processor noop             # measure transport ceiling
./scripts/run_server.sh --bind tcp://0.0.0.0:5600 --log-level DEBUG
```

Startup prints the address the robot should dial:

```text
listening on tcp://0.0.0.0:5555 | processor=stats workers=1 | queue depth=2 drop=oldest
  robot can connect to tcp://10.128.17.196:5555
```

Leave it running across SSH disconnects with `tmux new -s robocam` (or
`nohup ./scripts/run_server.sh > server.log 2>&1 &`).

## Client setup (the Orin)

`client/robocam_client.py` is standalone — one file, no dependency on the
`robocam` package.

```bash
scp -P 10113 client/robocam_client.py orin:~/
ssh orin
pip3 install pyzmq numpy          # opencv ships with JetPack
./netcheck.sh 10.128.17.196 5555  # do this first, see Networking
python3 robocam_client.py --server tcp://10.128.17.196:5555
```

Useful flags:

```bash
--source synthetic          # test pattern, no camera needed — isolates link problems
--source gst-jpeg           # hardware JPEG via the Orin's NVJPEG block
--device /dev/video0        # or an index, or a full GStreamer pipeline
--width 1280 --height 720 --fps 30
--quality 85                # JPEG quality
--max-inflight 3            # frames awaiting a result before dropping at source
--print-results             # dump every result as JSON, one per line
```

LiDAR flags (the robot's LDS-02, streamed alongside the frames):

```bash
--lidar auto                # ros2 topic first, then the serial device; off by default
--lidar-port /dev/tb3_lidar # default: the udev symlink, not a raw ttyUSBn
--lidar-baud 115200         # the LDS-02 rate; LD06/LD19 are 230400
--lidar-topic /scan         # for --lidar ros2
--lidar-health-every 5      # seconds between progress reports, 0 to silence
--print-scans               # dump every scan result as JSON
```

**`--lidar` is `off` unless you ask for it.** The client streams camera only by
default, which is why a working `ld08_driver` on the robot does not by itself put
scans on the wire — the two are unrelated consumers of the same sensor.

`--lidar-port` defaults to `/dev/tb3_lidar` and falls back to `/dev/ld08_lidar`
— the two names the udev rule has used for the scanner's CP2102 bridge on this
fleet. It deliberately does **not** try `/dev/ttyUSB0` or `/dev/ttyUSB1`: those
numbers move between boots (the scanner is on ttyUSB1 today) and on this robot
ttyUSB0 is as likely to be the OpenCR board. Reading a motor controller as
ranges is worse than failing, so a raw device is only ever used if you name one.

Serial mode needs `pip3 install pyserial`.

IMU flags (the robot's OpenCR board, ~100 Hz of inertial samples sent in bursts
alongside the frames):

```bash
--imu auto                  # ros2 topic first, then the board; off by default
--imu-topic /opencr_state   # the mecanumbot IO node's own topic
--imu-msg auto              # opencr = mecanumbot_msgs/OpenCRState, imu = sensor_msgs/Imu
--imu-port /dev/opencr      # only for --imu serial
--imu-health-every 5        # seconds between progress reports, 0 to silence
--print-imu                 # dump every imu result as JSON
```

**`--imu auto` prefers ROS 2 for a reason the scanner does not have.** The
OpenCR's serial port is normally owned by `mecanumbot_io_node` — the node driving
the wheels — and two processes reading one tty do not each get the stream, they
split it, so both frame garbage. Subscribing to what that node already publishes
costs nothing; competing for its port costs motor control. The serial path is the
fallback for a robot without the node running, and it refuses to open a port
another process is holding unless you pass `--imu-allow-shared-port`.

Samples go out in bursts rather than one message each: at 100 Hz the JSON header
would cost more than the data, so a burst carries everything taken since the last
frame and the message rate stays at the camera's while the sample rate stays at
the sensor's. Bursts that the client's own queue had to discard are counted and
the count travels with the next one, so a gap shows up as a number rather than as
a rate that quietly came out low.

Or use it as a library:

```python
from robocam_client import RoboCamClient, OpenCVSource

def on_result(r):
    if r["ok"]:
        print(r["width"], r["height"], r["data"]["session_fps"], r["rtt_ms"])

client = RoboCamClient("tcp://10.128.17.196:5555", on_result=on_result)
client.run(OpenCVSource("0", 1280, 720, 30))
```

`--source gst-jpeg` is worth using once the camera works. OpenCV decodes the
camera's MJPEG to BGR and the client re-encodes it, which costs most of a CPU
core at 720p30 on an Orin Nano; the GStreamer path forwards the camera's JPEG
untouched. It needs `sudo apt install python3-gi gir1.2-gstreamer-1.0`.

## Verifying it works

Three checks, cheapest first:

1. **Is the port reachable?** Run `scripts/netcheck.sh` on the robot. It
   separates "wrong address / firewall / VPN" from "bug in my code", which is
   where most of the time otherwise goes.
2. **Does the link carry frames?** Run the client with `--source synthetic`. If
   this works and the camera does not, the problem is the camera.
3. **Are the images real?** The server writes `snapshots/latest.jpg` every 150
   frames. Open it through your sshfs mount — that catches a lens cap, an
   upside-down camera or swapped colour channels in one glance.

The client's status line tells you the rest:

```text
sent=121 ok=120 bad=0 skipped=0 | 30.1 fps out | rtt 36.3 ms | inflight=1
```

### When the LiDAR is silent

Silence from a scanner has several causes that look identical from outside —
nothing plugged in, nothing spinning, wrong baud, wrong device, another process
holding the port. They are told apart by *how far the data gets*, so the client
counts the stages and reports the furthest one reached rather than just
producing nothing:

```text
lidar: /dev/tb3_lidar — no scan yet after 5.0s. not one byte has arrived. The port
  opened, so the device node is real, but nothing is transmitting: check the
  scanner's power lead and that the rotor is actually spinning.

lidar: /dev/tb3_lidar — no scan yet after 5.0s. 67072 bytes arrived (13414 B/s) but
  not one valid LD08 packet was framed, reading at 230400 baud. An LDS-02 sends
  about 7050 B/s, so this is ~1.9x too much — the signature of reading faster than
  the device transmits. Try `--lidar-baud 115200`.

lidar: /dev/tb3_lidar — first revolution after 0.4s, 358/360 points (99% coverage)
lidar: /dev/tb3_lidar — 27 revolutions, 5.4/s, 359/360 points (100% coverage)
```

That middle case is worth dwelling on, because it is the one that looks like a
dead scanner and is not. The LDS-02 contains an LD08, which shares its *packet
format* with the LD06 and LD19 but not their baud: those run at 230400 and the
LD08 at **115200**. Point a 230400 reader at it and every bit gets sampled
twice — bytes arrive steadily, at roughly double the true rate, and not one of
them frames. The rate is the clue, so the client compares what it is receiving
against what an LDS-02 should send and names a baud to try rather than just
reporting the symptom.

A port that will not open says which fix applies — missing udev rule, missing
`dialout` group, or a busy port (usually `ld08_driver` already holding it) — and
lists the serial devices that *do* exist.

On the ROS 2 path, `--lidar auto` requires an actual publisher before it settles
on ROS: subscribing to a topic nobody publishes succeeds happily, and without
that check the client would sit silent on `/scan` forever instead of falling
through to the serial device. It also distinguishes "nothing publishes `/scan`"
(driver down) from "a publisher exists but sends nothing" (driver up, scanner
not producing).

### What the IMU can and cannot tell you

Worth being blunt about, because the temptation to treat it as a pose source is
strong and the failure is silent:

* **Attitude is honest.** Roll and pitch are observable — gravity is a permanent
  reference — so "the robot is tipping" or "this ramp is 8°" is real.
* **Yaw is not.** Nothing in a gyro fixes an absolute heading, so `yaw_deg` has an
  arbitrary origin and drifts degrees per minute. Use `yaw_rate_dps` for control.
  The magnetometer would fix it in principle and does not in practice: it sits
  centimetres from four motors whose field swamps the Earth's.
* **Position is not, at all.** Double-integrating this accelerometer gives metres
  of error in seconds. Wheel odometry and the LiDAR are the position sensors.

Axes follow the ROS body convention the OpenCR firmware uses: **x forward, y
left, z up**, so a level robot at rest reads `az ≈ +9.81` and nothing else. That
is the assumption most likely to be wrong on a rebuilt robot, so the snapshot
draws an attitude disc in the corner opposite the scan plot: **a robot standing on
a flat floor must show a level horizon.** A permanent 90° tilt means the board is
mounted on a different axis and every attitude number is rotated with it — the
same class of mistake as a wrong `mount_yaw_deg`, caught the same way, by looking.

Two summary fields exist to catch the other silent failure. `gravity_ok` goes
false when the mean specific force is not near 9.81, which is what a units
mistake or a dead axis looks like; `rate_hz` is measured from the samples' own
offsets, so it disagreeing with the board's declared 100 Hz means samples are
being lost between the sensor and the socket.

`skipped` climbing means the server cannot keep up and the client is dropping at
the source. `bad` climbing with `reason: "dropped"` means the server's queue is
evicting. Both are the intended behaviour under load, not errors — but they tell
you the pipeline is saturated.

## Measured on nipg36

720p30 synthetic stream, `stats` processor, client and server both on nipg36
(so the client's JPEG encoding is competing for the same CPU — over the real LAN
the server-side numbers hold and RTT depends on your network):

| Measurement | Value |
| --- | --- |
| Sustained rate | 29.9 fps, 0 dropped |
| Bandwidth | 10.0 Mbit/s at quality 85 (≈47× compression) |
| Decode | 5.7 ms mean, 9.0 ms p95 |
| Processor (`stats`) | 2.0 ms |
| Queue wait | 0.5 ms |
| **Server total** | **8.3 ms mean, 11.3 ms p95** |
| Round trip | 29.0 ms mean |

Decode dominates the server budget. If that ever matters, the fixes in order are
`--processor noop` to confirm it is really decode, then a hardware JPEG decoder
(`nvjpeg` via DALI or torchvision) which also lands the frame directly in GPU
memory where the models want it anyway.

---

## Protocol

ZeroMQ, robot = DEALER, server = ROUTER. Every message is two frames:
`[header_json, payload]`. The header is small JSON — inspectable with tcpdump,
and negligible next to a JPEG. Only `frame` messages carry a payload.

| Message | Direction | Meaning |
| --- | --- | --- |
| `hello` | → server | opens a session, declares codec and geometry |
| `welcome` | → robot | accepted (or rejected, with a reason) |
| `frame` | → server | one encoded image |
| `result` | → robot | exactly one per frame |
| `ping` / `pong` | both | liveness |
| `bye` | both | graceful close |
| `error` | → robot | malformed request |

A `result` header:

```json
{"type":"result","seq":9,"ok":true,"reason":"ok",
 "width":640,"height":480,"channels":3,"dtype":"uint8","nbytes":921600,
 "payload_bytes":19410,"codec":"jpeg",
 "decode_ms":1.811,"process_ms":0.79,"queue_ms":0.155,"server_ms":22.014,
 "processor":"stats",
 "data":{"received":true,"shape":[480,640,3],"compression_ratio":47.48,
         "frames_seen":370,"session_fps":26.39,"since_prev_ms":200.08,
         "mean":125.54,"std":64.38,"looks_blank":false},
 "t_capture_ns":3001547970250763,"t_send_ns":3001547971818394}
```

`width`/`height` are what the server **actually decoded**, not what the client
claimed — that is how you catch a camera that silently renegotiated its format.
`data` is the per-processor payload and is where detections and geometry will
appear; everything outside it stays the same when models are added.

`reason` is one of `ok`, `dropped`, `decode_failed`, `processor_failed`,
`unsupported_codec`. Only `ok` means the frame was processed.

**Clocks are never compared across machines.** The Orin and the server have
unrelated monotonic clocks, so `t_capture_ns` and `t_send_ns` are echoed back
untouched and the client computes `rtt_ms` against its own clock. `server_ms` is
measured entirely on the server. There is no clock sync anywhere and none is
needed.

### Codecs

`jpeg` (default), `raw_bgr`, and `h264` (needs `uv pip install av` on the
server). JPEG and raw are stateless, so a dropped frame costs nothing and
reconnects are instant. H.264 is ~10× smaller on the wire but stateful: after a
loss the decoder produces garbage until the next keyframe. Use JPEG on the LAN;
consider H.264 only if you end up streaming over the VPN from home.

## Configuration

`config/server.yaml`, fully commented. Unknown keys are a startup error rather
than a silent no-op, so a typo fails immediately instead of quietly doing
nothing for hours.

The two that matter most:

- `queue.max_depth` (default 2) — raise only if you move to batched inference.
- `queue.drop_policy` (default `oldest`) — `oldest` keeps latency low. Switch to
  `newest` if a model needs strictly consecutive frames, which MASt3R and VGGT
  may well want.

## Networking

The server binds `0.0.0.0:5555`; the robot connects to `10.128.17.196:5555` on
the university LAN. If `scripts/netcheck.sh` fails from the robot and the server
is definitely running, the usual causes in order are: wrong address (check
`hostname -I` on nipg36), you are off the university network, or a host firewall
(needs an admin — `sudo ufw status`).

**From home**, the LAN address is unreachable. Either use the university VPN, or
tunnel over the SSH port that already works:

```bash
# On the robot (or your laptop): forward local 5555 to the server's 5555.
ssh -p 10113 -N -L 5555:localhost:5555 csengehubay@nipg36.inf.elte.hu
# Then point the client at the local end of the tunnel:
python3 robocam_client.py --server tcp://127.0.0.1:5555
```

Expect to lower `--fps` and `--quality` over the VPN — 10 Mbit/s at 720p30 is
comfortable on the LAN, less so on a home uplink.

---

## Adding a model

Write a `Processor` and register it. Nothing else changes — not the protocol,
not the client.

```python
# robocam/processors/yolo.py
from .base import Frame, Processor

class YoloProcessor(Processor):
    name = "yolo"

    def __init__(self, weights="yolo11n.pt", device="cuda:0", conf=0.25, imgsz=640, **kw):
        super().__init__(**kw)
        self.weights, self.device, self.conf, self.imgsz = weights, device, conf, imgsz

    def setup(self):
        # Load in setup(), not __init__: this runs on the worker thread, so the
        # CUDA context belongs to the thread that will use it.
        from ultralytics import YOLO
        self.model = YOLO(self.weights)
        self.model.to(self.device)
        # Warm up, so the first real frame does not pay cuDNN autotuning.
        import numpy as np
        self.model.predict(np.zeros((self.imgsz, self.imgsz, 3), "uint8"), verbose=False)

    def process(self, frame: Frame):
        r = self.model.predict(frame.image, conf=self.conf, imgsz=self.imgsz,
                               device=self.device, verbose=False)[0]
        return {"detections": [
            {"cls": int(c), "name": r.names[int(c)], "conf": round(float(p), 3),
             "xyxy": [round(v, 1) for v in b]}
            for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                               r.boxes.conf.tolist())
        ]}
```

Then in `robocam/processors/__init__.py`:

```python
from .yolo import YoloProcessor
register("yolo", YoloProcessor)
```

and in `config/server.yaml`:

```yaml
processor:
  name: yolo
  workers: 1
  options:
    weights: yolo11n.pt
    conf: 0.25
```

The contract: `setup()` runs once on the worker thread — load weights and warm
up there. `process()` returns a JSON-serialisable dict, or raises, in which case
that one frame comes back `ok: false` and the stream carries on. Each worker
gets its own processor instance, so one CUDA context and one set of weights per
thread; a stateful processor never sees interleaved frames.

### What this hardware will and will not do

Worth knowing before you install anything:

- **nipg36's GPUs are TITAN RTX, compute capability 7.5 (Turing).** No bf16 and
  no FlashAttention — both need sm_80+. VGGT and MASt3R must run in **fp16**
  with PyTorch's SDPA fallback, not `flash_attn`. Expect to pass
  `dtype=torch.float16` and to skip any `flash-attn` install step in their
  READMEs; it will not build usefully here.
- **GPU 0 is usually occupied by another user** (~20 GB of 24 GB in use).
  `run_server.sh` defaults to `CUDA_VISIBLE_DEVICES=1`. Check with `nvidia-smi`
  before assuming you have memory.
- **Install torch as cu121**, matching driver 535:
  `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
- The venv currently has **numpy 2.x**. Some older ultralytics/MASt3R pins want
  numpy <2; if you hit that, pin numpy in this venv rather than rebuilding it.

### Notes for the multi-view models

`stats` and `yolo` are per-frame, but MASt3R and VGGT need several frames at
once. The processor interface already supports this — keep a keyframe buffer in
the instance:

```python
def process(self, frame):
    self.buffer.append(frame.image)
    if len(self.buffer) < self.window:
        return {"status": "buffering", "have": len(self.buffer)}
    ...
```

Two things to plan for. Set `queue.drop_policy: newest` so the buffer sees
consecutive frames rather than a sequence with holes. And return something on
every call — returning `{"status": "buffering"}` keeps the client's flow control
healthy while the window fills.

Point clouds are large; do not put a full VGGT depth map in `result.data` at 30
fps. Either downsample it, or return a handle and fetch it out of band.

---

## Layout

```text
robocam/                 server package
  wire.py                protocol: message types, constructors, framing
  server.py              IO loop, sessions, CLI entry point
  pipeline.py            FrameQueue (drop-on-full) and WorkerPool
  decode.py              JPEG / raw / H.264 → numpy, per session
  config.py              YAML → validated dataclasses
  snapshot.py            periodic frame dumps, off the IO thread
  waker.py               lets workers interrupt poll() — see below
  processors/
    base.py              Processor interface and Frame
    stats.py             default: geometry, rate, brightness
    noop.py              transport-ceiling benchmark
client/robocam_client.py standalone, deploy to the Orin
config/server.yaml
scripts/                 setup_server.sh, run_server.sh, netcheck.sh
tests/                   32 tests, including real sockets end to end
```

`waker.py` earns its place: without it a finished result waits for the current
`poll()` to time out before being sent, which measured 22 ms of an 8 ms job.
Workers now push a byte down an inproc ZeroMQ pipe that the IO loop polls, and
the result goes out immediately.

## Tests

```bash
./.venv/bin/python -m pytest tests/ -q     # 32 passed
```

`tests/test_loopback.py` runs a real server on a real TCP socket and talks to it
both with a bare DEALER (to exercise the protocol directly) and with the actual
deployable client file. Those are the tests that catch a wire-format mistake, so
they deliberately do not mock the transport.
