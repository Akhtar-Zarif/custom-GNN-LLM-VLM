import argparse
import socket
import json
import time
import math


def main():
    parser = argparse.ArgumentParser(description="Simulated OAK-1 KLT Tracker UDP Streamer")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Target UDP server IP")
    parser.add_argument("--port", type=int, default=5005, help="Target UDP server port")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between packets")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print("Starting Simulated OAK-1 KLT Tracker Stream...")
    print(f"Sending tracking telemetry to {args.ip}:{args.port}...")

    try:
        node_id = 0
        while True:
            t = time.time()
            pos_x = 5.0 * math.cos(t * 0.5)
            pos_y = 5.0 * math.sin(t * 0.5)
            vx = -2.5 * math.sin(t * 0.5)
            vy = 2.5 * math.cos(t * 0.5)
            confidence = 0.95

            packet = {
                "node_id": node_id,
                "pos_x": round(pos_x, 4),
                "pos_y": round(pos_y, 4),
                "vx": round(vx, 4),
                "vy": round(vy, 4),
                "confidence": confidence
            }

            sock.sendto(json.dumps(packet).encode('utf-8'), (args.ip, args.port))
            print(f"Sent Node {node_id} -> Pos: ({pos_x:.2f}, {pos_y:.2f}), Vel: ({vx:.2f}, {vy:.2f})")

            node_id += 1
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
        sock.close()


if __name__ == "__main__":
    main()
