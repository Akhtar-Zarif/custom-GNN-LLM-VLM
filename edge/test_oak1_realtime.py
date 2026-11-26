import argparse
import cv2
import depthai as dai
from ultralytics import YOLO
import time
import json
import socket
from collections import deque


def main():
    parser = argparse.ArgumentParser(description="OAK-1 Drone Detection with Trajectory Trail & GNN Telemetry Stream")
    parser.add_argument("--model-path", type=str, default="checkpoints/best.pt", help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Target UDP server IP")
    parser.add_argument("--port", type=int, default=5005, help="Target UDP server port")
    parser.add_argument("--trail-length", type=int, default=30, help="Trajectory trail points")
    args = parser.parse_args()

    print(f"Loading YOLO model from: {args.model_path}")
    model = YOLO(args.model_path)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"UDP Telemetry Streamer active -> Target: {args.ip}:{args.port}")

    pipeline = dai.Pipeline()

    camRgb = pipeline.createColorCamera()
    camRgb.setPreviewSize(640, 640)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    xoutRgb = pipeline.createXLinkOut()
    xoutRgb.setStreamName("rgb")
    camRgb.preview.link(xoutRgb.input)

    print("Connecting to OAK-1 camera...")
    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        print("OAK-1 Camera Feed Active! Press 'q' in the window to exit.")

        prev_time = 0
        prev_box_center = None
        node_id = 0
        center_trail = deque(maxlen=args.trail_length)

        while True:
            inRgb = qRgb.get()
            frame = inRgb.getCvFrame()

            results = model.predict(source=frame, conf=args.conf, iou=args.iou, verbose=False)

            current_box_center = None
            vx, vy = 0.0, 0.0
            confidence = 0.0

            for box in results[0].boxes:
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()

                center_x = float((xyxy[0] + xyxy[2]) / 2.0)
                center_y = float((xyxy[1] + xyxy[3]) / 2.0)
                current_box_center = (center_x, center_y)
                break

            curr_time = time.time()
            dt = curr_time - prev_time if prev_time > 0 else 0.033

            if current_box_center and prev_box_center and dt > 0:
                vx = (current_box_center[0] - prev_box_center[0]) / dt
                vy = (current_box_center[1] - prev_box_center[1]) / dt

            if current_box_center:
                center_trail.append((int(current_box_center[0]), int(current_box_center[1])))
                packet = {
                    "node_id": node_id,
                    "pos_x": round(current_box_center[0], 2),
                    "pos_y": round(current_box_center[1], 2),
                    "vx": round(vx, 2),
                    "vy": round(vy, 2),
                    "confidence": round(confidence, 2),
                    "timestamp": round(curr_time, 3)
                }
                sock.sendto(json.dumps(packet).encode('utf-8'), (args.ip, args.port))
                node_id += 1
                prev_box_center = current_box_center

            annotated_frame = results[0].plot()

            for i in range(1, len(center_trail)):
                cv2.line(annotated_frame, center_trail[i - 1], center_trail[i], (0, 165, 255), 2)

            fps = 1 / dt if dt > 0 else 0
            prev_time = curr_time

            cv2.putText(annotated_frame, f"FPS: {fps:.1f} | Nodes: {node_id}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("OAK-1 Real-time Drone Detection & GNN Stream", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    sock.close()
    print("Camera stream closed and UDP socket released")


if __name__ == "__main__":
    main()
