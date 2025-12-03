import time
import cv2
import torch
from ultralytics import YOLO


class WebcamObjectDetector:
    """Real-time object detection using YOLOv8 and webcam."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.5):
        """
        Initialize the object detector.

        Args:
            model_name: YOLOv8 model name (default: yolov8n.pt)
            confidence: Confidence threshold for detections (default: 0.5)
        """
        self.confidence = confidence
        self.camera = None
        self.fps = 0.0
        self.smoothing_factor = 0.95
        self.frame_time = time.time()

        # Check CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Load YOLOv8 model
        print(f"Loading {model_name}...")
        self.model = YOLO(model_name)
        self.model.to(self.device)
        print("Model loaded successfully!")

    def initialize_camera(self, camera_id: int = 0) -> bool:
        """
        Initialize the webcam.

        Args:
            camera_id: Camera device ID (default: 0 for laptop webcam)

        Returns:
            True if camera initialized successfully, False otherwise
        """
        self.camera = cv2.VideoCapture(camera_id)

        if not self.camera.isOpened():
            print("Error: Could not open webcam")
            return False

        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

        print("Webcam initialized successfully!")
        return True

    def run_inference(self, frame):
        """
        Run YOLOv8 inference on a frame.

        Args:
            frame: Input frame from webcam

        Returns:
            Detection results from YOLO model
        """
        results = self.model(
            frame,
            device=self.device,
            conf=self.confidence,
            verbose=False
        )
        return results[0]

    def draw_detections(self, frame, results):
        """
        Draw bounding boxes and labels on frame.

        Args:
            frame: Input frame
            results: Detection results from YOLO

        Returns:
            Frame with drawn detections
        """
        boxes = results.boxes

        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Get class ID and confidence
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Get class name
            class_name = self.model.names[class_id]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Create label
            label = f"{class_name}: {confidence:.2f}"

            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                (0, 255, 0),
                -1
            )

            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )

        return frame

    def calculate_fps(self) -> float:
        """
        Calculate and smooth FPS.

        Returns:
            Smoothed FPS value
        """
        current_time = time.time()
        elapsed = current_time - self.frame_time
        current_fps = 1.0 / elapsed if elapsed > 0 else 0

        # Exponential moving average for smoothing
        self.fps = (self.smoothing_factor * self.fps +
                   (1 - self.smoothing_factor) * current_fps)

        self.frame_time = current_time
        return self.fps

    def draw_fps(self, frame):
        """
        Draw FPS counter on frame.

        Args:
            frame: Input frame

        Returns:
            Frame with FPS overlay
        """
        fps_text = f"FPS: {self.fps:.1f}"

        # Draw background rectangle
        cv2.rectangle(frame, (5, 5), (120, 35), (0, 0, 0), -1)

        # Draw FPS text
        cv2.putText(
            frame,
            fps_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        return frame

    def run(self):
        """Main detection loop."""
        if not self.initialize_camera():
            return

        print("\nStarting real-time object detection...")
        print("Press ESC or 'q' to exit\n")

        # Warmup inference
        print("Warming up model...")
        ret, frame = self.camera.read()
        if ret:
            _ = self.run_inference(frame)
            _ = self.run_inference(frame)
        print("Warmup complete!\n")

        try:
            while True:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break

                # Run inference
                results = self.run_inference(frame)

                # Draw detections
                frame = self.draw_detections(frame, results)

                # Calculate and draw FPS
                self.calculate_fps()
                frame = self.draw_fps(frame)

                # Display frame
                cv2.imshow("YOLOv8 Real-Time Object Detection", frame)

                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or 'q'
                    print("\nExiting...")
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources."""
        if self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()
        print("Resources released. Goodbye!")


def main():
    """Entry point for the application."""
    detector = WebcamObjectDetector(
        model_name="yolov8n.pt",
        confidence=0.5
    )
    detector.run()


if __name__ == "__main__":
    main()
