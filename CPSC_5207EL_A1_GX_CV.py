import cv2
import numpy as np


def estimate_distance(pixel_size, real_size_cm=7.5, frame_width=640):
    """
    Estimates distance to AprilTag without camera calibration.

    Args:
        pixel_size: Average edge length of the tag in pixels
        real_size_cm: Real physical size of the tag in cm (default: 7.5 cm)
        frame_width: Frame width in pixels (default: 640)

    Returns:
        float: Estimated distance in centimeters

    Note:
        This uses an approximate focal length based on typical webcam FOV (~60-70 degrees).
        For 640x480 resolution, focal length ≈ frame_width / (2 * tan(FOV/2))
        Assuming 60° horizontal FOV: f ≈ 640 / (2 * tan(30°)) ≈ 554 pixels
    """
    # Approximate focal length for typical webcam (in pixels)
    # This assumes ~60 degree horizontal field of view
    focal_length_px = frame_width / (2 * np.tan(np.radians(30)))

    # Distance = (focal_length * real_size) / pixel_size
    # Prevent division by zero
    if pixel_size > 0:
        distance_cm = (focal_length_px * real_size_cm) / pixel_size
        return distance_cm
    return None


def trackSign_CV(frame, detector, tag_to_action):
    """
    Detects AprilTags (36h11 family) in the frame and returns navigation decisions.
    Estimates distance to each detected tag.

    Args:
        frame: Input BGR frame from camera
        detector: cv2.aruco.ArucoDetector configured for AprilTag 36h11
        tag_to_action: Dictionary mapping tag IDs to navigation actions

    Returns:
        tuple: (annotated_frame, navigation_decision, detection_data)
            - annotated_frame: Frame with drawn detection results
            - navigation_decision: String with navigation command or None
            - detection_data: List of dictionaries containing detection details
    """
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags
    corners, ids, rejected = detector.detectMarkers(gray)

    navigation_decision = None
    detection_data = []
    annotated_frame = frame.copy()

    if ids is not None and len(ids) > 0:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(annotated_frame, corners, ids)

        for i, tag_id in enumerate(ids.flatten()):
            # Get corner points for this tag
            corner_points = corners[i][0]

            # Calculate center of the tag
            center_x = int(np.mean(corner_points[:, 0]))
            center_y = int(np.mean(corner_points[:, 1]))

            # Calculate tag size (average edge length for stability check)
            edge_lengths = []
            for j in range(4):
                p1 = corner_points[j]
                p2 = corner_points[(j + 1) % 4]
                length = np.linalg.norm(p2 - p1)
                edge_lengths.append(length)
            avg_edge_length = np.mean(edge_lengths)

            # Estimate distance to the tag
            distance_cm = estimate_distance(
                avg_edge_length, real_size_cm=7.5, frame_width=640
            )

            # Store detection data
            detection_info = {
                "id": int(tag_id),
                "center": (center_x, center_y),
                "corners": corner_points,
                "edge_length": avg_edge_length,
                "distance_cm": distance_cm,
            }
            detection_data.append(detection_info)

            # Get navigation decision for this tag
            action = tag_to_action.get(int(tag_id), "Unknown Tag")

            # Use the first detected tag for navigation (can be modified for priority logic)
            if navigation_decision is None:
                navigation_decision = action

            # Draw center point
            cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # Display tag ID and action
            text_y_offset = -30
            cv2.putText(
                annotated_frame,
                f"ID: {tag_id}",
                (center_x - 30, center_y + text_y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Action: {action}",
                (center_x - 30, center_y + text_y_offset - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Display estimated distance
            if distance_cm:
                cv2.putText(
                    annotated_frame,
                    f"Dist: {distance_cm:.1f} cm",
                    (center_x - 30, center_y + text_y_offset - 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    2,
                )

    # Display detection count and current navigation decision
    info_text = f"Tags detected: {len(ids) if ids is not None else 0}"
    cv2.putText(
        annotated_frame,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    if navigation_decision:
        cv2.putText(
            annotated_frame,
            f"Navigation: {navigation_decision}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    return annotated_frame, navigation_decision, detection_data


def main():
    """
    Opens the camera and displays live feedback using OpenCV.
    Detects AprilTags and provides navigation decisions.
    Press 'q' to quit the camera feed.
    """
    # Define tag ID to navigation action mapping
    # Modify these IDs and actions based on your specific signs
    tag_to_action = {
        0: "NO_ENTRY",
        1: "DEAD_END",
        2: "RIGHT",
        3: "LEFT",
        4: "FORWARD",
        5: "STOP",
    }

    # Initialize the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    # Set frame size to 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Initialize AprilTag detector (36h11 family)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    print("Camera opened successfully!")
    print("AprilTag 36h11 detector initialized!")
    print("Tag to Action Mapping:")
    for tag_id, action in tag_to_action.items():
        print(f"  ID {tag_id}: {action}")
    print("Press 'q' to quit")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break

        # Detect AprilTags and get navigation decision
        annotated_frame, navigation_decision, detection_data = trackSign_CV(
            frame, detector, tag_to_action
        )

        # Display the annotated frame
        cv2.imshow("AprilTag Navigation System", annotated_frame)

        # Wait for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed successfully!")


if __name__ == "__main__":
    main()
