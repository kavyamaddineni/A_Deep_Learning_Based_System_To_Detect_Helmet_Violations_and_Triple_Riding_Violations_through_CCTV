import cv2       # control + shift + p
from ultralytics import YOLO

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = box1[:4]  # Ensure only the first 4 values are unpacked
    x1_b, y1_b, x2_b, y2_b = box2[:4]  # Same here for box2
    
    xi1 = max(x1, x1_b)
    yi1 = max(y1, y1_b)
    xi2 = min(x2, x2_b)
    yi2 = min(y2, y2_b)
    
    # If there is no overlap, return IoU as 0
    if xi2 < xi1 or yi2 < yi1:
        return 0.0
    
    # Calculate the area of intersection
    intersection_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_b - x1_b) * (y2_b - y2_b)

    # Calculate the area of union
    union_area = box1_area + box2_area - intersection_area

    # Return the IoU
    return intersection_area / union_area

# Load YOLO models
helmet_model = YOLO('runs/detect/HelmetVioltionDetection-YOLOV8m/weights/best.pt')
triple_riding_model = YOLO('yolov8m.pt')

def helmet_and_triple_riding_detection(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return

    # Perform helmet detection
    helmet_results = helmet_model.predict(image, conf=0.4)
    helmet_violation_count = 0

    # Annotated image for results
    annotated_image = helmet_results[0].plot()

    # Check for helmet violations
    if helmet_results[0].boxes is not None:
        for box in helmet_results[0].boxes:
            # Extract coordinates for the bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Ensure the coordinates are integers
            
            # Check if the detected class is helmet (class 1)
            if int(box.cls[0]) == 1:  # Check for helmet class
                cv2.putText(annotated_image, "", (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
            if int(box.cls[0]) == 2:
                cv2.putText(annotated_image, "No Helmet Detected", (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

    # Add helmet violation status text to the image
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    line_type = cv2.LINE_AA
    font_color_red = (0, 0, 255)  # Red
    font_color_green = (0, 255, 0)  # Green
    # helmet_status_text = "Helmet Detected" if not Helmet_Not_Detected else "No Helmet Detected"
    # cv2.putText(annotated_image, helmet_status_text, (20, 30), font_face, font_scale,
    #             font_color_green if not Helmet_Not_Detected else font_color_red,
    #             font_thickness, line_type)
    # cv2.rectangle(annotated_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    # violation_text = f"Helmet Violation"
    # cv2.putText(annotated_image, violation_text, (box[0], box[1] - 10),
    #                     font_face, 0.5, font_color_red, 1)

    # Perform triple riding detection
    triple_riding_results = triple_riding_model.predict(image, conf=0.4)
    bikes = []
    people = []
    triple_riding_violation_count = 0

    # Extract bikes and people detections (only consider person and motorcycle classes)
    for box in triple_riding_results[0].boxes:
        class_id = int(box.cls)  # Get class ID
        if class_id == 0:  # Person class
            people.append(box.xyxy[0].cpu().numpy())  # Person bounding box (xyxy)
        elif class_id == 3:  # Bike class
            bikes.append(box.xyxy[0].cpu().numpy())  # Bike bounding box (xyxy)

        # Draw bounding boxes and labels
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = box.conf.item()
        label = f"{triple_riding_model.names[class_id]}: {confidence:.2f}"
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(annotated_image, label, (x1, y1 - 10), font_face, 0.4, (0, 255, 0), 1)

    # Highlight bikes with triple riding violations
    violation_detected = False
    for bike in bikes:
        bike_box = bike.astype(int)
        people_count = 0
        for person in people:
            person_box = person.astype(int)
            # Calculate IoU for better precision in detecting persons on bikes
            iou = calculate_iou(bike_box, person_box)
            if iou > 0.2:  # Use IoU threshold for detecting person on bike
                people_count += 1

        # Highlight violation
        if people_count > 2:  # Triple riding threshold
            violation_detected = True
            triple_riding_violation_count += 1
            cv2.rectangle(annotated_image, (bike_box[0], bike_box[1]), (bike_box[2], bike_box[3]), (0, 0, 255), 2)
            violation_text = f"Triple Riding Violation"
            cv2.putText(annotated_image, violation_text, (bike_box[0], bike_box[1] - 10),
                        font_face, 0.5, font_color_red, 1)

    # Add triple riding violation status
    if violation_detected:
        cv2.putText(annotated_image, "Triple Riding Violation Detected", (20, 50), font_face, font_scale,
                    font_color_red, font_thickness, line_type)
    else:
        cv2.putText(annotated_image, "No Triple Riding Violation", (20, 50), font_face, font_scale,
                    font_color_green, font_thickness, line_type)

    # Display the final annotated image
    cv2.imshow("Helmet and Triple Riding Detection", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = "Files\img11.jpg"  # Replace with the path to your image
helmet_and_triple_riding_detection(image_path)