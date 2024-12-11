from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load the YOLOv11n model
model = YOLO("yolo11n.pt")

# Traffic light status based on time
def get_traffic_light_status(current_time, red_duration, green_duration):
    cycle_duration = red_duration + green_duration  # Total cycle duration
    time_in_cycle = current_time % cycle_duration  # Time within the current cycle
    if time_in_cycle < red_duration:
        return "RED"
    else:
        return "GREEN"

# Function to draw angled boundary lines and update line color based on signal
def draw_boundary_lines(frame, signal):
    height, width = frame.shape[:2]

    # Define the starting positions for the lines
    sidewalk_line_x = int(width * 0.5) - 100  # Left starting point
    crosswalk_line_x = int(width * 0.5)  # Right starting point

    # Line rotation angles (in degrees)
    left_angle = -50  # Rotate left line clockwise
    right_angle = 50  # Rotate right line counterclockwise

    # Calculate tangent values for the angles
    tan_left = np.tan(np.radians(left_angle))
    tan_right = np.tan(np.radians(right_angle))

    # Left line: Rotate rightward
    left_x1, left_y1 = sidewalk_line_x, 0
    left_x2 = int(sidewalk_line_x + tan_left * height)
    left_y2 = height

    # Right line: Rotate leftward
    right_x1, right_y1 = crosswalk_line_x, 0
    right_x2 = int(crosswalk_line_x + tan_right * height)
    right_y2 = height

    # Set line color based on signal
    if signal == "RED":
        line_color = (0, 0, 255)  # Red
    elif signal == "GREEN":
        line_color = (0, 255, 0)  # Green

    # Draw the lines with updated color
    cv2.line(frame, (left_x1, left_y1), (left_x2, left_y2), line_color, 3)  # Left line
    cv2.line(frame, (right_x1, right_y1), (right_x2, right_y2), line_color, 3)  # Right line

    return (left_x1, left_x2), (right_x1, right_x2)

# Function to draw Korean text using Pillow (adjusted for larger and clearer text)
def draw_text_korean(img, text, position, font_path="NanumGothic.ttf", font_size=50, color=(255, 255, 255)):
    # Convert OpenCV BGR to RGB for Pillow
    color_rgb = (color[2], color[1], color[0])  # BGR to RGB
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert OpenCV image to PIL
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color_rgb)  # Use RGB color for Pillow
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  # Convert PIL image back to OpenCV

# Process video
video_path = "CrossWalk_CCTV_Original02.mp4"  # Path to your video file
cap = cv2.VideoCapture(video_path)

# Initialize start time for traffic light control
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get current time and traffic light status
    current_time = int(time.time() - start_time)
    signal = get_traffic_light_status(current_time, red_duration=4, green_duration=7)

    # Draw angled boundary lines with updated line color
    left_line, right_line = draw_boundary_lines(frame, signal)

    # Display traffic light status on the screen
    if signal == "RED":
        frame = draw_text_korean(frame, "빨간불", (50, 50), font_size=50, color=(0, 0, 255))
    elif signal == "GREEN":
        frame = draw_text_korean(frame, "초록불", (50, 50), font_size=50, color=(0, 255, 0))

    # Run YOLO detection
    results = model.predict(frame)

    # Filter only "person" detections
    for box in results[0].boxes:
        cls = int(box.cls[0])  # Class ID
        if cls == 0:  # Assuming "person" is class ID 0
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Check if the person's center is within the crosswalk
            person_center_x = (x1 + x2) // 2
            text_font_size = 50  # Adjust the font size for better visibility

            # Display "인도에 머무르세요!" only if the person is outside the crosswalk and signal is RED
            if left_line[1] < person_center_x < right_line[1]:
                if signal == "GREEN":
                    frame = draw_text_korean(frame, "건너세요!", (x1, y1 - 50), font_size=text_font_size, color=(0, 255, 0))
                elif signal == "RED":
                    frame = draw_text_korean(frame, "위험합니다!", (x1, y1 - 50), font_size=text_font_size, color=(0, 0, 255))

    # Display the video frame with annotations
    cv2.imshow("Crosswalk Assistance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

