'''
    Dan Shafman
    Refactored by Brandon Wood
    FRC 334
    2/16/18
    Tape Detection
'''
import numpy as np
import cv2

class Tape:
    def __init__(self):
        # Storage
        self.LOWER_BOUNDS = np.array([38, 120, 117])
        self.UPPER_BOUNDS = np.array([117, 255, 255])
        
        self.NUM_RECTANLGES = 2


def init_table(server = "10.3.34.50"):
    # Initialize network tables
    logging.basicConfig(level=logging.DEBUG)
    NetworkTables.initialize(server)
    VISION_TABLE = NetworkTables.getTable('vision')

    return VISION_TABLE

VISION_TABLE = init_table("10.3.34.50")

def find_distance(contours):
    perimeter_a = cv2.arcLength(contours[0], True)
    perimeter_b = cv2.arcLength(contours[1], True)
    total_perim = perimeter_a + perimeter_b

    # Distance to tape from camera as a function of perimeter of rectangle contours
    distance = 46569 * (total_perim ** -0.995)
    return distance

def find_midpoint(points):
    # Find midpoint of multiple points
    x_total = 0
    y_total = 0
    for point in points:
        x_total += point[0]
        y_total += point[1]
    x_total /= len(points)
    y_total /= len(points)
    return x_total, y_total

def draw_descriptors(img, text_to_write, circles_to_write):
    # Function to draw descriptors onto "actual" image, such as text and circles
    # Draw text descriptors
    text_location_y = 50
    for text in text_to_write:
        cv2.putText(img, text + str(distance_from_center),
                    (20, text_location_y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
        text_location_y += 50

    # Draw circle descriptors
    for circle in circles_to_write:
        cv2.circle(img, circle, 7, (255, 255, 0), 2)

    return img

    def find_distance(contours):
        perimeter_a = cv2.arcLength(contours[0], True)
        perimeter_b = cv2.arcLength(contours[1], True)
        total_perim = perimeter_a + perimeter_b

        # Distance to tape from camera as a function of perimeter of rectangle contours
        distance = 46569 * (total_perim ** -0.995)
        return distance


    def find_midpoint(points):
        # Find midpoint of multiple points
        x_total = 0
        y_total = 0
        for point in points:
            x_total += point[0]
            y_total += point[1]
        x_total /= len(points)
        y_total /= len(points)
        return x_total, y_total

    def find_tape(self, frame):
        # Take image and change colorspace
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold values 
        mask = cv2.inRange(self.LOWER_BOUNDS, self.UPPER_BOUNDS)
        thresh = cv2.bitwise_and(hsv, hsv, mask = mask)

        recolored = cv2.cvtColor(thresh, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

        contours, hierarchy = cv2.findContour(gray, 1, 2)
        # Find contours and sort
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

        centers = []
        tape_contours = []
        if type(contours) !=  type(None):
            M = cv2.moments(contours_sorted[i])

            # Find center x/y of the largest contours and draw circles
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            centers.append([center_x, center_y])
            # cv2.circle(actual, (center_x, center_y), 7, (255, 0, 0), 2)

            tape_contours.append(contours_sorted[i])

        x_midpoint, y_midpoint = find_midpoint(centers)

        # Returns distance of midpoint from center-x of camera feed. Since the video
        # stream is 640 x 480, the center-x is 320.
        distance_from_center = 320 - x_midpoint
        distance_to_tape = find_distance(tape_contours)

        return ['distance', 'offset'], [distance_to_tape, distance_from_center]
