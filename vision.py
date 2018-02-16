'''
    Brandon Wood
    FRC 334
    2/15/2018
    R18 detection methods.

    Return coordinates, areas, and perimeters of found contours. 
    Comms with the robot is handled by the network class completely.

    TOOD:
        Decide which cube to grab ht
'''
class Vis:
    def __init__(self, LOWER_BOUNDS, UPPER_BOUNDS, verbose = False):
        # Storage
        self.LOWER_BOUNDS = LOWER_BOUNDS
        self.UPPER_BOUNDS = UPPER_BOUNDS

        #Settings
        self.verbose = verbose

    # Robot Methods
    def ht(self, image):
        # Change colorspace and threshold
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LOWER_BOUNDS, self.UPPER_BOUNDS)
        thresh = cv2.bitwise_and(hsv, hsv, mask = mask)

        # Run hough transform detection
        coords = self.find_houghs(thresh)
        if coords != []:
            return coords[0] # Just returing the first until we decide how to pick a centroid
        else:
            return [0, 0]

        return coords[0]
    
    def contours(self, img):
        # Change colorspace and threshold
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LOWER_BOUNDS, self.UPPER_BOUNDS)
        thresh = cv2.bitwise_and(hsv, hsv, mask = mask)

    # Implementations
    def find_cubes_with_contours(self, hsv):
        # Expects image in HSV color space, with threshold applied
        # Find contours
        color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        img, cnts, wow = cv2.findContours(
            gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find COM for each contour
        coords = (0, 0)
        
        # Find contour with greatest area
        c = self.max_area(cnts)
        
        # Find moment of contour
        M = cv2.moments(c)
        if M["m00"] > 0 and self.detect(c):
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            coords = (cX, cY)
        return coords

    def detect(self, c):
        peri = cv2.arcLength(c, False)
        #self.MIN_PERIM * 10	self.SQUARENESS/1000
        if peri < self.MIN_PERIM * 10:
            return False
        else:
            # Approximate number of sides
            approx = cv2.approxPolyDP(c, (self.SQUARENESS*.001) * peri, True)
            return len(approx) == 4

	def find_houghs(self, img):
		# Cast to 1-Channel ( Not using canny, finds text too easily)
		color = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
		gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

		# Compute and Draw Hough Lines
		coords = []
		lines = cv2.HoughLines(gray, self.RHO_ACC * .1, np.pi /
		                       self.THETA_ACC, self.MIN_LENGTH)
		if type(lines) != type(None):
            hits = self.find_hits(lines)
            coords = self.collect_points(hits)
		return coords

	def find_hits(self, lines):
		horz = []
		vert = []
		for rho, theta in lines[0]:
			if theta < np.pi/180*self.THETA_THRESH or theta > np.pi/180*(180-self.THETA_THRESH):
				horz.append(rho)
			else:
				vert.append(rho)
		hits = []

		for x in horz:
			for y in vert:
				hits.append([x, y])

		return np.array(hits, dtype=np.float32)

	def collect_points(self, hits):
		centers = [[0, 0]]
		if len(hits) > 0:
			compactness, labels, centers = cv2.kmeans(hits, int(math.ceil(len(hits)/self.NUM_CUBES)) + 1, (
			    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.NUM_ITERS, self.EPSILON*.001), 2, cv2.KMEANS_RANDOM_CENTERS)
        return centers
    
    def max_area(self, contours):
        areas = [cv2.contourArea(contour) for contour in contours]
        return max(area)
