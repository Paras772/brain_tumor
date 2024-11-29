import numpy as np
import cv2 as cv

class DisplayTumor:
    def __init__(self):
        self.curImg = 0
        self.Img = 0
        self.kernel = np.ones((3, 3), np.uint8)  # Kernel for morphological operations

    def readImage(self, img):
        # Convert PIL image to numpy array and store it
        self.Img = np.array(img)
        self.curImg = np.array(img)
        gray = cv.cvtColor(self.Img, cv.COLOR_RGB2GRAY)  # Convert to grayscale
        self.ret, self.thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    def getImage(self):
        return self.curImg

    # Remove noise using morphological operations
    def removeNoise(self):
        opening = cv.morphologyEx(self.thresh, cv.MORPH_OPEN, self.kernel, iterations=2)
        self.curImg = opening

    # Display tumor region by marking the tumor area
    def displayTumor(self):
        # Sure background area
        sure_bg = cv.dilate(self.curImg, self.kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv.distanceTransform(self.curImg, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Find unknown region (subtract background from foreground)
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)

        # Marker labeling for region identification
        ret, markers = cv.connectedComponents(sure_fg)

        # Add 1 to labels to ensure sure background is not labeled 0 (label as 1)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv.watershed(self.Img, markers)

        # Mark the edges of the regions with red color (255, 0, 0)
        self.Img[markers == -1] = [255, 0, 0]

        # Convert the HSV color image to BGR for tumor highlighting
        tumorImage = cv.cvtColor(self.Img, cv.COLOR_HSV2BGR)
        self.curImg = tumorImage

        return tumorImage  # Return the highlighted image
