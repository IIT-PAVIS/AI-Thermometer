import cv2
import numpy
import sys

imagePath = '../data/thermal_16_bit/FLIR_00327.tiff'
if len(sys.argv) == 2:
	imagePath = str(sys.argv[1])
image = cv2.imread(imagePath, -1)
img_scaled = cv2.normalize(image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)

smallest = numpy.amin(image)
biggest = numpy.amax(image)

print('Min: {} - Max: {}'.format(smallest, biggest))

cv2.imshow('tiff', img_scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(image)
