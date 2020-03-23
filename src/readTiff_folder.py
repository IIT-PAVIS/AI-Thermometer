import cv2
import numpy
import sys
import os


if len(sys.argv) == 2:
	folder_path = str(sys.argv[1])
else:
	print('## USAGE ## \n python readTiff_folder.py path_to_folder. \n Space Bar for next image. Any other key to exit. \n##')
	exit

dirs = os.listdir(folder_path)
cv2.namedWindow(folder_path)

for imagePath in dirs:
	
	image = cv2.imread(os.path.join(folder_path,imagePath), -1)
	img_scaled = cv2.normalize(image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)

	smallest = numpy.amin(image)
	biggest = numpy.amax(image)

	print('Min: {} - Max: {}'.format(smallest, biggest))
	print(image)
	print(imagePath)

	cv2.imshow(folder_path, img_scaled)
	
	if cv2.waitKey() == 32:
		continue
	else:
		break

cv2.destroyAllWindows()
