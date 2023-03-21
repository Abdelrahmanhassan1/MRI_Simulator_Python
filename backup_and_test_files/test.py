import cv2

img1 = cv2.imread('../images/shepp_logan_phantom/480px-Shepp_logan.png', 0)

img = cv2.resize(img1, (300, 300))

cv2.imwrite('../images/shepp_logan_phantom/300px-Shepp_logan.png', img)
