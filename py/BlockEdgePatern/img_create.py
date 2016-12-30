import cv2
import matplotlib.pyplot
import numpy as np

step = 64
m1 = np.random.randint(0 + step*0, 63 + step*0, size=(256, 256))
m2 = np.random.randint(0 + step*1, 63 + step*1, size=(256, 256))
m3 = np.random.randint(0 + step*2, 63 + step*2, size=(256, 256))
m4 = np.random.randint(0 + step*3, 63 + step*3, size=(256, 256))

m5 = np.random.randint(127, 255, size=(256, 256))


for i in range(0, 256):
    for j in range(0, i+1):
        m3[i, j] = 0;
        m4[i, j] = 0;
        
for i in range(0, 256):
    for j in range(0, i):
        m1[i, j] = 0;
        m2[i, j] = 0;
        m5[i, j] = 0;
		
		
img=np.zeros([512, 512], dtype=np.uint8)
img2=np.zeros([512, 512], dtype=np.uint8)
img3=np.zeros([512, 512], dtype=np.uint8)

img[0:256, 0:256] = m1+m3.transpose()
img[256:512, 256:512] = m1.transpose()+m3
img[0:256, 256:512] = m2+m4.transpose()
img[256:512, 0:256] = m2.transpose()+m4

img2[0:256, 0:256] = m1+m4.transpose()
img2[256:512, 256:512] = m1.transpose()+m4
img2[0:256, 256:512] = m2+m3.transpose()
img2[256:512, 0:256] = m2.transpose()+m3


img2[0:256, 0:256] = m1+m5.transpose()
img2[256:512, 256:512] = m1.transpose()+m5
img2[0:256, 256:512] = m2+m5.transpose()
img2[256:512, 0:256] = m2.transpose()+m5

# cv2.imshow("1", cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
new_img = cv2.merge([img, img2, img3])
cv2.imshow("1", new_img)
cv2.imshow("2", cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY))
cv2.waitKey(0)
cv2.imwrite("test11.jpg", new_img)
cv2.destroyWindow("1")


