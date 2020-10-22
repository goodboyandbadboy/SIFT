import cv2

img = cv2.imread('./index_1.bmp')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptor = sift.detectAndCompute(gray,None)
print(keypoints)
# print(descriptor)
img = cv2.drawKeypoints(image=img,outImage=img,keypoints=keypoints,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,color=(51,163,236))
#在关键点的部位绘制一个小圆圈
#image 原始图像，可以是三通道或者单通道图像
#keypoints:特征点向量，向量内每一个元素是一个KeyPoint对象，包含了特征点的各种属性信息
#outimage :特征点绘制的画布图像，可以是原图像
#color:绘制的特征点颜色信息
#flags:特征点的绘制模式
#DRAW_RICH_KEYPOINTS 绘制特征点的时候绘制的是一个带有方向的圆，这种方法同时显示图像的坐标和大小和方向
cv2.imshow('sift_keypoints',img)

cv2.waitKey(0)
cv2.destroyAllWindows()