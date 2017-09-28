import os
import cv2

def cut_video():
    vidcap = cv2.VideoCapture('/home/lizhuyun/video/WeChat.mp4')
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        cv2.imwrite("/home/lizhuyun/video/result/frame%d.jpg" % count, image)  # save frame as JPEG file
        count += 1


def combine_pic():
  img_root = '/home/lizhuyun/video/result/result'
  fps = 30

  fourcc = cv2.VideoWriter_fourcc(*'MJPG')
  videoWriter = cv2.VideoWriter('/home/lizhuyun/video/saveVideo.mp4',fourcc,fps,(800,600))
  for i in range(2323):
     frame = cv2.imread(img_root+str(i)+'.png')
     videoWriter.write(frame)
     videoWriter.release()



if __name__ == '__main__':
    cut_video()
    combine_pic()