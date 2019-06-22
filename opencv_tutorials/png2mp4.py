#引数で取ったファイル内にある画像をmp4に変換するソース

import os
import sys
import glob
import cv2
import numpy as np

video_name = "output.mp4" 
video =  cv2.VideoWriter(str(video_name),cv2.VideoWriter_fourcc('M','P','4','v'),2.0,(1000,1000))

def main(args):
    image_dir_path= args
    files = glob.glob(str(image_dir_path) +"/*") #順番関係なく詰め込む→ リネームとかに使える
    data_num = len(files)
    #print(files)
    # for fname in files:
    for i in range(data_num):
        print(str(image_dir_path)+'epoch-{0:03d}.png'.format(i))
        bgr = cv2.imread(str(image_dir_path)+'epoch-{0:03d}.png'.format(i), cv2.IMREAD_COLOR)
        bgr = cv2,resize(bgr, (1000,1000))
        # bgr = cv2.imread(fname, cv2.IMREAD_COLOR)
        # bgr = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        # cv2.imwrite(str(i) + ".png",bgr)
        
        video.write(bgr)


if __name__ == "__main__":
    args = sys.argv
    print(args[1])
    main(args[1])

