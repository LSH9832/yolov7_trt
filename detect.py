from model import YOLOTRT
import cv2
import os
from time import time


def main():

    # load engine model
    # engine_file = "yolov7_640x640.engine"
    # yaml_file = "yolov7_640x640.yaml"
    # model = YOLOTRT(
    #     engine_file=engine_file,
    #     yaml_file=yaml_file
    # )
    # if not os.path.isfile("trtOutput/yolov7_640x640.pt"):
    #     model.save2pth("trtOutput/yolov7_640x640.pt")

    source = "/home/lsh/Videos/test.avi"

    model = YOLOTRT(pt_file="trtOutput/yolov7_640x640.pt")

    model.conf_thres = 0.35
    model.nms_thres = 0.5

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    while cap.isOpened():
        t0 = time()
        success, frame = cap.read()
        if not success:
            break

        img = model(frame.copy(), end2end=True, count_time=False, draw=True)[0][0]

        cv2.imshow("result", img)
        if cv2.waitKey(1) == 27:
            break

        print(f"\r{(time() - t0) * 1000:.1f}ms ", end="")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()