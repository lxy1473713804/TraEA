import argparse
import warnings
import os
from deep_sort import build_tracker
from deep_utils.draw_test import draw_boxes
from deep_utils.parser import get_config
from detect import *
from utils import torch_utils


class VideoTracker(object):
    #------------------初始化---------------------
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args  #各类参数的传递
        #---------------------------启动GPU-------------------------------
        use_cuda = args.use_cuda and torch.cuda.is_available()  #torch.cuda.is_available() 判断GPU是否可用
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        # if args.display:
        #     cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        #     #cv2.resizeWindow("test", args.display_width, args.display_height*2)
        #------------------------摄像机，视频----------------------------------
        if args.cam != -1:# 等于-1的时候，是连接电脑的摄像头
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)#连接摄像机
        else:
            self.vdo = cv2.VideoCapture()
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)#建立跟踪模型
        #------------------模型相关------------------------------------------
        #self.args.yolo_weights=args.yolo_weights

    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            #ret, frame = self.vdo(self.args.cam)

            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
            self.vdo.open(self.args.VIDEO_PATH)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path: #保存视频
            #fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width,self.im_height))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):#视频播放与运行
        idx_frame = 0

        # Load yolov3_tiny_se detect
        weights = args.yolo_weights  #pt文件

        #weights = 'yolov5/weights/traffic2000.pt'
        #weights = 'yolov5/weights/caochang.pt'
        device = torch_utils.select_device(device='0')
        half = device.type != 'cpu'

        model = torch.load(weights, map_location=device)['model'].float()
        names = model.module.names if hasattr(model, 'module') else model.names#标签
        model.to(device).eval()
        if half:
            model.half()

        fps_list = []

        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            _, ori_im = self.vdo.retrieve()

            img_copy = ori_im
            im_crop = img_copy.copy()

            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            im_blank = ori_im.copy()
            im_blank.fill(255)
            #新建一个文件夹， 下面存储每一帧的目标

            SaveFile = "D:\\SocialDistance Picture\\"+str(idx_frame)
            mkdir(SaveFile)


            #print(im.shape)
            start = time.time()
            # do detection
            bbox_xywh, cls_conf, cls_ids, bbox_xyxy1,cls_name = detector(im, model, device, half)
            #print(cls_name)


            # print(cls_ids)
            # s=''
            # for c in cls_ids[:, -1].unique():
            #     n = (cls_ids[:, -1] == c).sum()  # detections per class
            #     s += '%g %ss, ' % (n, names[int(c)])  # add to string
           # print(cls_ids)
            #print(model)
            #print("-----------------------")
            #print(bbox_xywh)
            if bbox_xywh is not None:
                # do tracking
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)
                #labelview = f'{id} {names[c]}'
                #print(labelview)
                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:,:4]
                   # print(bbox_xyxy)
                    identities = outputs[:,-1]
                    #print(outputs)

                    #print(identities)
                    # print(bbox_xywh)
                    im_crop=img_copy.copy()
                    ori_im, im_blank = draw_boxes(ori_im, im_blank,im_crop, bbox_xyxy, identities,cls_name,SaveFile )


                    # #保存目标图像至文件夹
                    # for i in range(len(bbox_xyxy)):
                    #     im_crop = img_copy.copy()
                    #     # print(i)
                    #     im_crop = im_crop[int(bbox_xyxy[i][1].item()):int(bbox_xyxy[i][3].item()),int(bbox_xyxy[i][0].item()):int(bbox_xyxy[i][2].item())]
                    #     #输出坐标串
                    #     im_str=str(int(bbox_xyxy[i][0].item()))+","+str(int(bbox_xyxy[i][1].item()))+","+str(int(bbox_xyxy[i][2].item()))+","+str(int(bbox_xyxy[i][3].item()))
                    #     # print(im_str)
                    #
                    #     if(im_crop.any()):
                    #         #print(SaveFile+"\\"+ str(label) + "_" + str(cls_ids[i]) +"_"+im_str+"_"+ str(time.time())+".jpg")
                    #         # print(SaveFile+"\\"+ str(i) + "_" + str(time.time())+".jpg")
                    #         new_im = cv2.imwrite(SaveFile+"\\"+ str(i) + "_" + str(time.time())+".jpg", im_crop)



            end = time.time()
            fps = 1/(end-start+0.001)
            fps_list.append(fps)
            #print("total-time: {:.03f}s, fps: {:.03f}".format(end-start, fps))

            if self.args.display:
                cv2.imshow("t1", ori_im)#显示每一帧视频

                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)
        avg_fps = np.mean(fps_list)
        #print("avg_fps: {:.03f}".format(avg_fps))


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--VIDEO_PATH",default='caochang09.mp4', type=str)
    parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/traffic.pt', help='model.pt path')
    parser.add_argument("--VIDEO_PATH", default='sanxiao.mp4', type=str)
    parser.add_argument("--config_deepsort", type=str, default="deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument("--save_path", type=str,default='output/video.avi', help='display tracking video results')
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    #parser.add_argument("--camera", action="store", dest="cam", type=str, default="rtmp://47.93.227.44:1935/live/home3")
    # parser.add_argument("--camera", action="store", dest="cam", type=str, default="rtmp://47.93.227.44:1935/live/home1")
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)
    with VideoTracker(cfg, args) as vdo_trk:
        vdo_trk.run()
