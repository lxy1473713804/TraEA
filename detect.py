from torch.utils import data

from utils.utils import *
from utils.datasets import *


from PIL import Image

def detector(frame, model, device, half):
    img_size = 640
    img0 = frame

    img = letterbox(img0, new_shape=img_size)[0]
    imgcrop = img
    img = img[:, :, :].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # print("-----------------------")
    # print(img)

    with torch.no_grad():
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres=0.50, iou_thres=0.45,classes=[0,1,2,3])
        # print("-----------------------")
        names = model.module.names if hasattr(model, 'module') else model.names
        for i, det in enumerate(pred):

            #print("-----------------------")
            #print(pred)
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                bbox_xywh1 = []
                bbox_xyxy1 = []

                cls_conf1 = []
                cls_id1 = []
                cls_name1=[]
                s=''
                # --------------我后面加的一部分，获取目标类别的-------------
                s += '%gx%g ' % img.shape[2:]  # print string
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in det:
                    cls_confirm = int(cls.cpu().numpy())
                    #print(cls_confirm)
                    xyxy = [x.cpu() for x in xyxy]


                    #label = f'{names[int(cls)]} {conf:.2f}'
                    label = f'{names[int(cls)]}'
                    cls_name1.append(label)

                    #if cls_confirm == 0:
                    bbox_xyxy1.append(xyxy)


                    bbox_xywh1.append((xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist())
                    cls_conf1.append(conf.item())
                    cls_id1.append(int(cls))

               #print(det)
                bbox_xyxy1 = np.array(bbox_xyxy1)

                bbox_xywh1 = np.array(bbox_xywh1)
                cls_conf1 = np.array(cls_conf1)
                cls_id1 = np.array(cls_id1)
                cls_name1=np.array(cls_name1)
                #获取视频中每个目标的前景图像


                return bbox_xywh1, cls_conf1, cls_id1, bbox_xyxy1,cls_name1
            else:
                return None, 0, 0, None
