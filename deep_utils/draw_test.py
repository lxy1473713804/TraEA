import datetime
import numpy as np
import cv2
import time
import pymysql
import threading
#打开数据库连接

conn = pymysql.connect(host='localhost',user = "root",passwd = "gis6391700",db = "geotradb",charset='utf8')
cursor = conn.cursor()
sql = "DELETE FROM geopiccoor"
cursor.execute(sql)
conn.commit()
print('成功删除上次保存的轨迹', cursor.rowcount, '条数据')

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
pp_name = {}


def draw_dis(img, point_list):
    for i in range(len(point_list)):
        origin = point_list.copy()
        left_point = origin.pop(i)
        dist = {}
        for j in range(len(origin)):
            dist0 = np.sqrt(np.square(left_point[0]-origin[j][0])+np.square(left_point[1]-origin[j][1]))
            dist[j] = dist0
        if dist:
            m = min(dist, key=dist.get)
            cv2.putText(img, str(round(dist[m], 2)), left_point, 0, 0.5, (0, 0, 255), 2)
            cv2.line(img, left_point, origin[m], (255, 0, 0), 2)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


class Person():
    def __init__(self, id):
        self.id = id
        self.tracklets = []

    def append_center(self, center):
        self.tracklets.append(center)
        if len(self.tracklets) > 50:
            self.tracklets = self.tracklets[-50:-1]


def draw_boxes(img, im0, imcrop, bbox, identities=None ,cls_name=None, Savefile=None,idx_frame=None):

    v = int(0)

    im2=imcrop.copy()

    if len(bbox)>len(cls_name):
        length=len(cls_name)
    else:
        length=len(bbox)

    ALlReslutForTxt=[]
    for i,box in enumerate(bbox):


        x1,y1,x2,y2 = [int(i) for i in box]

        GroundPoint=(int((x1+x2)/2), y2)

        center = (int((x1+x2)/2), int((y1+y2)/2))

        id = int(identities[i]) if identities is not None else 0

        ppname =str(id)
        if i<=length:
            savetxt = f'{id} {cls_name[i-1]}'
            type =f'{cls_name[i-1]}'
        else:
            savetxt = f'{id}'
            type = 'no'

        label=f'{id}'

        #print(label)
        if ppname not in pp_name.keys():
            pp_name[ppname] = Person(id)
        pp_name[ppname].append_center(GroundPoint)
        count = len(pp_name)
        v += 1
        color = compute_color_for_labels(id)
        # 轨迹
        pts = pp_name[ppname].tracklets

        # #本次我注释了这个地方  11.17
        # print(str(id)+"------------------------")
        # print(pts)


        for j in range(1, len(pts)):

            if pts[j - 1] is None or pts[j] is None:
                continue


            cv2.line(img, (pts[j - 1]), (pts[j]), (color), 3)

        #将ALlReslutForTxt中的字符串存入到



        Sstr = ""
        for j in range(1, len(pts)):

            cx = pts[j][0]
            cy = pts[j][1]
            if j != len(pts) - 1:
                strr = '(' + str(cx) + ',' + str(cy) + ')*'
            else:
                strr = '(' + str(cx) + ',' + str(cy) + ')'
            Sstr += strr

        #print(Sstr)
        c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))

        # -----------------------剪切目标图像，并保存-------------------------------
        im_crop = im2.copy()
        image= im_crop[int(y1):int(y2),int(x1):int(x2)]
        im_str = str(int(x1)) + "," + str(int(y1)) + "," + str(int(x2)) + "," + str(int(y2))
        # print(im_str)
        if (image.any()):

            time1 = time.strftime("%Y-%m-%d/%H:%M:%S", time.localtime())
            savefile = Savefile + "\\" + str(savetxt) + "_" + im_str + ".jpg";

            new_im = cv2.imwrite(Savefile + "\\" + str(savetxt) + "_" + im_str + ".jpg", image)#注释之后即没有保存图像


        # --------------------------数据库相关操作----------------------------
        if Sstr!='':
            #if idx_frame%25==0:
                # savefile.replaceAll("\\\\", "/")
                savesql = str(savefile).replace("\\", "\\\\\\")
                # print(savesql)
                #TXT 保存实时数据
                savetotxt=str(idx_frame)+"+"+str(label)+"+"+str(type)+"+"+str(Sstr)
                ALlReslutForTxt.append(savetotxt)
                # with open("f:\\test.txt", "w") as f:
                #     f.write(savetotxt)

                    #数据库 保存实时数据
                sql = "INSERT INTO geopiccoor (frame,objid, type, piccoor,time,piclocal) VALUES ( '%s', '%s', '%s', '%s', '%s', '%s' )"
                data = (idx_frame, label, type, Sstr, time1, savesql)
                cursor.execute(sql % data)
                conn.commit()
                # 保存历史数据
                sql = "INSERT INTO geopichistory (frame,objid, type, piccoor,time,piclocal) VALUES ('%s','%s', '%s', '%s', '%s', '%s' )"
                data = (idx_frame, label, type, Sstr, time1, savesql)
                cursor.execute(sql % data)
                conn.commit()
            # print('成功插入', cursor.rowcount, '条数据')




        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, 0.5, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        cv2.rectangle(img,c1,c2,color,3)
        cv2.rectangle(im0,c1,c2,color,3)
    if (i == len(bbox) - 1):
        print(Sstr)
    lock = threading.Lock()
    lock.acquire()
    if len(ALlReslutForTxt) > 0:

        with open("f:\\test.txt", "w") as f:
            for single in ALlReslutForTxt:
                # print(ALlReslutForTxt[i])
                f.write(single)
                f.write('\r\n')
    lock.release()


    # print(strtra)
    # 车之间的距离
    list_nodo = []
    for z in range(len(identities)):
        id0 = identities[z]
        pp =str(id0)
        list_nodo.append(pp_name[pp].tracklets[-1])
    draw_dis(im0, list_nodo)

    cv2.putText(img, "Total Object Counter: " + str(count), (int(10), int(110)), 0, 1, (0, 255, 0), 2)
    cv2.putText(img, "Current Object Counter: " + str(v), (int(10), int(70)), 0, 1, (0, 255, 0), 2)
    return img, im0


if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
