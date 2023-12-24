import numpy as np
import cv2
import random
from ultralytics import YOLO
from sort.sort import Sort

import argparse

def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)
# plot_one_box(bboxes, overlayImage, label=label, color=color, line_thickness=line_thickness, bottom_label=bottom_label)
def plot_one_box(x, img, color=None, label=None, line_thickness=None, bottom_label=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1# line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    
    tf = max(tl - 1, 1)# font thickness
    if label:
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c4 = c1[0] + t_size[0], c1[1] - t_size[1] - 3# filled
        cv2.rectangle(img, c1, c4, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    if bottom_label:
        a_size = cv2.getTextSize(bottom_label, 0, fontScale=tl / 4, thickness=tf)[0]
        c3 = c1[0] + a_size[0], c2[1] + a_size[1] + 3
        cv2.rectangle(img, (c1[0], c2[1]), c3, color, -1, cv2.LINE_AA)
        cv2.putText(img, bottom_label, (c1[0], c2[1] + 12), 0, tl / 4, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def resize_img(im, target_width = 640):
    h,w,_  = im.shape
    target_height = int(h / w * target_width)
    im = cv2.resize(im , (target_width , target_height), interpolation = cv2.INTER_AREA)  
    return im,target_height,target_width

def xyxy_xywh(x1,y1,x2,y2):
    w = abs(x2-x1)
    h = abs(y2-y1)
    x,y = x1+w/2,y1+h/2
    return x,y,w,h
def xywh_xyxy(x,y,w,h):
    x1 = x-w/2
    y1 = y-h/2
    x2 = x+w/2
    y2 = y+h/2
    return int(x1),int(y1),int(x2),int(y2)

def scale_con(x,y,w,h,old,new):
    # old : tuple(height,width)
    x = int(x* old[0]/new[0])
    y = int(y*old[1]/new[1])
    h = int(h*old[0]/new[0])
    w = int(w*old[1]/new[1])
    return x,y,w,h 


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--video_path", help="input video path", type=str, default='./road_vehicle.mp4')
    parser.add_argument("--output", help="Subdirectory in seq_path.", type=str, default='./output.mp4')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=20)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=5)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    parser.add_argument("--model_path", help= "Detection model path", type=str, default="./runs/detect/train/weights/best.pt")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    display = args.display
    # Yolo model loading
    model = YOLO(args.model_path)
    # Tracker Initialization
    sort_tracker = Sort(max_age=args.max_age,
                    min_hits=args.min_hits,
                    iou_threshold=args.iou_threshold)

    # video reading

    cap = cv2.VideoCapture(args.video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
    print(frame_width,frame_height)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            img,res_height,res_width = resize_img(frame)
            result = model.predict(img)[0]
            bboxes = np.array(result.boxes.xywh)
            mapped_bboxes = [[0]]*len(bboxes)
            for ind,bbox in enumerate(bboxes):
                x,y,w,h = scale_con(bbox[0],bbox[1],bbox[2],bbox[3],(frame_height,frame_width),(res_height,res_width))
                x1,y1,x2,y2 = xywh_xyxy(x,y,w,h)
                
                mapped_bboxes[ind] = [x1,y1,x2,y2]
            # tracker updations
            outputs = sort_tracker.update(np.array(mapped_bboxes))
            for box in outputs:
                x1,y1,x2,y2,tid = box
                plot_one_box(x=(x1,y1,x2,y2),img=frame,color=compute_color_for_id(int(tid))
                            ,label=str(int(tid)),line_thickness=1)
                x,y,w,h = xyxy_xywh(x1,y1,x2,y2)
            output.write(frame)
            if args.display:
                cv2.imshow('Frame',frame)
                # press esc to quit video
                if cv2.waitKey(25) == 27:
                    break
        else:
            break
    cap.release()
    if args.display:
        cv2.destroyAllWindows()
    output.release()
            
            
            