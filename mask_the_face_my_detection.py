# Author: aqeelanwar
# Created: 27 April,2020, 10:22 PM
# Email: aqeel.anwar@gatech.edu

import argparse
import copy
import os
import random

import numpy as np
import cv2
import insightface
from utils.aux_functions import *



# Command-line input setup
parser = argparse.ArgumentParser(
    description="MaskTheFace - Python code to mask faces dataset"
)
parser.add_argument(
    "--path",
    type=str,
    default="D:\DataBase\Glint360k",
    help="Path to either the folder containing images or the image itself",
)

parser.add_argument(
    "--outpath",
    type=str,
    default="",
    help="Path to either the folder containing images or the image itself",
)
parser.add_argument(
    "--mask_type",
    type=str,
    default="surgical",
    choices=["surgical", "N95", "KN95", "cloth", "gas", "inpaint", "random", "all"],
    help="Type of the mask to be applied. Available options: all, surgical_blue, surgical_green, N95, cloth",
)

parser.add_argument(
    "--pattern",
    type=str,
    default="",
    help="Type of the pattern. Available options in masks/textures",
)

parser.add_argument(
    "--pattern_weight",
    type=float,
    default=0.5,
    help="Weight of the pattern. Must be between 0 and 1",
)

parser.add_argument(
    "--color",
    type=str,
    default="#0473e2",
    help="Hex color value that need to be overlayed to the mask",
)

parser.add_argument(
    "--color_weight",
    type=float,
    default=0.5,
    help="Weight of the color intensity. Must be between 0 and 1",
)

parser.add_argument(
    "--code",
    type=str,
    # default="cloth-masks/textures/check/check_4.jpg, cloth-#e54294, cloth-#ff0000, cloth, cloth-masks/textures/others/heart_1.png, cloth-masks/textures/fruits/pineapple.png, N95, surgical_blue, surgical_green",
    default="",
    help="Generate specific formats",
)
parser.add_argument(
    "--process",
    type=int,
    default=16,
)

parser.add_argument(
    "--verbose", dest="verbose", action="store_true", help="Turn verbosity on"
)
parser.add_argument(
    "--write_original_image",
    dest="write_original_image",
    action="store_true",
    help="If true, original image is also stored in the masked folder",
)
parser.set_defaults(feature=False)

args = parser.parse_args()

# Set up dlib face detector and predictor
# args.detector = dlib.get_frontal_face_detector()
# path_to_dlib_model = "dlib_models/shape_predictor_68_face_landmarks.dat"
# if not os.path.exists(path_to_dlib_model):
#     download_dlib_model()
#
# args.predictor = dlib.shape_predictor(path_to_dlib_model)

# Extract data from code
mask_code = "".join(args.code.split()).split(",")
args.code_count = np.zeros(len(mask_code))
args.mask_dict_of_dict = {}


for i, entry in enumerate(mask_code):
    mask_dict = {}
    mask_color = ""
    mask_texture = ""
    mask_type = entry.split("-")[0]
    if len(entry.split("-")) == 2:
        mask_variation = entry.split("-")[1]
        if "#" in mask_variation:
            mask_color = mask_variation
        else:
            mask_texture = mask_variation
    mask_dict["type"] = mask_type
    mask_dict["color"] = mask_color
    mask_dict["texture"] = mask_texture
    args.mask_dict_of_dict[i] = mask_dict



types=["surgical", "N95", "KN95", "cloth", "gas", "inpaint", "random", "all"]
types_probability=[0,0.7,0.8,0.9,1]
patterns=[]
for path,_,files in os.walk("./masks/textures/"):
    for file in files:
        patterns.append(os.path.join(path,file))

r = lambda: random.randint(0,255)
def get_random_args(arg):
    my_arg=copy.deepcopy(arg)
    type_random=random.random()

    my_arg.color_weight=random.random()*0.3
    my_arg.color='#%02X%02X%02X' % (r(),r(),r())
    for i in range(len(types_probability)):
        if(type_random > types_probability[i] and type_random<types_probability[i+1]): my_arg.mask_type=types[i]
    if random.random()<0.1:
        my_arg.pattern=patterns[int(random.random()*len(patterns))]
        my_arg.pattern_weight=random.random()
    return my_arg

random_args_list=[]
for _ in range(100):
    random_args_list.append(get_random_args(args))
def parse_ann_line(line):
    values = [float(x) for x in line.strip().split()]
    bbox = np.array(values[0:4], dtype=np.float32 )
    kps = np.array( values[4:19], dtype=np.float32 ).reshape((5,3))
    if kps[0][2]<-0.001: return None
    kps=kps[:,:2]
    return dict(bbox=bbox, kps=kps)

def read_txt(path):
    info_list=[]
    ground_truth_list = []

    for line in open(path, 'r'):
        if line.startswith('#'):

            if ground_truth_list:
                if None not in ground_truth_list:
                    info_list.append([file_rl,ground_truth_list])
            ground_truth_list = []

            file_rl = line[1:].strip()
        else:
            ground_truth_list.append(parse_ann_line(line))
    return info_list
input_dir=r"E:\Github\insightface\detection\scrfd\data\retinaface\train\images"
output_dir=r"E:\Github\insightface\detection\scrfd\data\retinaface\train_masked\images"
def add_mask_single(my_args,landmark106detector,img_info):
    landmark106detector=landmark106detector[random.randint(0,1)]
    img_path=os.path.join(input_dir,img_info[0])
    img=cv2.imread(img_path,cv2.IMREAD_COLOR)
    faces_infos = img_info[1]
    for i,face_info in enumerate(faces_infos):
        face=insightface.app.common.Face(kps=face_info['kps'],bbox=np.array([face_info['bbox'][0],face_info['bbox'][1],face_info['bbox'][0]+face_info['bbox'][2],face_info['bbox'][1]+face_info['bbox'][3]]))
        landmarks=landmark106detector.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),face)
        faces_infos[i]['landmark']=np.array(landmarks).astype(int)[[1,10,12,14,16,3,5,7,0,23,21,19,32,30,28,26,17,    # 脸颊17点
                 43,48,49,51,50,      # 左眉毛5点
                 102,103,104,105,101, # 右眉毛5点
                 72,73,74,86,78,79,80,85,84, # 鼻子9点
                 35,41,42,39,37,36,   # 左眼睛6点
                 89,95,96,93,91,90,   # 右眼睛6点
                 52,64,63,71,67,68,61,58,59,53,56,55,65,66,62,70,69,57,60,54 # 嘴巴20点
                 ]]
    img_result=mask_image_my(img_path,random_args_list,faces_infos)
    write_path=os.path.join(output_dir,img_info[0])
    if not os.path.isdir(os.path.dirname(write_path)):
        os.mkdir(os.path.dirname(write_path))
    cv2.imwrite(os.path.join(output_dir,img_info[0]),img_result)
    return

   # print_orderly("Masking image directories", 60)
from multiprocessing import Pool

from functools import partial
if __name__ == "__main__":
    handler1 = insightface.model_zoo.get_model(r"C:\Users\beich\.insightface\models\buffalo_l\2d106det.onnx",
                                              providers=['CUDAExecutionProvider'])
    handler1.prepare(ctx_id=0)

    handler2 = insightface.model_zoo.get_model(r"C:\Users\beich\.insightface\models\buffalo_l\2d106det.onnx",
                                              providers=['CUDAExecutionProvider'])
    handler2.prepare(ctx_id=1)
    # for walk in os.walk(args.path,followlinks=True):
    #     add_mask(walk,args)
    # #print  (list(zip(os.walk(args.path, followlinks=True), repeat(args))))
    # if is_directory:
    info_list=read_txt(r"C:\Users\beich\Downloads\retinaface_gt_v1.1\train\label.txt")


    func=partial(add_mask_single, args,[handler1,handler2])
    # with Pool(processes=args.process) as pool:
    #     list(tqdm(pool.map(func, os.walk(args.path)),total=360232))
    pool = Pool(processes=6)
    for _ in tqdm(pool.imap_unordered(func, info_list)):
        pass
    # for walk in os.walk(args.path):
    #     add_mask_single(walk)
