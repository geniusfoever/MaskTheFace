# Author: aqeelanwar
# Created: 27 April,2020, 10:22 PM
# Email: aqeel.anwar@gatech.edu

import argparse
import copy
import random

import dlib
from utils.aux_functions import *



# Command-line input setup
parser = argparse.ArgumentParser(
    description="MaskTheFace - Python code to mask faces dataset"
)
parser.add_argument(
    "--path",
    type=str,
    default="",
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
    default=8,
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
args.write_path = args.path + "_masked"

# Set up dlib face detector and predictor
args.detector = dlib.get_frontal_face_detector()
path_to_dlib_model = "dlib_models/shape_predictor_68_face_landmarks.dat"
if not os.path.exists(path_to_dlib_model):
    download_dlib_model()

args.predictor = dlib.shape_predictor(path_to_dlib_model)

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

# Check if path is file or directory or none
is_directory, is_file, is_other = check_path(args.path)
display_MaskTheFace()

types=["surgical", "N95", "KN95", "cloth", "gas", "inpaint", "random", "all"]
types_probability=[0.5,0.6,0.8,1]
patterns=[]
for path,_,files in os.walk("./masks/textures/"):
    for file in files:
        patterns.append(os.path.join(path,file))

r = lambda: random.randint(0,255)
def get_random_args(my_arg):
    my_arg.color_weight=random.random()
    my_arg.color='#%02X%02X%02X' % (r(),r(),r())
    for i in range(len(types_probability)):
        if(random.random()<=types_probability[i]): my_arg.mask_type=types[i]
    my_arg.pattern=patterns[int(random.random()*len(patterns))]
    my_arg.pattern_weight=random.random()
    return my_arg
def add_mask(walk_args):
    path, dirs, files=walk_args[0]
    my_args=copy.deepcopy(walk_args[1])
    # file_count = len(files)
    # dirs_count = len(dirs)
    # if len(files) > 0:
    #     print_orderly("Masking image files", 60)

    img_index_list=[]
    for f in files:
        split_path = f.rsplit(".")
        img_index_list.append(int(split_path[0]))
    # Process files in the directory if any
    for f in files:

        split_path = f.rsplit(".")
        img_index=int(split_path[0])
        if img_index>100000000:
            continue
        elif img_index+100000000 in img_index_list:
            continue
        image_path = path + "/" + f

        #write_path = os.path.join(my_args.outpath, os.path.relpath(path, my_args.path))

        if is_image(image_path):
            # Proceed if file is image
            if my_args.verbose:
                str_p = "Processing: " + image_path
                tqdm.write(str_p)
            masked_image, mask, mask_binary_array, original_image = mask_image(
                image_path, get_random_args(my_args)
            )
            for i in range(len(mask)):
                w_path = (
                        path
                        + "/"
                        + str(int(split_path[0])+100000000)

                        + "."
                        + split_path[1]
                )
                img = masked_image[i]
                cv2.imwrite(w_path, img)

   # print_orderly("Masking image directories", 60)
from multiprocessing import Pool

from itertools import repeat
from functools import partial
if __name__ == "__main__":
    # for walk in os.walk(args.path,followlinks=True):
    #     add_mask(walk,args)
    #print  (list(zip(os.walk(args.path, followlinks=True), repeat(args))))
    if is_directory:
        func=partial(add_mask)
        with Pool(processes=8) as pool:
            results = list(tqdm(pool.map(func, zip(os.walk(args.path), repeat((args))))))



        # # Process directories withing the path provided
        # for d in tqdm(dirs):
        #     dir_path = args.path + "/" + d
        #     dir_write_path = args.write_path + "/" + d
        #     if not os.path.isdir(dir_write_path):
        #         os.makedirs(dir_write_path)
        #     _, _, files = os.walk(dir_path).__next__()
        #
        #     # Process each files within subdirectory
        #     for f in files:
        #         image_path = dir_path + "/" + f
        #         if args.verbose:
        #             str_p = "Processing: " + image_path
        #             tqdm.write(str_p)
        #         write_path = dir_write_path
        #         if is_image(image_path):
        #             # Proceed if file is image
        #             split_path = f.rsplit(".")
        #             masked_image, mask, mask_binary, original_image = mask_image(
        #                 image_path, args
        #             )
        #             for i in range(len(mask)):
        #                 w_path = (
        #                     write_path
        #                     + "/"
        #                     + split_path[0]
        #                     + "_"
        #                     + mask[i]
        #                     + "."
        #                     + split_path[1]
        #                 )
        #                 w_path_original = write_path + "/" + f
        #                 img = masked_image[i]
        #                 # Write the masked image
        #                 cv2.imwrite(w_path, img)
        #                 if args.write_original_image:
        #                     # Write the original image
        #                     cv2.imwrite(w_path_original, original_image)

                # if args.verbose:
                #     print(args.code_count)


