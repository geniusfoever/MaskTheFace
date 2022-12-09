import copy

import mxnet as mx
from mxnet import recordio
import argparse
parser = argparse.ArgumentParser(description='do dataset merge')
# general
parser.add_argument('--include', default=r"C:\Dataset\glint_umd", type=str, help='')
parser.add_argument('--output', default=r"E:\dataset\glint\imgs", type=str, help='')
# Author: aqeelanwar
# Created: 27 April,2020, 10:22 PM
# Email: aqeel.anwar@gatech.edu

import dlib
from utils.aux_functions import *




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
dlib.DLIB_USE_CUDA=True
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

random_args_list=[get_random_args(copy.copy(args)) for _ in range(100)]
output=args.output
path_imgidx = os.path.join(args.include,"train.idx") # path to train.rec
path_imgrec = os.path.join(args.include,"train.rec") # path to train.idx


def extract(start,end,p_id,args_list):
    imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
    header, s = recordio.unpack(imgrec.read_idx(start))

    def blur(img_input):
        v_kernel_size = random.randrange(1, 3)
        h_kernel_size = random.randrange(3, 15)
        # Create the vertical kernel.
        kernel_v = np.zeros((v_kernel_size, v_kernel_size))

        # Create a copy of the same for creating the horizontal kernel.
        kernel_h = np.zeros((h_kernel_size, h_kernel_size))

        # Fill the middle row with ones.
        kernel_v[:, int((v_kernel_size - 1) / 2)] = np.ones(v_kernel_size)
        kernel_h[int((h_kernel_size - 1) / 2), :] = np.ones(h_kernel_size)

        # Normalize.
        kernel_v /= v_kernel_size
        kernel_h /= h_kernel_size

        # Apply the vertical kernel.
        vertical_mb = cv2.filter2D(img_input, -1, kernel_v)

        # Apply the horizontal kernel.
        return cv2.filter2D(vertical_mb, -1, kernel_h)
    def bright_contrast(img_input):
        brightness = int(random.randrange(175, 340) + (-255))
        contrast = int(random.randrange(75, 190) + (-127))

        shadow = max(brightness, 0)
        max_num = max(255, 255 + brightness)

        al_pha = (max_num - shadow) / 255
        ga_mma = shadow

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(img_input, al_pha,
                              img_input, 0, ga_mma)

        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)

        # The function addWeighted calculates
        # the weighted sum of two arrays
        return cv2.addWeighted(cal, Alpha,
                               cal, 0, Gamma)


    a = tqdm(total=end-start,postfix=p_id+1)
    for i in range(start,end):

        #print(str(header.label))
        #img = np.array(mx.image.imdecode(s))
        img = mx.image.imdecode(s).asnumpy()
        #print(type(img))
        path = os.path.join(output,str(round(header.label[0])))
        if not os.path.isdir(path):os.mkdir(path)
        path_0 = os.path.join(path,str(header.id)+'.jpg')
        #--------------------------------------------
        if os.path.isfile(path_0):continue
        #--------------------------------------------

        path_1 = os.path.join(path,str(header.id+100000000)+'.jpg')
        path_2 = os.path.join(path,str(header.id+200000000)+'.jpg')
        path_3 = os.path.join(path,str(header.id+300000000)+'.jpg')
        path_4 = os.path.join(path,str(header.id+400000000)+'.jpg')
        path_5 = os.path.join(path,str(header.id+500000000)+'.jpg')
        #---------------------------------------------------------------------------------------------------------------

        # if os.path.isfile(path+'.jpg'):
        #     header, s = recordio.unpack(imgrec.read())
        #     continue


        #fig = plt.figure(frameon=False)
        #fig.set_size_inches(124,124)
        #ax = plt.Axes(fig, [0., 0., 1., 1.])
        #ax.set_axis_off()
        #fig.add_axes(ax)
        #ax.imshow(img, aspect='auto')
        #dpi=1
        #fname= str(i)+'jpg'
        #fig.savefig(fname, dpi)
        #plt.savefig(path+'.jpg',bbox_inches='tight',pad_inches=0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path_0, img)

        #---------------------------------------------------------------------------------------------------------------

        img_1 = mask_image_from_array(
            img,random.choice(args_list)
        )

        if img_1 is not None:
            cv2.imwrite(path_1,img_1)
            img_3=blur(img_1)
            cv2.imwrite(path_3, img_3)
            img_5=bright_contrast(img_1)
            cv2.imwrite(path_5, img_5)

        #---------------------------------------------------------------------------------------------------------------


        img_2=blur(img)
        cv2.imwrite(path_2, img_2)

        #---------------------------------------------------------------------------------------------------------------



        img_4=bright_contrast(img)
        cv2.imwrite(path_4, img_4)



        header, s = recordio.unpack(imgrec.read())
        if i%100==0:
            a.set_description(str(header.label[0]))
            a.update(100)

from multiprocessing import Process
from tqdm import tqdm
if __name__ == "__main__":
    # for walk in os.walk(args.path,followlinks=True):
    #     add_mask(walk,args)
    # #print  (list(zip(os.walk(args.path, followlinks=True), repeat(args))))
    # if is_directory:
    process=args.process
    pandding=5000000
    total_number=1000000
    # total_number=17091657
    start_end_list=[pandding+total_number//process*i for i in range(process)]
    start_end_list.append(total_number)
    start_end_list[0]=1
    process_list=[]
    for i in range(process):
        process_list.append(Process(target=extract,args=(start_end_list[i],start_end_list[i+1],i,random_args_list)))

    for p in process_list:
        p.start()

    for p in process_list:
        p.join()