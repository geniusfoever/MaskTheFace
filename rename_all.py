import argparse
from multiprocessing import Process
from tqdm import tqdm
import os
import shutil
os.environ["PYTHONHASHSEED"]='0'
parser = argparse.ArgumentParser(description='do dataset merge')
# general
parser.add_argument('--include', default=r"C:\Dataset\glint_umd", type=str, help='')
parser.add_argument('--output', default=r"E:\dataset\glint\imgs", type=str, help='')
parser.add_argument('--process', default=1, type=int, help='')

args = parser.parse_args()

root="E:/dataset/glint"
alternative_folders_number=10

alternative_folders=[]
for i in range(alternative_folders_number):
    d=os.path.join(root,"imgs"+str(i))
    alternative_folders.append(d)
    if not os.path.exists(d): os.mkdir(d)
def rename(start,end,p_id,original_folder,target_folders):
    a = tqdm(total=end-start,postfix=p_id+1,position=p_id)
    folder_number=len(target_folders)
    for i in range(start,end):
        for target_folder in target_folders:
            if not os.path.isdir(os.path.join(target_folder,str(i))): os.mkdir(os.path.join(target_folder,str(i)))

        current_folder=os.path.join(original_folder,str(i)+".0")
        for img_name in os.listdir(current_folder):
            target_index=hash(img_name)%folder_number
            shutil.move(os.path.join(current_folder, img_name), os.path.join(alternative_folders[target_index], str(i),img_name))
        if i % 100 == 0:
            a.set_description(str(i))
            a.update(100)
if __name__ == "__main__":
    # for walk in os.walk(args.path,followlinks=True):
    #     add_mask(walk,args)
    # #print  (list(zip(os.walk(args.path, followlinks=True), repeat(args))))
    # if is_directory:
    process=args.process
    pandding=0
    total_number=73382
    # total_number=17091657
    start_end_list=[pandding+total_number//process*i for i in range(process)]
    start_end_list.append(total_number)
    start_end_list[0]=0
    print(start_end_list)
    process_list=[]
    for i in range(process):
        process_list.append(Process(target=rename,args=(start_end_list[i],start_end_list[i+1],i,os.path.join(root,"imgs"),
                                                        alternative_folders)))

    for p in process_list:
        p.start()

    for p in process_list:
        p.join()