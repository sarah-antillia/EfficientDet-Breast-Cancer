# 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# create_yolo_annotation.py
# 2023/04/10 Antillia.com Toshiyuki Arai
#

import sys
import os
import glob

import cv2
import shutil
import traceback


def create_yolo_annotation(input_dir, output_dir, debug=False):
  classes_dirs = [[0, "benign"], [1, "malignant"]]
  for class_id, sub_dir  in classes_dirs:
    org_images_dir = os.path.join(input_dir, sub_dir)
    pattern = org_images_dir + "/*mask.png"
    print("--- pattern {}".format(pattern))
    mask_files     = glob.glob(pattern)
    SP  = " "
    NL  = "\n"
    for mask_file in mask_files:
       mask_img = cv2.imread(mask_file, cv2.COLOR_BGR2GRAY)
       H, W = mask_img.shape[:2]

       contours, hierarchy = cv2.findContours(mask_img, 
           cv2.RETR_EXTERNAL, 
           cv2.CHAIN_APPROX_SIMPLE)
       contours = max(contours, key=lambda x: cv2.contourArea(x))
       x, y, w, h = cv2.boundingRect(contours)
       print("---x {} y {} w {} h {}".format(x, y, w, h))
       
       print("--- mask file {}".format(mask_file))
       
       cx = x + w/2
       cy = y + h/2
       #Convert to relative coordinates for YOLO annotations
       rcx = round(cx / W, 5)
       rcy = round(cy / H, 5)
       rw  = round( w / W, 5)
       rh  = round( h / H, 5)

       basename = os.path.basename(mask_file)
       nomask_name = basename.split("_")[0]
       nomask_file_name = nomask_name + ".png"
       print("---- {}".format(nomask_file_name))
       #cv2.imshow("image", img)
       #input("HIT")
       org_image = os.path.join(org_images_dir, nomask_file_name)


       #output_sub_dir = os.path.join(output_dir, sub_dir)
       #if not os.path.exists(output_sub_dir):
       #  os.makedirs(output_sub_dir)

       non_mask_img = cv2.imread(org_image)
       resized_img  = cv2.resize(non_mask_img, (512, 512), interpolation = cv2.INTER_LANCZOS4)
       basename     = os.path.basename(org_image)
       RH, RW       = resized_img.shape[:2]
       resize_ratio_w = RW/W
       resize_ratio_h = RH/H

       name = basename.split(".")[0]
       #Save as JPG file
       fname = name + ".jpg"
       if debug:
         output_dir_annotated = os.path.join(output_dir, "annotated")
         if not os.path.exists(output_dir_annotated):
           os.makedirs(output_dir_annotated)
         #non_mask_img = cv2.imread(org_image, cv2.COLOR_BGR2RGB)
         x = int(x * resize_ratio_w)
         y = int(y * resize_ratio_h)
         w = int(w * resize_ratio_w)
         h = int(h * resize_ratio_h)
         non_mask_img = cv2.rectangle(resized_img, (x, y), (x+w, y+h), (255, 255, 0), 3)

         ouput_image_file_annotated = os.path.join( output_dir_annotated, fname)
         cv2.imwrite( ouput_image_file_annotated, non_mask_img)
         print("--- create a annotated image file {}".format(ouput_image_file_annotated))
       resized_img_file_path = os.path.join(output_dir, fname)
       cv2.imwrite(resized_img_file_path, resized_img)

       #shutil.copy2(org_image, output_sub_dir)
       annotation = str(class_id ) + SP + str(rcx) + SP + str(rcy) + SP + str(rw) + SP + str(rh) 
       annotation_file = nomask_name + ".txt"
       annotation_file_path = os.path.join(output_dir, annotation_file)
       with open(annotation_file_path, "w") as f:
         f.writelines(annotation + NL)
         print("---Created annotation file {}".format(annotation_file_path))
         print("---YOLO annotation {}".format(annotation))

"""
./BUSI_master/
├─test/
│  ├─benign/
│  └─malignant/
├─train/
│  ├─benign/
│  └─malignant/
└─valid/
    ├─benign/
    └─malignant/      
      
"""

if __name__ == "__main__":
  try:
    input_dir  = "./BUSI_master"
    output_dir = "./YOLO"
    targets    = ["test", "train", "valid"]

    if not os.path.exists(input_dir):
      raise Exception("===NOT FOUND " + input_dir)
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    debug = False
    for target in targets:
      input_sub_dir     = os.path.join(input_dir, target)
      output_target_dir = os.path.join(output_dir, target)
      if not os.path.exists(output_target_dir):
        os.makedirs(output_target_dir)
      # Create yolo annotation files from BUSI_master dataset
      # ./BUSI_master
      create_yolo_annotation(input_sub_dir, output_target_dir, debug=debug)

  except:
    traceback.print_exc()
    