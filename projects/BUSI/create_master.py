#
# create_master.py
# 2023/04/10 Antillia.com Toshiyuki Arai
#
# This splits the original Dataset_BUSI_with_GT dataset 
# to three subsets (0.6) train (0.2), test (0.2) and valid. 
#    
import sys
import os
import glob
import random
import shutil
import traceback


def create_master(input_dir, output_dir):
  class_dirs = ["benign", "malignant"]
  for class_dir  in class_dirs:
    images_dir = os.path.join(input_dir, class_dir)
    pattern = images_dir + "/*mask.png"
    
    print("--- pattern {}".format(pattern))
    mask_files  = glob.glob(pattern)
    num_files   = len(mask_files)
    # 1 shuffle mask_files
    random.shuffle(mask_files)
    
    # 2 Compute the number of images to split
    # train= 0.6 test=0.2 valid=0.2
    num_train = int(num_files * 0.6)
    num_test  = int(num_files * 0.2)
    num_valid = int(num_files * 0.2)

    train_files = mask_files[0: num_train]
    test_files  = mask_files[num_train: num_train+num_test]
    valid_files = mask_files[num_train+num_test: num_files]

    print("=== number of train_files {}".format(len(train_files)))
    print("=== number of test_files  {}".format(len(test_files)))
    print("=== number of valid_files {}".format(len(valid_files)))

    copy_files(images_dir, output_dir, class_dir, train_files, "train")
    copy_files(images_dir, output_dir, class_dir, test_files,  "test")
    copy_files(images_dir, output_dir, class_dir, valid_files, "valid")

   
def copy_files(images_dir, output_dir, class_dir, mask_files, target): 
  # target = train_or_test_or_valid_dir:

  output_class_dir  = os.path.join(output_dir, target)
  target_output_dir = os.path.join(output_class_dir, class_dir)
  if not os.path.exists(target_output_dir):
    os.makedirs(target_output_dir)
  for mask_file in mask_files:
    # 1 Copy mask file to target_output_dir
    shutil.copy2(mask_file, target_output_dir)
    basename  = os.path.basename(mask_file)
    extension = basename.split(".")[1]

    non_mask_filename  = basename.split("_")[0] + "." + extension
    non_mask_file_path = os.path.join(images_dir, non_mask_filename)
    # 2 Copy non_mask file to target_output_dir
    shutil.copy2(non_mask_file_path, target_output_dir)


"""
Input:
./DATASET_BUSI_WITH_GT/
├─benign/
├─malignant/
└─normal/

"""

"""
Output:
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
    input_dir  = "./Dataset_BUSI_with_GT"
    output_dir = "./BUSI_master"
    if not os.path.exists(input_dir):
      raise Exception("===NOT FOUND " + input_dir)
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
      
    # create BUS_master dataset with train, test, valid from orignal Dataset_BUSI_with_GT .
    create_master(input_dir, output_dir)

  except:
    traceback.print_exc()
    