import argparse
import os
from random import shuffle

# parser = argparse.ArgumentParser()
# parser.add_argument('--folder_path', default='./training_data/CTimg', type=str,
#                     help='The folder path')
# parser.add_argument('--train_filename', default='./data_flist/train_shuffled.flist', type=str,
#                     help='The train filename.')
# parser.add_argument('--validation_filename', default='./data_flist/validation_shuffled.flist', type=str,
#                     help='The validation filename.')
# parser.add_argument('--is_shuffled', default='1', type=int,
#                     help='Needed to be shuffled')

if __name__ == "__main__":

    # args = parser.parse_args()
    dir_name = './validation_data_10_imgs/'
    # get the list of directories and separate them into 2 types: training and validation
    training_dirs = os.listdir(dir_name)
    # validation_dirs = os.listdir('./validation_data_10_imgs' + "/validation")

    # make 2 lists to save file paths
    training_file_names = []
    training_folder = ''
    i = 0
    # append all files into 2 lists
    for training_dir in sorted(training_dirs):
        i += 1
        # append each file into the list file names
        training_folder = training_folder + ' ' + dir_name + training_dir
        print(training_folder)

        if i % 3 == 0:

            training_file_names.append(training_folder)
            training_folder = ''
            i = 0
        # for training_item in training_folder:
        #     # modify to full path -> directory
        #     training_item = args.folder_path + "/training" + "/" + training_dir + "/" + training_item
        #     training_file_names.append(training_item)
    fo = open('validation_10_imgs.txt', "w")
    fo.write("\n".join(training_file_names))
    fo.close()

    # make output file if not existed
    # if not os.path.exists(args.train_filename):
    #     os.mknod(args.train_filename)
    #
    # if not os.path.exists(args.validation_filename):
    #     os.mknod(args.validation_filename)
    #
    # # write to file
    # fo = open(args.train_filename, "w")
    # fo.write("\n".join(training_file_names))
    # fo.close()
    #
    # fo = open(args.validation_filename, "w")
    # fo.write("\n".join(validation_file_names))
    # fo.close()
    #
    # # print process
    # print("Written file is: ", args.train_filename, ", is_shuffle: ", args.is_shuffled)


