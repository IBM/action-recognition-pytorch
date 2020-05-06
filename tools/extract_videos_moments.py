#!/usr/bin/env python3

import os
import concurrent.futures
from shutil import copyfile
import subprocess

input_folder_root = ""
if input_folder_root == "":
    raise ValueError("Please set input_folder_root")

output_folder_root = ""
if output_folder_root == "":
    raise ValueError("Please set output_folder_root")

# input
label_file = "{}/moments_categories.txt".format(input_folder_root)
train_file = "{}/trainingSet.csv".format(input_folder_root)
val_file = "{}/validationSet.csv".format(input_folder_root)
video_folder = input_folder_root

# output
train_img_folder = "{}/train".format(output_folder_root)
val_img_folder = "{}/val".format(output_folder_root)
train_file_list = "{}/train.txt".format(output_folder_root)
val_file_list = "{}/val.txt".format(output_folder_root)


def load_categories(file_path):
    id_to_label = {}
    label_to_id = {}
    with open(file_path) as f:
        for label in f.readlines():
            label = label.strip()
            if label == "":
                continue
            label = label.split(',')
            cls_id = int(label[-1])
            id_to_label[cls_id] = label[0]
            label_to_id[label[0]] = cls_id
    return id_to_label, label_to_id


id_to_label, label_to_id = load_categories(label_file)


def load_video_list(file_path):
    videos = []
    with open(file_path) as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            video_id, label_name, _, _= line.split(",")
            label_name = label_name.strip()
            videos.append([video_id, label_name])
    return videos


train_videos = load_video_list(train_file)
val_videos = load_video_list(val_file)


def video_to_images(video, basedir, targetdir):
    try:
        cls_id = label_to_id[video[1]]
    except:
        cls_id = -1
    assert cls_id >= 0
    filename = os.path.join(basedir, video[0])
    video_basename = video[0].split('.')[0]
    output_foldername = os.path.join(targetdir, video_basename)
    if not os.path.exists(filename):
        print("{} is not existed.".format(filename))
        return video[0], cls_id, 0
    else:
        if not os.path.exists(output_foldername):
            os.makedirs(output_foldername)

        command = ['ffmpeg',
                   '-i', '"%s"' % filename,
                   '-threads', '1',
                   '-loglevel', 'panic',
                   '-q:v', '0',
                   '{}/'.format(output_foldername) + '"%05d.jpg"']
        command = ' '.join(command)
        try:
            subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        except:
            print("fail to convert {}".format(filename))
            return video[0], cls_id, 0

        # get frame num
        i = 0
        while True:
            img_name = os.path.join(output_foldername + "/{:05d}.jpg".format(i + 1))
            if os.path.exists(img_name):
                i += 1
            else:
                break

        frame_num = i
        print("Finish {}, id: {} frames: {}".format(filename, cls_id, frame_num))
        return video_basename, cls_id, frame_num


def create_train_video():
    with open(train_file_list, 'w') as f, concurrent.futures.ProcessPoolExecutor(max_workers=36) as executor:
        futures = [executor.submit(video_to_images, video, os.path.join(video_folder, 'train'), train_img_folder)
                   for video in train_videos]
        total_videos = len(futures)
        curr_idx = 0
        for future in concurrent.futures.as_completed(futures):
            video_id, label_id, frame_num = future.result()
            if frame_num == 0:
                print("Something wrong: {}".format(video_id))
            else:
                print("{} 1 {} {}".format(os.path.join(train_img_folder, video_id), frame_num, label_id), file=f, flush=True)
            print("{}/{}".format(curr_idx, total_videos), flush=True)
            curr_idx += 1
    print("Completed")


def create_val_video():
    with open(val_file_list, 'w') as f, concurrent.futures.ProcessPoolExecutor(max_workers=36) as executor:
        futures = [executor.submit(video_to_images, video, os.path.join(video_folder, 'val'), val_img_folder)
                   for video in val_videos]
        total_videos = len(futures)
        curr_idx = 0
        for future in concurrent.futures.as_completed(futures):
            video_id, label_id, frame_num = future.result()
            if frame_num == 0:
                print("Something wrong: {}".format(video_id))
            else:
                print("{} 1 {} {}".format(os.path.join(val_img_folder, video_id), frame_num, label_id), file=f, flush=True)
            print("{}/{}".format(curr_idx, total_videos))
            curr_idx += 1
    print("Completed")


create_train_video()
create_val_video()
