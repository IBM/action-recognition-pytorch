#!/usr/bin/env python3

import os
import json
import skvideo.io
import concurrent.futures
import subprocess

folder_root = ""

if folder_root == "":
    raise ValueError("Please set folder_root")

# input
label_file = "{}/something-something-v2-labels.json".format(folder_root)
train_file = "{}/something-something-v2-train.json".format(folder_root)
val_file = "{}/something-something-v2-validation.json".format(folder_root)
test_file = "{}/something-something-v2-test.json".format(folder_root)
video_folder = "{}/20bn-something-something-v2".format(folder_root)

# output
train_img_folder = "{}/train".format(folder_root)
val_img_folder = "{}/val".format(folder_root)
test_img_folder = "{}/test".format(folder_root)
train_file_list = "{}/train.txt".format(folder_root)
val_file_list = "{}/val.txt".format(folder_root)
test_file_list = "{}/test.txt".format(folder_root)

def load_categories(file_path):
    id_to_label = {}
    label_to_id = {}
    with open(file_path) as f:
        labels = json.load(f)
        for label, cls_id in labels.items():
            label = label
            id_to_label[int(cls_id)] = label
            label_to_id[label] = int(cls_id)
    return id_to_label, label_to_id


id_to_label, label_to_id = load_categories(label_file)


def load_video_list(file_path):
    videos = []
    with open(file_path) as f:
        file_list = json.load(f)
        for temp in file_list:
            videos.append([temp['id'], temp['template'].replace(
                "[", "").replace("]", ""), temp['label'], temp['placeholders']])
    return videos


def load_test_video_list(file_path):
    videos = []
    with open(file_path) as f:
        file_list = json.load(f)
        for temp in file_list:
            videos.append([temp['id']])
    return videos


train_videos = load_video_list(train_file)
val_videos = load_video_list(val_file)
test_videos = load_test_video_list(test_file)


def resize_to_short_side(h, w, short_side=360):
    newh, neww = h, w
    if h < w:
        newh = short_side
        neww = (w / h) * newh
    else:
        neww = short_side
        newh = (h / w) * neww
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return newh, neww


def video_to_images(video, basedir, targetdir, short_side=256):
    try:
        cls_id = label_to_id[video[1]]
    except:
        cls_id = -1
    filename = os.path.join(basedir, video[0] + ".webm")
    output_foldername = os.path.join(targetdir, video[0])
    if not os.path.exists(filename):
        print("{} is not existed.".format(filename))
        return video[0], cls_id, 0
    else:
        try:
            video_meta = skvideo.io.ffprobe(filename)
            height = int(video_meta['video']['@height'])
            width = int(video_meta['video']['@width'])
        except:
            print("Can not get video info: {}".format(filename))
            return video[0], cls_id, 0

        if width > height:
            scale = "scale=-1:{}".format(short_side)
        else:
            scale = "scale={}:-1".format(short_side)
        if not os.path.exists(output_foldername):
            os.makedirs(output_foldername)

        command = ['ffmpeg',
                   '-i', '"%s"' % filename,
                   '-vf', scale,
                   '-threads', '1',
                   '-loglevel', 'panic', '-qmin', '1', '-qmax', '1',
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
        return video[0], cls_id, frame_num


def create_train_video(short_side):
    with open(train_file_list, 'w') as f, concurrent.futures.ProcessPoolExecutor(max_workers=36) as executor:
        futures = [executor.submit(video_to_images, video, video_folder, train_img_folder, int(short_side))
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


def create_val_video(short_side):
    with open(val_file_list, 'w') as f, concurrent.futures.ProcessPoolExecutor(max_workers=36) as executor:
        futures = [executor.submit(video_to_images, video, video_folder, val_img_folder, int(short_side))
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


def create_test_video(short_side):
    with open(test_file_list, 'w') as f, concurrent.futures.ProcessPoolExecutor(max_workers=36) as executor:
        futures = [executor.submit(video_to_images, video, video_folder, test_img_folder, int(short_side))
                   for video in test_videos]
        total_videos = len(futures)
        curr_idx = 0
        for future in concurrent.futures.as_completed(futures):
            video_id, label_id, frame_num = future.result()
            if frame_num == 0:
                print("Something wrong: {}".format(video_id))
            else:
                print("{} 1 {}".format(os.path.join(test_img_folder, video_id), frame_num), file=f, flush=True)
            print("{}/{}".format(curr_idx, total_videos))
            curr_idx += 1
    print("Completed")


create_train_video(256)
create_val_video(256)
create_test_video(256)