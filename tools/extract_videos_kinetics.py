#!/usr/bin/env python3

import argparse
import os
import json
import numpy as np
import skvideo.io
import cv2
import sys
import concurrent.futures
from shutil import copyfile
import subprocess
from tqdm import tqdm
from random import shuffle

parser = argparse.ArgumentParser(description='[Something-Something-V2] Video conversion')
parser.add_argument('-i', '--input_root', help='location of input video and csv files.', type=str)
parser.add_argument('-o', '--output_root', type=str, help='output image locations, '
                                                          'generates `train` `val` `test` folders')
parser.add_argument('-s', '--shorter_side', default=[256], type=int, nargs="+",
                    help='shorter side of the generated image, if two values are provided, '
                         'convert to the targeted size regardless the aspect ratio. [w,h]')
parser.add_argument('-p', '--num_processes', type=int, help='number of processor', default=36)
parser.add_argument('-n', '--num_classes', type=int, help='number of classes',
                    choices=[400, 600], default=400)
parser.add_argument('--do-test-set', action='store_true', help='convert test set')

args = parser.parse_args()

# input
# is_400 = True
# shorter_side = 331
# image_format = 'jpg'

train_file = "{}/data/kinetics-{}_train.csv".format(args.input_root, args.num_classes)
val_file = "{}/data/kinetics-{}_val.csv".format(args.input_root, args.num_classes)
test_file = "{}/data/kinetics-{}_test.csv".format(args.input_root, args.num_classes)
train_video_folder = "{}/train".format(args.input_root)
val_video_folder = "{}/val".format(args.input_root)
test_video_folder = "{}/test".format(args.input_root)

# output
label_file = "{}/images/kinetics-{}_label.txt".format(args.output_root, args.num_classes)
train_img_folder = "{}/images/train/".format(args.output_root)
val_img_folder = "{}/images/val/".format(args.output_root)
test_img_folder = "{}/images/test/".format(args.output_root)
train_file_list = "{}/train_{}.txt".format(args.output_root, args.num_classes)
val_file_list = "{}/val_{}.txt".format(args.output_root, args.num_classes)
test_file_list = "{}/test_{}.txt".format(args.output_root, args.num_classes)

train_fail_file_list = "{}/train_fail_{}.txt".format(args.output_root, args.num_classes)
val_fail_file_list = "{}/val_fail_{}.txt".format(args.output_root, args.num_classes)
test_fail_file_list = "{}/test_fail_{}.txt".format(args.output_root, args.num_classes)


if not os.path.exists(os.path.join(args.output_root, 'images')):
    os.makedirs(os.path.join(args.output_root, 'images'))

def load_video_list(file_path, build_label=False):
    labels = []
    videos = []
    with open(file_path) as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            if args.num_classes == 400:
                label, youtube_id, start_time, end_time, temp, _ = line.split(",")
                label = label.replace("\"", "")
            else:
                label, youtube_id, start_time, end_time, temp = line.split(",")
            if temp.strip() == 'split':
                continue
            label_name = label.strip()
            video_id = "{}_{:06d}_{:06d}".format(youtube_id, int(start_time), int(end_time))
            videos.append([video_id, label_name])
            labels.append(label_name)
    if not build_label:
        return videos
    labels = sorted(list(set(labels)))
    id_to_label = {}
    label_to_id = {}
    with open(label_file, 'w') as f:
        for i in range(len(labels)):
            label_to_id[labels[i]] = i
            id_to_label[i] = labels[i]
            print(labels[i], file=f)
    return videos, label_to_id, id_to_label


def load_test_video_list(file_path):
    videos = []
    with open(file_path) as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            youtube_id, start_time, end_time, temp = line.split(",")
            if temp.strip() == 'split':
                continue
            video_id = "{}_{:06d}_{:06d}".format(youtube_id, int(start_time), int(end_time))
            videos.append([video_id, "x"])
    return videos


train_videos, label_to_id, id_to_label = load_video_list(train_file, build_label=True)
val_videos = load_video_list(val_file)
test_videos = load_test_video_list(test_file)


def video_to_images(video, basedir, targetdir, shorter_side):
    try:
        cls_id = label_to_id[video[1]]
        filename = os.path.join(basedir, video[1], video[0] + ".mp4")
        output_foldername = os.path.join(targetdir, video[1], video[0])
    except Exception as e: # for test videos
        cls_id = -1
        filename = os.path.join(basedir, video[0] + ".mp4")
        output_foldername = os.path.join(targetdir, video[0])

    if not os.path.exists(filename):
        print("{} is not existed.".format(filename))
        return video[0], video[1], -2
    else:
        try:
            video_meta = skvideo.io.ffprobe(filename)
            height = int(video_meta['video']['@height'])
            width = int(video_meta['video']['@width'])
        except:
            print("Can not get video info: {}".format(filename))
            return video[0], video[1], 0

        if len(shorter_side) == 1:
            if width > height:
                scale = "scale=-1:{}".format(shorter_side[0])
            else:
                scale = "scale={}:-1".format(shorter_side[0])
        else:
            scale = "scale={}:{}".format(shorter_side[0], shorter_side[1])
        if not os.path.exists(output_foldername):
            os.makedirs(output_foldername)

        command = ['ffmpeg',
                   '-i', '"%s"' % filename,
                   '-vf', scale,
                   '-threads', '1',
                   '-loglevel', 'panic',
                   '-q:v', '0',
                   '"{}/'.format(output_foldername) + '%05d.jpg"']
        command = ' '.join(command)
        try:
            # print(command)
            subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        except:
            print("fail to convert {}".format(filename))
            return video[0], video[1], 0

        # get frame num
        i = 0
        while True:
            img_name = os.path.join(output_foldername, "{:05d}.jpg".format(i + 1))
            if os.path.exists(img_name):
                i += 1
            else:
                break

        frame_num = i
        # print("Finish {}, id: {} frames: {}".format(filename, cls_id, frame_num))
        return video[0], cls_id, frame_num


def create_train_video(shorter_side):
    print("Resizing to shorter side: {}".format(shorter_side))
    with open(train_file_list, 'w') as f, open(train_fail_file_list,
                                               'w') as f_w, concurrent.futures.ProcessPoolExecutor(
            max_workers=64) as executor:
        futures = [executor.submit(video_to_images, video, train_video_folder, train_img_folder,
                                   shorter_side)
                   for video in train_videos]
        total_videos = len(futures)
        curr_idx = 0
        print("label,youtube_id,time_start,time_end,split", file=f_w, flush=True)
        for future in concurrent.futures.as_completed(futures):
            video_id, label_id, frame_num = future.result()
            if frame_num == 0:
                youtube_id = video_id[:11]
                time_start = int(video_id[12:18])
                time_end = int(video_id[19:25])
                print("{},{},{},{},train".format(label_id, youtube_id, time_start, time_end),
                      file=f_w, flush=True)
            elif frame_num == -2:
                youtube_id = video_id[:11]
                time_start = int(video_id[12:18])
                time_end = int(video_id[19:25])
                print("{},{},{},{},train,missed".format(label_id, youtube_id, time_start, time_end),
                      file=f_w, flush=True)
            else:
                print("{};1;{};{}".format(
                    os.path.join('images/train', id_to_label[label_id], video_id), frame_num,
                    label_id), file=f, flush=True)
            if curr_idx % 1000 == 0:
                print("{}/{}".format(curr_idx, total_videos), flush=True)
            curr_idx += 1
    print("Completed")


def create_val_video(shorter_side):
    print("Resizing to shorter side: {}".format(shorter_side))
    with open(val_file_list, 'w') as f, open(val_fail_file_list,
                                             'w') as f_w, concurrent.futures.ProcessPoolExecutor(
            max_workers=36) as executor:
        futures = [
            executor.submit(video_to_images, video, val_video_folder, val_img_folder, shorter_side)
            for video in val_videos]
        total_videos = len(futures)
        curr_idx = 0
        print("label,youtube_id,time_start,time_end,split", file=f_w, flush=True)
        for future in concurrent.futures.as_completed(futures):
            video_id, label_id, frame_num = future.result()
            if frame_num == 0:
                youtube_id = video_id[:11]
                time_start = int(video_id[12:18])
                time_end = int(video_id[19:25])
                print("{},{},{},{},val".format(label_id, youtube_id, time_start, time_end),
                      file=f_w, flush=True)
                # print("{},{},{},{},val".format(label_id, youtube_id, time_start, time_end), flush=True)
            elif frame_num == -2:
                youtube_id = video_id[:11]
                time_start = int(video_id[12:18])
                time_end = int(video_id[19:25])
                print("{},{},{},{},val,missed".format(label_id, youtube_id, time_start, time_end),
                      file=f_w, flush=True)
            else:
                print("{};1;{};{}".format(
                    os.path.join('images/val', id_to_label[label_id], video_id), frame_num,
                    label_id), file=f, flush=True)
            if curr_idx % 1000 == 0:
                print("{}/{}".format(curr_idx, total_videos))
            curr_idx += 1
    print("Completed")


def create_test_video(shorter_side):
    with open(test_file_list, 'w') as f, open(test_fail_file_list,
                                              'w') as f_w, concurrent.futures.ProcessPoolExecutor(
            max_workers=36) as executor:
        futures = [executor.submit(video_to_images, video, test_video_folder, test_img_folder,
                                   shorter_side)
                   for video in test_videos]
        total_videos = len(futures)
        curr_idx = 0
        print("youtube_id,time_start,time_end,split", file=f_w, flush=True)
        for future in concurrent.futures.as_completed(futures):
            video_id, label_id, frame_num = future.result()
            if frame_num == 0:
                youtube_id = video_id[:11]
                time_start = int(video_id[12:18])
                time_end = int(video_id[19:25])
                print("{},{},{},test".format(youtube_id, time_start, time_end), file=f_w,
                      flush=True)
            else:
                print("{} 1 {}".format(os.path.join('images/test', video_id), frame_num), file=f,
                      flush=True)
            print("{}/{}".format(curr_idx, total_videos))
            curr_idx += 1
    print("Completed")


if __name__ == "__main__":
    create_train_video(args.shorter_side)
    create_val_video(args.shorter_side)
    if args.do_test_set:
        create_test_video(args.shorter_side)
