import os
import json
import numpy as np
import torch
import av
from decord import VideoReader, cpu
from PIL import Image
import random
import cv2
from tqdm.auto import tqdm
import concurrent.futures


num_segments = 8

# root directory of evaluation dimension 10
dimension10_dir = "./videos/20bn-something-something-v2"
# root directory of evaluation dimension 11
dimension11_dir = "./videos/EPIC-KITCHENS"
# root directory of evaluation dimension 12
dimension12_dir = "./videos/BreakfastII_15fps_qvga_sync"

def transform_video(buffer):
    try:
        buffer = buffer.numpy()
    except AttributeError:
        try:
            buffer = buffer.asnumpy()
        except AttributeError:
            print("Both buffer.numpy() and buffer.asnumpy() failed.")
            buffer = None
    images_group = list()
    for fid in range(len(buffer)):
        images_group.append(Image.fromarray(buffer[fid]))
    del buffer
    return images_group

def get_index(num_frames, num_segments):
    if num_segments > num_frames:
        offsets = np.array([
            idx for idx in range(num_frames)
        ])
    else:
        # uniform sampling
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array([
            start + int(np.round(seg_size * idx)) for idx in range(num_segments)
        ])
    return offsets


def fetch_images(qa_item):
    use_pyav = False
    segment = None
    data_path =  qa_item['data_id']
    start = 0.0
    end = 0.0
    # using decord for decoding videos in evaluation dimension 10-11
    try:
        vr = VideoReader(data_path, num_threads=1, ctx=cpu(0))
        video_len = len(vr)
        fps = vr.get_avg_fps()
        if segment is not None:
            # obtain start and end frame for the video segment in evaluation dimension 11
            start_frame = int(min(max(start * fps, 0), video_len - 1))
            end_frame = int(min(max(end * fps, 0), video_len - 1))
            tot_frames = int(end_frame - start_frame)
            offset = get_index(tot_frames, num_segments)
            frame_indices = offset + start_frame
        else:
            # sample frames of the video in evaluation dimension 10
            frame_indices = get_index(video_len - 1, num_segments)
        vr.seek(0)
        buffer = vr.get_batch(frame_indices)
        del vr
        del buffer
        return True
    except:
        return False
    return False


def fetch_images_parallel(qa_item):
    # image_id = qa_item['image_id'].replace('.avi', '').replace('.mkv', '').replace('.mp4', '').replace('.webm','')
    # image_dir = qa_item['data_id'].replace(qa_item['image_id'], '').replace('all_test', 'images')
    valid=fetch_images(qa_item)
    return qa_item,valid


if __name__ == "__main__":
    data_base='valley'
    root='/data/luogen_code/LLaVA-robust/playground/data/video/Video-LLaVA/'+data_base
    qa_items=[]
    decoder_lists=set()
    for root, dirs, files in os.walk(root):
        for name in files:
            if 'mp4' in name:
                qa_items.append({'data_id':os.path.join(root, name), 'video_id':root.split('/')[-1]+'/'+name.replace('.mp4','') })
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_images = {executor.submit(fetch_images_parallel, qa_item): qa_item for qa_item in qa_items}
        for future in tqdm(concurrent.futures.as_completed(future_to_images), total=len(future_to_images)):
            qa_item, valid = future.result()
            # print(qa_item['video_id'])
            if valid:
                decoder_lists.add(qa_item['video_id'])

    # filter valley  json
    anns=json.load(open('./Valley-webvid2M-Pretrain-703K/chat.json'))
    print(len(anns))
    new_anns=[]
    for item in anns:
        if item['video'].replace('.mp4','') in decoder_lists:
            # print(item['video'])
            item_=item.copy()
            item_['video'] = 'valley/' + item['video']
            item_['conversations'][0]['value']=item_['conversations'][0]['value'].replace('<video>','<image>')
            new_anns.append(item_)
    print(len(new_anns))
    json.dump(new_anns,open('./Valley-webvid2M-Pretrain-703K/chat_filter.json','w'))
    #
    # # filter video instruct  json
    # anns = json.load(open('./VideoInstruct-100K/VideoInstruct100K.json'))
    # new_anns = []
    # for item in anns:
    #     if item['video_id'] in decoder_lists:
    #         item_ = {}
    #         conv = []
    #         item_['video'] = 'videochatgpt_tune/' + item['video_id']+'.mp4'
    #         item_['id'] = item['video_id']
    #         conv.append({'from': 'human', 'value': '<image>\n' + item['q']})
    #         conv.append({'from': 'gpt', 'value': item['a']})
    #         item_['conversations'] = conv
    #         new_anns.append(item_)
    # print(len(anns))
    # print(len(new_anns))
    # json.dump(new_anns, open('./VideoInstruct-100K/VideoInstruct100K_filter.json', 'w'))
