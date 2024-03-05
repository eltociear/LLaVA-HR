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

    # cv2_vr = cv2.VideoCapture(data_path)
    # duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
    # frame_id_list = np.linspace(0, duration - 1, num_segments, dtype=int)
    #
    # video_data = []
    # for frame_idx in frame_id_list:
    #     cv2_vr.set(1, frame_idx)
    #     _, frame = cv2_vr.read()
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     video_data.append(frame)
    # cv2_vr.release()


    if use_pyav:
        # using pyav for decoding videos in evaluation dimension 12
        reader = av.open(data_path)
        frames = [torch.from_numpy(f.to_rgb().to_ndarray()) for f in reader.decode(video=0)]
        video_len = len(frames)
        start_frame, end_frame = start, end
        end_frame = min(end_frame, video_len)
        offset = get_index(end_frame - start_frame, num_segments)
        frame_indices = offset + start_frame
        video = torch.stack([frames[idx] for idx in frame_indices])
    else:
        # using decord for decoding videos in evaluation dimension 10-11
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
        video=transform_video(buffer)
        del vr
        del buffer
    return video


def fetch_images_parallel(qa_item):
    image_id = qa_item['image_id'].replace('.mp4', '')
    image_dir = qa_item['data_id'].replace(qa_item['image_id'], '').replace('valley', 'valley_images')

    images=fetch_images(qa_item)
    os.makedirs(image_dir, exist_ok=True)
    for i, image in enumerate(images):
        img_file = f"{image_id}_{i}.png"
        images[i].save(os.path.join(image_dir, img_file))


if __name__ == "__main__":
    root='/data/luogen_code/LLaVA-robust/playground/data/video/Video-LLaVA/valley'
    video_img_dir = '/data/luogen_code/LLaVA-robust/playground/data/video/Video-LLaVA/valley_images'
    qa_items=[]
    fail_list=json.load(open('fail_list.json'))
    # print(fail_list)
    for root, dirs, files in os.walk(root):
        for name in files:
            for fail_file in fail_list:
                if 'mp4' in name and fail_file in name:
                    qa_items.append({'data_id':os.path.join(root, name), 'image_id':name })
    print(len(qa_items))
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_images = {executor.submit(fetch_images_parallel, qa_item): qa_item for qa_item in qa_items}
        for future in tqdm(concurrent.futures.as_completed(future_to_images), total=len(future_to_images)):
            pass
            # qa_item = future_to_images[future]
            # image_id=qa_item['image_id'].replace('.mp4','')
            # image_dir=qa_item['data_id'].replace(qa_item['image_id'],'').replace('valley','valley_images')
            # os.makedirs(image_dir,exist_ok=True)
            # try:
            #     qa_item, images = future.result()
            # except Exception as exc:
            #     print(f'{qa_item} generated an exception: {exc}')
            # else:
            #     for i,image in enumerate(images):
            #         img_file = f"{image_id}_{i}.png"
            #         images[i].save(os.path.join(image_dir, img_file))
            #     del images
