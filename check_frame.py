import os
import json
import numpy as np
import torch
import av
from decord import VideoReader, cpu
from PIL import Image
import random

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
    images=fetch_images(qa_item)
    # qa_item = future_to_images[future]
    image_id = qa_item['image_id'].replace('.mp4', '')
    image_dir = qa_item['data_id'].replace(qa_item['image_id'], '').replace('valley', 'valley_images')

    for i, image in enumerate(images):
        img_file = f"{image_id}_{i}.png"
        images[i].save(os.path.join(image_dir, img_file))


if __name__ == "__main__":
    data_name='valley'
    root='/data/luogen_code/LLaVA-robust/playground/data/video/Video-LLaVA/'+data_name
    qa_items=[]
    for root, dirs, files in os.walk(root):
        for name in files:
            if 'mp4' in name:
                qa_items.append({'data_id':os.path.join(root, name), 'image_id':name,'image_path':root.split('/')[-1]+'/'+name })

    count=0
    undecoder_lists=[]
    devoder_lists=set()
    fail_list=[]
    print(len(qa_items))
    for qa_item in qa_items:
        image_id = qa_item['image_id'].replace('.mp4', '')
        image_dir = qa_item['data_id'].replace(qa_item['image_id'], '').replace(data_name,
                                                                                data_name+'_images')
        # if 'v_OHNH7IV0768' in image_id:
        #     print(image_id)
        for i in range(8):
            img_file = f"{image_id}_%d.png"%i
            try:
                image = Image.open(os.path.join(image_dir, img_file)).convert('RGB')
            except:
                fail_list.append(image_id)
                break
        if not os.path.exists(os.path.join(image_dir, img_file)):
            # print(os.path.join(image_dir, img_file))
            undecoder_lists.append(qa_item['image_path'] if data_name=='valley' else image_id)
            count+=1
            # print(os.path.join(image_dir, img_file))
            continue
        else:
            devoder_lists.add(qa_item['image_path'] if data_name=='valley' else image_id)
    print(undecoder_lists)
    print(count//8)
    print(len(qa_items))
    print(fail_list)
    json.dump(fail_list,open('fail_list.json','w'))
    undecoder_lists+=fail_list

    #filter valley  json
    # anns=json.load(open('./Valley-webvid2M-Pretrain-703K/chat.json'))
    # print(len(anns))
    # new_anns=[]
    # for item in anns:
    #     if item['video'] in devoder_lists:
    #         # print(item['video'])
    #         item_=item.copy()
    #         item_['image'] = 'valley_images/' + item['video'].replace('.mp4','_4.png')
    #         item_['conversations'][0]['value']=item_['conversations'][0]['value'].replace('<video>','<image>')
    #         new_anns.append(item_)
    # print(len(new_anns))
    # json.dump(new_anns,open('./Valley-webvid2M-Pretrain-703K/chat_filter.json','w'))



    #filter video instruct  json
    anns=json.load(open('./VideoInstruct-100K/VideoInstruct100K.json'))
    new_anns=[]
    for item in anns:
        if item['video_id'] in devoder_lists:
            item_={}
            conv=[]
            item_['image']='videochatgpt_tune_images/'+item['video_id']#+'_4.png'
            item_['id']=item['video_id']
            conv.append({'from': 'human', 'value': '<image>\n' + item['q']})
            conv.append({'from': 'gpt', 'value': item['a']})
            item_['conversations']=conv
            new_anns.append(item_)
    print(len(anns))
    print(len(new_anns))
    json.dump(new_anns,open('./VideoInstruct-100K/VideoInstruct100K_filter.json','w'))