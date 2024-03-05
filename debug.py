from PIL import Image
from decord import VideoReader, cpu
import numpy as np

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
        images_group.append(Image.fromarray(buffer[fid]).convert('RGB'))
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

def preprocess_video(video_path,num_segments=4):
    start = 0.0
    end = 0.0
    vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    video_len = len(vr)
    print(video_len)
    fps = vr.get_avg_fps()
    # obtain start and end frame for the video segment in evaluation dimension 11
    frame_indices = get_index(video_len - 1, num_segments)
    vr.seek(0)
    buffer = vr.get_batch(frame_indices)
    print(buffer)
    video = transform_video(buffer)
    return video

video=preprocess_video('/data/data/images/valley/033951_034000/26015456.mp4',num_segments=4)
print(video)