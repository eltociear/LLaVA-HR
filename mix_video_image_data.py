import json

def change_video_keys(video_anns):
    for i in range(len(video_anns)):
        video_anns[i]['video']=video_anns[i].pop('image')
        video_anns[i]['video']=video_anns[i]['video'].replace('_4.png','')
    return video_anns
video_instruct_path='/data/luogen_code/LLaVA-robust/playground/data/video/Video-LLaVA/VideoInstruct-100K/VideoInstruct100K_filter.json'
image_instruct_path='/data/luogen_code/LLaVA-robust/playground/data/llava_v1_5_mix665k.json'

video_pretrain_path='/data/luogen_code/LLaVA-robust/playground/data/video/Video-LLaVA/Valley-webvid2M-Pretrain-703K/chat_filter.json'
image_pretrain_path='/data/data/blip_laion_cc_sbu_558k.json'

video_instruct_file=json.load(open(video_instruct_path))
image_instruct_file=json.load(open(image_instruct_path))
new_instruct_anns=[]
new_instruct_anns.extend(change_video_keys(video_instruct_file))
new_instruct_anns.extend(image_instruct_file)
print(len(image_instruct_file))
print(len(new_instruct_anns))
json.dump(new_instruct_anns,open(image_instruct_path.replace('llava_v1_5_mix665k','llava_v1_5_mix665k_video'),'w'))



video_pretrain_file=json.load(open(video_pretrain_path))
image_pretrain_file=json.load(open(image_pretrain_path))
new_pretrain_anns=[]
new_pretrain_anns.extend(change_video_keys(video_pretrain_file))
new_pretrain_anns.extend(image_pretrain_file)
print(len(image_pretrain_file))
print(len(new_pretrain_anns))
json.dump(new_pretrain_anns,open(image_pretrain_path.replace('blip_laion_cc_sbu_558k','blip_laion_cc_sbu_558k_video'),'w'))