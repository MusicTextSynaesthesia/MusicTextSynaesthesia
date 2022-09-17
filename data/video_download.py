from multiprocessing import Pool
import os
import pathlib
import subprocess
import json
import re


def get_video_src(vid):
    for i in range(3):
        video_src = subprocess.Popen('you-get -o video_missing --json "https://www.youtube.com/watch?v=%s"' % vid, stdout=subprocess.PIPE).stdout.read().decode('utf-8')
        if video_src != "":
            return json.loads(video_src)
    else:
        return None


def get_video(vid):
    if len(list(pathlib.Path("./video_missing").glob(f"{vid}.*"))) > 0:
        return "got"
    video_src = get_video_src(vid)
    if video_src:
        video_src = video_src['streams']
        itags = {k: video_src[k]['size'] for k in video_src.keys() if video_src[k].__contains__('size')}
        itag = sorted(itags.keys(), key=itags.__getitem__)[0]
        os.system('you-get -O "video_missing/%s" --itag=%s "https://www.youtube.com/watch?v=%s"' % (vid, itag, vid))
    if len(list(pathlib.Path("./video_missing").glob(f"{vid}.*"))) > 0:
        return "got"
    return ""


if __name__ == "__main__":
    video_list = []
    with open("missing_video.txt") as f:
        lines = list(f.readlines())
        for l in lines:
            v = l.strip()
            video_list.append(v)
    print(video_list)
    with Pool(10) as pool:
        pool.map(get_video, video_list)
    # for v in video_list:
        # get_video(v)
