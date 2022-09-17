import csv
import re
from pathlib import Path
from multiprocessing import Pool
import subprocess


def execute(cmd):
    try:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE).stdout.read().decode('utf-8')
    except Exception as e:
        print(e)
        return

if __name__ == "__main__":
    with open('video_add.csv', encoding="utf-8") as csvfile:
        lines = list(csv.reader(csvfile))
    update_dict = {}
    for l in lines:
        split_info = l[2].strip().split("\n")
        if split_info and not re.match("^.*?\d+:\d+.*?", split_info[0]):
            split_info[0] += " 0:00"
        
        curl_id, ytb_id = l[0], re.search("www.youtube.com/watch\?v=(.*)", l[1]).group(1)
        video_list = []
        for c, mov in enumerate(split_info):
            mov_info = re.findall("(.*?)[(\[]?(\d+:\d+)[)\]]?(.*)", mov)[0]
            mov_name = re.sub("\d+:\d+", "", (mov_info[0]+mov_info[2]).replace(" - ", "")).strip()
            mov_start = mov_info[1].split(":")
            mov_start = int(mov_start[0])*60+int(mov_start[1])
            vid = ytb_id+str(c+1)
            video_list.append({'youtube_id': ytb_id, "video_id": vid, "movement_id": c+1, "start": mov_start, "duration" : 0,
                               "title": "", "name": mov_name, "got": "1"})
        update_dict[curl_id] = video_list
    for curl_id in update_dict.keys():
        for c, mov in enumerate(update_dict[curl_id]):
            if c+1 < len(update_dict[curl_id]):
                end = update_dict[curl_id][c+1]["start"]
                nxt = update_dict[curl_id][c+1]["video_id"]
            else:
                end = nxt = 0
            update_dict[curl_id][c].update({"end": end, "next_video": nxt})

    p = Path('video_add')
    commands = []
    for k in update_dict.keys():
        for v in update_dict[k]:
            fn = list(p.glob(f"{v['youtube_id']}.*"))
            if len(fn) == 1:
                fn = str(fn[0]).replace('\\', '/')
            else:
                print(k, fn)
            fn2 = f"split_add/{v['video_id']}.ogg"
            start, end = v["start"], v["end"]
            duration = max(0, end-start)
            commands.append(f"ffmpeg -vn -ss {start} -t {duration} -i {fn} {fn2}")
    with Pool(10) as pool:
        pool.map(execute, commands)
