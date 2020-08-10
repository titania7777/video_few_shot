import os
import av
import glob

videos_path = './UCF101_videos/'
frames_path = './UCF101_frames/'
videos_path_list = glob.glob(videos_path + '*.avi')
print("will start extract frames from {} videos...".format(len(videos_path_list)))

# check directory
assert os.path.exists(frames_path) is False, "{} directory is alreay exist!!".format(frames_path)
os.makedirs(frames_path)

for i, video_path in enumerate(videos_path_list):
    frame_name = video_path.split("\\" if os.name == 'nt' else "/")[-1].split('.avi')[0]
    frame_path = os.path.join(frames_path, frame_name)
    frames = [f.to_image() for f in av.open(video_path).decode(0)]
    os.makedirs(frame_path)
    for j, frame in enumerate(frames):
        frame.save(os.path.join(frame_path, "{}.jpg".format(j)))
    print("{}/{} extracted {} frames from '{}'".format(i, len(videos_path_list), len(frames), frame_name + '.avi'))


