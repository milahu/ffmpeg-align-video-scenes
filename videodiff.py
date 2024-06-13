#!/usr/bin/env python3

"""
compare video frames by structural similarty (ssim)

based on
https://aws.amazon.com/blogs/media/metfc-automatically-compare-two-videos-to-find-common-content/
https://forum.videohelp.com/threads/408408-Is-it-possible-to-compare-2-video-files-frame-by-frame
https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python
"""

import os
import sys
import time
import subprocess
import json
import tempfile
import atexit

import imagehash

from skimage.metrics import structural_similarity
import cv2
import numpy as np
import matplotlib.pyplot as plt

import functools

def wrap_timing(f):
    @functools.wraps(f)
    def wrapper(*args, **kw):
        t1 = time.time()
        result = f(*args, **kw)
        t2 = time.time()
        #print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, t2-t1))
        print('func:%r took %2.6f sec' % (f.__name__, t2-t1))
        return result
    return wrapper

'''
lavfi = ""

# TODO seek seek_point=3,trim=3:duration=5

if height0 == height1 and width0 == width1:
    # same size
    lavfi += "[0:v]null[v0];"
    lavfi += "[1:v]null[v1];"
elif width0 < width1:
    # image 1 is larger
    lavfi += "[0:v]null[v0];"
    lavfi += f"[1:v]scale={width0}:{height0}[v1];"
else:
    # image 0 is larger
    lavfi += f"[0:v]scale={width1}:{height1}[v0];"
    lavfi += "[1:v]null[v1];"

lavfi += "[v0][v1]ssim=f=-;"

args = [
    "ffmpeg",
    "-hide_banner",
    "-loglevel", "error",

    "-i", video0_path,
    #"-ss", "51:00", "-to", "52:00", # no effect?

    "-i", video1_path,
    #"-ss", "51:00", "-to", "52:00", # no effect?

    "-to", "1",

    "-lavfi", lavfi,

    # no output video, write to stdout
    "-f", "null", "-",

    # debug: write frames
    #"-vsync", "vfr", "-frame_pts", "true", "-f", "image2", "debug-frame-%010d.jpg",

]

# FIXME ffmpeg keeps runnign when the script is killed

ffmpeg_proc = subprocess.Popen(
    args,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

for line in iter(ffmpeg_proc.stdout.readline, ""):
    # line: n:248 Y:0.996318 U:0.998400 V:0.998165 All:0.996973 (25.189478)
    sys.stdout.write("line: " + line)
    sys.stdout.flush()
    # get similarity of all channels
    sim = float(line.split(" ")[4][4:])
    frame = int(line.split(" ")[0][2:])
    if sim < 0.5:
        # similarity is too low = diff is too high
        sys.stdout.write("diff: " + line)
        sys.stdout.flush()
        #sys.exit()
    if frame > 1000:
        print("done 1000 frames")
        ffmpeg_proc.kill()
        break

sys.exit()
'''

from skimage.metrics import structural_similarity
import cv2
import numpy as np
import matplotlib.pyplot as plt

# https://github.com/dmlc/decord
# read videos. faster than cv2?
import decord

import PIL

'''
vr = decord.VideoReader('examples/flipping_a_pancake.mkv', ctx=decord.cpu(0))
# a file like object works as well, for in-memory decoding
with open('examples/flipping_a_pancake.mkv', 'rb') as f:
  vr = decord.VideoReader(f, ctx=decord.cpu(0))
print('video frames:', len(vr))
# 1. the simplest way is to directly access frames
for i in range(len(vr)):
    # the video reader will handle seeking and skipping in the most efficient manner
    frame = vr[i]
    print(frame.shape)

# To get multiple frames at once, use get_batch
# this is the efficient way to obtain a long list of frames
frames = vr.get_batch([1, 3, 5, 7, 9])
print(frames.shape)
# (5, 240, 320, 3)
# duplicate frame indices will be accepted and handled internally to avoid duplicate decoding
frames2 = vr.get_batch([1, 2, 3, 2, 3, 4, 3, 4, 5]).asnumpy()
print(frames2.shape)
# (9, 240, 320, 3)

# 2. you can do cv2 style reading as well
# skip 100 frames
vr.skip_frames(100)
# seek to start
vr.seek(0)
batch = vr.next()
print('frame shape:', batch.shape)
print('numpy frames:', batch.asnumpy())
'''

def process_img(image1, image2):
    # Convert images to grayscale
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images, score is between 0 and 1, diff is actuall diff with all floats
    #(score, diff) = structural_similarity(image1, image2, full=True)
    score = structural_similarity(image1, image2)

    return score * 100

# https://pyimagesearch.com/2017/11/27/image-hashing-opencv-python/

def dhash(image, hashSize=8):
	# resize the input image, adding a single column (width) so we
	# can compute the horizontal gradient
	resized = cv2.resize(image, (hashSize + 1, hashSize))
	# compute the (relative) horizontal gradient between adjacent
	# column pixels
	diff = resized[:, 1:] > resized[:, :-1]
	# convert the difference image to a hash
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def read_images(video_reader_1, video_reader_2):
    success1, image1 = video_reader_1.read()
    if not success1:
        return None, None
    success2, image2 = video_reader_2.read()
    if not success2:
        return None, None
    return image1, image2

#@wrap_timing
def hash_image(image):

    # AttributeError: 'numpy.ndarray' object has no attribute 'convert'
    # hash = imagehash.phash(image)

    # big numbers ...
    # hash = dhash(image)

    # reduce image to middle third
    #L = len(image); a = round(L/3); b = round(L*2/3) # TODO refactor
    # reduce image to middle half
    L = len(image); a = round(L/4); b = round(L*3/4) # TODO refactor
    image = image[a:b]

    # convert to gray
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    '''
    # https://stackoverflow.com/a/55786485/10440128
    # hash is cv2.img_hash.BlockMeanHash
    hash = cv2.img_hash.BlockMeanHash_create(); hash.compute(image)
    return hash
    '''

    # https://stackoverflow.com/a/65866320/10440128
    hash = cv2.img_hash.pHash(image) # 8-byte hash
    hash = int.from_bytes(hash.tobytes(), byteorder='big', signed=False)

    return hash

# https://stackoverflow.com/questions/40901906
# black_image = np.zeros((500, 500, 3), dtype = "uint8")
# black_hash = hash_image(black_image) # 0

def compare_hashes(a, b):
    #return a.compare(a, b) # cv2.img_hash.BlockMeanHash # FIXME
    #return a - b
    return abs(a - b)

#@wrap_timing
def compare_images(image1, image2, width, height):
    # smaller is faster
    # width, height = width, height # 2 seconds
    # 1280x720/10 = 128x72
    # 1280x720/8 = 160x90
    # 1280x720/4 = 320x180
    # width, height = width // 4, height // 4 # 0.1 seconds
    # width, height = width // 10, height // 10 # 0.04 seconds
    width, height = width // 8, height // 8 # 0.04 seconds

    # Convert images to grayscale
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # must have same size. smaller is faster
    image1 = cv2.resize(image1, (width, height), interpolation=cv2.INTER_LINEAR)
    image2 = cv2.resize(image2, (width, height), interpolation=cv2.INTER_LINEAR)

    # Compute SSIM between the two images, score is between 0 and 1, diff is actuall diff with all floats
    #(score, diff) = structural_similarity(image1, image2, full=True)
    score = structural_similarity(image1, image2)
    return score
    return score * 100 # percent

# https://stackoverflow.com/a/6822761/10440128
# sliding window iterator
from collections import deque
def window(seq, n=2):
    it = iter(seq)
    #win = deque((next(it, None) for _ in xrange(n)), maxlen=n)
    win = deque((next(it, None) for _ in range(n)), maxlen=n)
    yield win
    append = win.append
    for e in it:
        append(e)
        yield win

# TODO benchmark: cv2.VideoCapture vs decord.VideoReader
# vs https://github.com/chenxinfeng4/ffmpegcv
# vs https://github.com/abhiTronix/vidgear
# TODO also benchmark memory usage. space versus time tradeoff
# TODO cv2 with multithreading
# https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/

# cv2.VideoCapture
def create_video_reader(video_path):
    return cv2.VideoCapture(video_path)

def iter_images(video_reader, start_idx=0):
    print(f"iter_images: seeking to {start_idx}")
    video_reader.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    while True:
        success, image = video_reader.read()
        if not success:
            return # StopIteration?
        frame = int(video_reader.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        yield (frame, image)

def get_width_height(video_reader):
    success, image = video_reader.read()
    assert success, f"failed to read first frame of video_reader {video_reader}"
    frame = int(video_reader.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame)
    height = len(image)
    width = len(image[0])
    return width, height

# decord.VideoReader
'''
def create_video_reader(video_path):
    return decord.VideoReader(video_path)

def iter_images(video_reader, start_idx=0):
    for frame in range(start_idx, len(video_reader) - start_idx):
        # .asnumpy() # https://github.com/dmlc/decord/issues/208
        image = video_reader[frame].asnumpy()
        yield (frame, image)

def get_width_height(video_reader):
    frame = 0
    image = video_reader[frame].asnumpy()
    height = len(image)
    width = len(image[0])
    return width, height
'''

def iter_hashes(video_reader, start_idx=0):
    for frame, image in iter_images(video_reader, start_idx):
        yield frame, image, hash_image(image)



if len(sys.argv) > 2:
    video_path_1 = sys.argv[1]
    video_path_2 = sys.argv[2]

elif len(sys.argv) > 1 and sys.argv[1] == "--test":
    video_path_1 = "videodiff.py.test.video1.mp4"
    video_path_2 = "videodiff.py.test.video2.mp4"

    # https://stackoverflow.com/questions/30509573
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    video_writer_1 = cv2.VideoWriter(video_path_1, fourcc, 20.0, (600, 400))
    video_writer_2 = cv2.VideoWriter(video_path_2, fourcc, 20.0, (600, 400))

    while(True):
        ret, frame = cap.read()
        out.write(frame)
        cv2.imshow('frame', frame)
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

else:
    print("usage:")
    print("videodiff videoA videoB")
    print("videodiff --test")
    sys.exit(1)

print("videodiff.py", video_path_1, video_path_2)

video_reader_1 = create_video_reader(video_path_1)
video_reader_2 = create_video_reader(video_path_2)

width0, height0 = get_width_height(video_reader_1)
width1, height1 = get_width_height(video_reader_1)

min_width = min(width0, width1)
min_height = min(height0, height1)

#y_axis = []
#frame_number = 0

#hash1_it = iter(iter_hashes(video_reader_1))
#win2_it = iter(window(iter_hashes(video_reader_2)))

# should be odd number so we have a center
# more needs more memory
# win2_size = 9
#win2_size = 45
win2_size = 545

win2_center_idx = win2_size // 2

start_idx = 0
# debug: seek into video to first diff
start_idx = 300 - win2_center_idx
start_idx = 860 - win2_center_idx
start_idx = 43208 - win2_center_idx
start_idx = 44120 - win2_center_idx
start_idx = 44390 - win2_center_idx
start_idx = 45450 - win2_center_idx # todo 45460 no match
start_idx = 67340 - win2_center_idx

# TODO
# video1 914 =ssim= 914 video2 @ offset 0
# video1 915: no match
start_idx = 910 - win2_center_idx

# TODO
# video1 5626 =ssim= 5626 video2 @ offset 0
# video1 5627: no match
# start_idx = 5620 - win2_center_idx



assert start_idx >= 0, f"expected: start_idx >= 0. actual: start_idx = {start_idx}"

hash1_it = iter(iter_hashes(video_reader_1, start_idx))
win2_it = iter(window(iter_hashes(video_reader_2, start_idx), win2_size))

win2 = next(win2_it)

# seek to middle of win2
for _ in range(win2_size // 2):
    next(hash1_it)

offset = 0

'''
# fix PIL.Image.show
# https://stackoverflow.com/questions/77204977
# from PIL import ImageShow
PIL.ImageShow.register(PIL.ImageShow.XDGViewer(), 0)
'''

import subprocess
import io
def show_image(image, label=None, dt=0.1):
    # this requires opencv with gtk
    # cv2.error: OpenCV(4.9.0) /build/source/modules/highgui/src/window.cpp:1272: error: (-2:Unspecified error)
    # The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support.
    '''
    cv2.imshow(f"image1 {frame1}", image1)
    c = cv2.waitKey(0.1)
    if c & 0xFF == ord('q'):
        break
    '''
    # this requires PIL.ImageShow.register
    # this is not efficient, because it writes image to file (tmpfs?)
    #PIL.Image.fromarray(image1).show()
    image = PIL.Image.fromarray(image) # numpy to PIL
    # no. feh still writes the file to disk, using ssd @ /tmp not tmpfs @ /run/user/1000
    '''
    image_bytes = io.BytesIO()
    #image.save(image_bytes, format="tiff") # tiff is faster than png
    image.save(image_bytes, format="png")
    image_bytes = image_bytes.getvalue()
    # TODO set label
    args = ["feh", "-"]
    proc = subprocess.Popen(args, stdin=subprocess.PIPE)
    out, err = proc.communicate(input=image_bytes, timeout=dt)
    print("out, err", out, err)
    time.sleep(dt)
    proc.kill()
    '''
    tempdir = f"/run/user/{os.getuid()}"
    if not os.path.exists(tempdir):
        tempdir = None
    image_path = tempfile.mktemp(suffix='.png', prefix='videodiff.py.frame.', dir=tempdir)
    image.save(image_path, format="png")
    # TODO label
    args = ["feh", image_path]
    proc = subprocess.Popen(args)
    time.sleep(dt)
    proc.kill()
    os.unlink(image_path)


'''
tempdir = f"/run/user/{os.getuid()}"
if not os.path.exists(tempdir):
    tempdir = None
image_path = tempfile.mktemp(suffix='.png', prefix='videodiff.py.frame.', dir=tempdir)
image = np.zeros((500, 500, 3), dtype = "uint8")
#PIL.Image.fromarray(image).save(image_path, format="png")
image_fd = open(image_path, "wb")
PIL.Image.fromarray(image).save(image_fd, format="png")
args = ["feh", image_path]
proc = subprocess.Popen(args)
time.sleep(1) # wait for feh
'''

# TODO use rtmp network stream
# https://stackoverflow.com/questions/69379674
# https://stackoverflow.com/questions/68545688
# https://stackoverflow.com/questions/38686359
# https://stackoverflow.com/questions/69188430
# https://github.com/kkroening/ffmpeg-python/issues/246
def open_ffmpeg_stream_process(width, height):
    args = (
        #"ffmpeg -re -stream_loop -1 -f rawvideo -pix_fmt "
        #"ffplay -re -stream_loop -1 -f rawvideo -pix_fmt "
        #"ffplay -stream_loop -1 -f rawvideo -pix_fmt "
        #"rgb24 -s {width}x{height} -i pipe:0 -pix_fmt yuv420p "
        #"-f rtsp rtsp://rtsp_server:8554/stream"
        #"ffplay -f rawvideo -pix_fmt rgb24 -s {width}x{height} -i -"
        #f"ffplay -f rawvideo -s {width}x{height} -i -"
        #f"ffplay -f rawvideo -i -"
        f"ffplay -f mjpeg -i -"
    ).split()
    return subprocess.Popen(args, stdin=subprocess.PIPE)

video_writer = None

def kill_video_writer():
    global video_writer
    if not video_writer:
        return
    video_writer.stdin.close()
    video_writer.kill()

video1_width, video1_height = None, None

'''
video1 The.Brothers.Grimsby.2016.720p.BluRay.x264-YTS.mp4.cut-51.00-52.00.mkv
video2 vector-spionbruder-h1080p.mkv.cut-50.57-52.00.mkv

video1
51:08.774 - 51:00 = 8.774
>>> 8.774 * (24000/1001)
210.36563436563435

video2
51:08.899 - 50:57 = 11.899
>>> 11.899 * (24000/1001)
285.2907092907093

last sync
Screenshot: 'The.Brothers.Grimsby.2016.720p.BluRay.x264-YTS_shots/shot-00_51_08.774.png'
Screenshot: 'vector-spionbruder-h1080p_shots/shot-00_51_08.899.png'
8.899-8.774 = 0.125

0.917-0.125 = 0.792
0.792 * (24000/1001) = 18.989010989010993 = 19
-> 18 frames extra in vector-spionbruder-h1080p

next sync
Screenshot: 'The.Brothers.Grimsby.2016.720p.BluRay.x264-YTS_shots/shot-00_51_08.816.png' =
Screenshot: 'vector-spionbruder-h1080p_shots/shot-00_51_09.733.png'
9.733-8.816 = 0.917
'''

# maximum hash difference (imagehash)
# TODO verify
#max_hashdiff = 999999
max_hashdiff = 9999

# minimum image similarity (ssim)
# TODO verify. some matches have 0.95, some mismatches have 0.6
# frame1 45460 - imagesim 0.8967622243723549
min_imagesim = 0.8

# FIXME get random seek access to image1 and image2
# when the offset changes in one iteration
# then use the new offset in the next iteration
offset = 0

# diff loop
# expected: 18 frames extra in vector-spionbruder-h1080p = video2
#for step in range(100):
for step in range(999999999):

    frame1, image1, hash1 = next(hash1_it)

    #show_image(image1, f"image1 {frame1}", 1)
    # TODO show video stream with image1
    # https://stackoverflow.com/questions/71302054/how-can-i-play-opencv-frames-in-a-media-player-such-as-mpv

    if not video1_width:
        video1_width, video1_height = len(image1[0]), len(image1)

    '''
    if not video_writer:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video1_width, video1_height = len(image1[0]), len(image1)
        video_writer = cv2.VideoWriter('vide-pipe', fourcc, 25, (video1_width, video1_height));

    video_writer.write(image1)
    '''

    '''
    image = image1
    #image_bytes = io.BytesIO()
    #PIL.Image.fromarray(image).save(image_bytes, format="tiff") # tiff is faster than png
    #PIL.Image.fromarray(image).save(image_bytes, format="png")
    #image_bytes = image_bytes.getvalue()
    #PIL.Image.fromarray(image).save(image_path, format="png")
    image_fd.seek(0)
    PIL.Image.fromarray(image).save(image_fd, format="png")
    '''

    """
    if not video_writer:
        video1_width, video1_height = len(image1[0]), len(image1)
        video_writer = open_ffmpeg_stream_process(video1_width, video1_height)

        atexit.register(kill_video_writer)

        '''
        @atexit.register
        def kill_video_writer():
            video_writer.kill()
        '''

    #video_writer.stdin.write(image1.astype(np.uint8).tobytes()) # raw
    # jpeg
    image_bytes = io.BytesIO()
    PIL.Image.fromarray(image1).save(image_bytes, format="jpeg")
    image_bytes = image_bytes.getvalue()
    video_writer.stdin.write(image_bytes)
    time.sleep(1)
    """

    matches = []

    # search hash1 in win2
    # TODO start from middle of win2
    #for frame2, hash2 in win2:

    # center of win2
    win2_idx = win2_center_idx
    frame2, image2, hash2 = win2[win2_idx]

    '''
    image1_path = f"/run/user/{os.getuid()}/videodiff.py.frame1-{frame1:05d}-image1.jpg"
    PIL.Image.fromarray(image1).resize((video1_width, video1_height)).save(image1_path, format="jpeg")
    '''
    #print(f"writing {image1_path}")

    # center of win2
    # pass 1: compare_hashes
    # cheap
    hashdiff = compare_hashes(hash1, hash2)
    if hashdiff < max_hashdiff:
        offset = frame2 - frame1
        print(f"video1 {frame1} =hash= {frame2} video2 @ offset {offset} @ hashdiff {hashdiff}")
        '''
        image2_path = f"/run/user/{os.getuid()}/videodiff.py.frame1-{frame1:05d}-image2-{frame2:05d}-hashdiff-{hashdiff}.jpg"
        PIL.Image.fromarray(image2).resize((video1_width, video1_height)).save(image2_path, format="jpeg")
        #print(f"writing {image2_path}")
        '''
        win2 = next(win2_it)
        continue

    # center of win2
    # pass 2: compare_images
    # expensive: 10x slower than hash_image
    # plus, hash_image runs 1x per frame
    # but compare_images runs 1x per comparison
    imagesim = compare_images(image1, image2, min_width, min_height)
    #if True: # debug
    if imagesim > min_imagesim:
        offset = frame2 - frame1
        print(f"video1 {frame1} =ssim= {frame2} video2 @ offset {offset} @ imagesim {imagesim}")
        '''
        image2_path = f"/run/user/{os.getuid()}/videodiff.py.frame1-{frame1:05d}-image2-{frame2:05d}-hashdiff-{hashdiff}-imagesim-{imagesim}.jpg"
        PIL.Image.fromarray(image2).resize((video1_width, video1_height)).save(image2_path, format="jpeg")
        #print(f"writing {image2_path}")
        '''
        win2 = next(win2_it)
        continue

    '''
    image2_path = f"/run/user/{os.getuid()}/videodiff.py.frame1-{frame1:05d}-image2-{frame2:05d}-hashdiff-{hashdiff}-imagesim-{imagesim}.jpg"
    PIL.Image.fromarray(image2).resize((video1_width, video1_height)).save(image2_path, format="jpeg")
    #print(f"writing {image2_path}")
    '''

    # TODO find next best match in a limited range, or none
    # TODO first try hash_image, then try compare_images
    #print(f"video1 {frame1}")
    #print(f"  video2 {frame2}: {hashdiff} {imagesim}")
    '''
    if hashdiff != 0:
        image1_path = f"/run/user/{os.getuid()}/videodiff.py.frame1-{frame1:05d}-image1.jpg"
        PIL.Image.fromarray(image1).resize((video1_width, video1_height)).save(image1_path, format="jpeg")
        image2_path = f"/run/user/{os.getuid()}/videodiff.py.frame1-{frame1:05d}-image2-{frame2:05d}-diff-{hashdiff}.jpg"
        PIL.Image.fromarray(image2).resize((video1_width, video1_height)).save(image2_path, format="jpeg")
        print("todo:")
        print(f"  feh {image1_path} {image2_path}")
    #if hashdiff < max_hashdiff:
    if True:
        matches.append((hashdiff, frame2))
    '''

    # move from center to start/end of win2
    # pass 1: compare_hashes
    found_match = False
    for win2_idx_left in range((win2_center_idx - 1), -1, -1):
        if found_match:
            break
        win2_idx_right = win2_size - win2_idx_left - 1
        for win2_idx in (win2_idx_left, win2_idx_right):
            frame2, image2, hash2 = win2[win2_idx]
            hashdiff = compare_hashes(hash1, hash2)
            #imagesim = compare_images(image1, image2, min_width, min_height)
            #print(f"  frame2 {frame2}: {hashdiff}")
            #if hashdiff < max_hashdiff:
            #if True:
            #    matches.append((hashdiff, frame2))
            if hashdiff < max_hashdiff:
                offset = frame2 - frame1
                print(f"video1 {frame1} =hash= {frame2} video2 @ offset {offset} @ hashdiff {hashdiff}")
                '''
                image2_path = f"/run/user/{os.getuid()}/videodiff.py.frame1-{frame1:05d}-image2-{frame2:05d}-hashdiff-{hashdiff}.jpg"
                PIL.Image.fromarray(image2).resize((video1_width, video1_height)).save(image2_path, format="jpeg")
                #print(f"writing {image2_path}")
                '''
                win2 = next(win2_it)
                found_match = True
                break

    # move from center to start/end of win2
    # pass 2: compare_images
    for win2_idx_left in range((win2_center_idx - 1), -1, -1):
        if found_match:
            break
        win2_idx_right = win2_size - win2_idx_left - 1
        for win2_idx in (win2_idx_left, win2_idx_right):
            frame2, image2, hash2 = win2[win2_idx]
            #hashdiff = compare_hashes(hash1, hash2)
            imagesim = compare_images(image1, image2, min_width, min_height)
            #print(f"  frame2 {frame2}: {imagesim}")
            #if hashdiff < max_hashdiff:
            #if True:
            #    matches.append((hashdiff, frame2))
            if imagesim > min_imagesim:
                offset = frame2 - frame1
                print(f"video1 {frame1} =ssim= {frame2} video2 @ offset {offset} @ imagesim {imagesim}")
                '''
                image2_path = f"/run/user/{os.getuid()}/videodiff.py.frame1-{frame1:05d}-image2-{frame2:05d}-hashdiff-{hashdiff}-imagesim-{imagesim}.jpg"
                PIL.Image.fromarray(image2).resize((video1_width, video1_height)).save(image2_path, format="jpeg")
                #print(f"writing {image2_path}")
                '''
                win2 = next(win2_it)
                found_match = True
                break

    if not found_match:
        print(f"video1 {frame1}: no match")
        image1_path = f"/run/user/{os.getuid()}/videodiff.py.frame1-{frame1:05d}-no-match-image1.jpg"
        print(f"writing {image1_path}")
        PIL.Image.fromarray(image1).resize((video1_width, video1_height)).save(image1_path, format="jpeg")
        # center of win2
        win2_idx = win2_center_idx
        frame2, image2, hash2 = win2[win2_idx]
        hashdiff = compare_hashes(hash1, hash2)
        imagesim = compare_images(image1, image2, min_width, min_height)
        image2_path = f"/run/user/{os.getuid()}/videodiff.py.frame1-{frame1:05d}-no-match-image2-{frame2:05d}-hashdiff-{hashdiff}-imagesim-{imagesim}.jpg"
        print(f"writing {image2_path}")
        PIL.Image.fromarray(image2).resize((video1_width, video1_height)).save(image2_path, format="jpeg")

    """
    if matches:
        #matches.sort(key=lambda x: x[0])
        for hashdiff, frame2 in matches:
            print(f"  frame2 {frame2}: {hashdiff}")
    """

    win2 = next(win2_it)

kill_video_writer()

'''
import difflib

print("difflib.SequenceMatcher ...")

# FIXME this blocks

s = difflib.SequenceMatcher(
    None,
    iter_hashes(video_reader_1),
    iter_hashes(video_reader_2),
)

print("difflib.SequenceMatcher ok")

print("s.get_matching_blocks ...")

for block in s.get_matching_blocks():
    print("a[%d] and b[%d] match for %d elements" % block)
'''



sys.exit()


###########

t1 = time.time()

# read the first image to get video resolutions
image1, image2 = read_images(video_reader_1, video_reader_2)

height1, width1 = len(image1), len(image1[0])
height2, width2 = len(image2), len(image2[0])

print(f"video1 {width1}x{height1}")
print(f"video2 {width2}x{height2}")

if height1 == height2 and width1 == width2:
    # same size
    resize1 = lambda img: img
    resize2 = lambda img: img
elif width1 < width2:
    # image 2 is larger
    resize1 = lambda img: img
    resize2 = lambda img: cv2.resize(img, (width1, height1), interpolation=cv2.INTER_LINEAR)
else:
    # image 1 is larger
    resize1 = lambda img: cv2.resize(img, (width2, height2), interpolation=cv2.INTER_LINEAR)
    resize2 = lambda img: img

#while True:
#for frame_idx in range(10):
for frame_idx in range(99999999):

    hash1 = hash_image(image1)
    hash2 = hash_image(image2)

    '''
    image1 = resize1(image1)
    image2 = resize2(image2)

    score = process_img(image1, image2)
    frame_number += 1
    y_axis.append(score)
    print(f"frame {frame_idx}: {score}")
    '''

    # TODO ignore black frames by hash

    hashdiff = compare_hashes(hash1, hash2)

    print(f"frame {frame_idx}: {hash1} - {hash2} = {hashdiff}")

    # read next image
    image1, image2 = read_images(video_reader_1, video_reader_2)

t2 = time.time()
print("dt", t2 - t1)

sys.exit()

x_axis = list(range(0, frame_number))
plt.title("differences between videos")
plt.xlabel("Frames")
plt.ylabel("% of difference")
plt.scatter(x_axis, y_axis, s=10, c='red')
plt.show()
