#!/usr/bin/env bash

# TODO move to ffmpeg_align_video_scenes.py

fs=()
fs+=(Mike.Judges.Beavis.and.Butt-Head.S01E01.S01E02.German.DL.720p.WEB.x264-WvF.mkv)
fs+=(Mike.Judges.Beavis.and.Butt-Head.S01E01.720p.AMZN.WEBRip.x264-GalaxyTV.mkv)

for f in "${fs[@]}"; do

# limit time range
start=0
#duration=10
duration=30

# Media type mismatch between the 'Parsed_amovie_0' filter output pad 0 (audio) and the 'Parsed_trim_1' filter input pad 0 (video)
#a1=":seek_point=$start,trim=$start:duration=$duration"
# fix: trim -> atrim
a1=":seek_point=$start,atrim=$start:duration=$duration"

v1=":seek_point=$start,trim=$start:duration=$duration"
v1=

set -x

# silencedetect returns wrong positions?!
if false; then
# Set silence duration until notification (default is 2 seconds). See (ffmpeg-utils)the Time duration section in the ffmpeg-utils(1) manual for the accepted syntax.
silence_duration=0.1
# Set noise tolerance. Can be specified in dB (in case "dB" is appended to the specified value) or amplitude ratio. Default is -60dB, or 0.001.
silence_noise=-60dB
silencedetect="silencedetect=noise=$silence_noise:duration=$silence_duration"
#silencedetect="silencedetect" # no result. default d=2
# find silent audio
a=(
  ffprobe -f lavfi -i "amovie=$f${a1},$silencedetect" -show_entries tags=lavfi.silence_start,lavfi.silence_end -of default=nw=1
  #-v quiet
)
"${a[@]}" | tee "$f".quiet.txt
echo done "$f".quiet.txt
fi

# find black video
ffprobe -f lavfi -i "movie=$f${v1},blackdetect[out0]" -show_entries tags=lavfi.black_start,lavfi.black_end -of default=nw=1 -v quiet | tee "$f".black.txt
echo done "$f".black.txt

done
