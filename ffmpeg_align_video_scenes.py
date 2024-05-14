#!/usr/bin/env python3

video_file_list = [
  "Mike.Judges.Beavis.and.Butt-Head.S01E01.S01E02.German.DL.720p.WEB.x264-WvF.mkv",
  "Mike.Judges.Beavis.and.Butt-Head.S01E01.720p.AMZN.WEBRip.x264-GalaxyTV.mkv",
]

# note: tags can be repeated (dont know why)
"""
line 'TAG:lavfi.black_start=0\n'
line 'TAG:lavfi.black_start=0\n'
line 'TAG:lavfi.black_end=1.293\n'
line 'TAG:lavfi.black_end=1.293\n'
"""

def parse_black_range_list(f_black):
  black_range_list = []
  with open(f_black) as f:
    black_start = None
    black_end = None
    for line in f.readlines():
      #line = line.rstrip()
      #print("line", repr(line))
      if black_start == None and line.startswith("TAG:lavfi.black_start="):
        # TAG:lavfi.black_start=1.234
        #                       ^^^^^
        black_start = float(line[22:])
        continue
      if black_start != None and black_end == None and line.startswith("TAG:lavfi.black_end="):
        # TAG:lavfi.black_end=1.234
        #                     ^^^^^
        black_end = float(line[20:])
        black_range_list.append((black_start, black_end))
        black_start = black_end = None
        continue
  return black_range_list

parsed_shit_list = []

black_range_list_list = []

for video_file in video_file_list:
  f = video_file
  f_quiet = f + ".quiet.txt"
  black_range_list = parse_black_range_list(video_file + ".black.txt")
  black_range_list_list.append(black_range_list)

for file_idx, black_range_list in enumerate(black_range_list_list):
  f = video_file_list[file_idx]
  print(f"black_range_list for input {f}")
  for start, end in black_range_list:
    diff = end - start
    print(f"  black_range: {start:10.3f} -> {end:10.3f} = {diff:10.3f}")

joined_black_range_list = []
for file_idx, black_range_list in enumerate(black_range_list_list):
  for range_idx, (start, end) in enumerate(black_range_list):
    joined_black_range_list.append((file_idx, range_idx, start, end))
joined_black_range_list.sort(key=lambda x: x[1]) # sort by start
print("joined_black_range_list")
for file_idx, range_idx, start, end in joined_black_range_list:
  diff = end - start
  print(f"  {file_idx} {range_idx:3d}: {start:10.3f} -> {end:10.3f} = {diff:10.3f}")

# TODO align sequence of diff = end - start
