cd ~/Documents/csz-project/multiAgentChasing/data/test

ffmpeg -r  15 -f image2 -s 1920x1080 -i  %04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ~/Documents/csz-project/multiAgentChasing/demo/demo3.mp4

cd ~/Documents/csz-project/multiAgentChasing/src
