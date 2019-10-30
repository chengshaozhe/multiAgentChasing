for condition in test
do
    cd ~/Documents/multiAgentChasing/data/${condition}

    ffmpeg -r  30 -f image2 -s 1920x1080 -i  %04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ~/Documents/multiAgentChasing/demo/${condition}.mp4

    cd ~/Documents/multiAgentChasing/exec
done