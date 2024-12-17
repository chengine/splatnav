#%%
from moviepy.editor import VideoFileClip
import glob

for filepath in glob.glob("static/videos/*.mp4"):
    videoclip = VideoFileClip(filepath)
    new_clip = videoclip.without_audio()
    new_clip.write_videofile(filepath.replace(".mp4", "_noaudio.mp4"), codec="libx265")
# %%
