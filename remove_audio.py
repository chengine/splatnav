#%%
from moviepy.editor import VideoFileClip
import glob

for filepath in glob.glob("static/videos/*.MP4"):
    videoclip = VideoFileClip(filepath)
    new_clip = videoclip.without_audio()
    new_clip.write_videofile(filepath.replace(".MP4", "_noaudio.MP4"), codec="libx264")
# %%
