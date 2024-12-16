#%%
from moviepy.editor import VideoFileClip

videoclip = VideoFileClip("static/videos/trial2_clip.mov")
new_clip = videoclip.without_audio()
new_clip.write_videofile("static/videos/trial2_clip_.mov", codec="libx264")
# %%
