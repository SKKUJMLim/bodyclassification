import os 
import cv2 
import numpy as np 
from pyk4a import PyK4APlayback  

DATASET_PATH = 'D:/data/Data/' 
NAME_OF_MASTER_VIDEO = 'M.mkv'

def make_directory(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def decode_mjpg_image(mjpg_image):

    buffer_array = np.ctypeslib.as_array(mjpg_image, shape=(2048, 1536, 3)) 
    decoded_image = cv2.imdecode(np.frombuffer(buffer_array, dtype=np.uint8).copy(), -1)

    return decoded_image    

def colorize_depth_image(depth_image):
    depth_color_image = cv2.convertScaleAbs(depth_image, alpha=0.05)
    depth_color_image = cv2.applyColorMap(depth_color_image, cv2.COLORMAP_JET)
    depth_color_image = cv2.cvtColor(depth_color_image, cv2.COLOR_RGBA2RGB)

    return depth_color_image

def list_of_videos():
    list_of_video_path = [] 
    
    for file in os.listdir(DATASET_PATH):
        list_of_video_path.append(os.path.join(DATASET_PATH, file, NAME_OF_MASTER_VIDEO))
    
    return list_of_video_path 

def save_frames_from_video(dataset_path, playback: PyK4APlayback): 
    make_directory(dataset_path + 'rgb_images')
    make_directory(dataset_path + 'depth_images')

    
    color_image_count = 0
    depth_image_count = 0
    while True:
        try:
            capture = playback.get_next_capture() 
 
            if capture.color is not None:
                cv2.imwrite(dataset_path + '/rgb_images/%03d.jpg' % color_image_count, decode_mjpg_image(capture.color))
                color_image_count += 1 
            
            if capture.depth is not None:
                cv2.imwrite(dataset_path + '/depth_images/%03d.jpg' % depth_image_count, colorize_depth_image(capture.depth)) 
                depth_image_count += 1
        
        except EOFError:
            break 
        
def start_playback(video_path):
    playback = PyK4APlayback(video_path)
    playback.open()
    save_frames_from_video(video_path[:-5], playback)
    playback.close()
    
        

if __name__ == "__main__":
    video_path = list_of_videos()
    for path in video_path:
        start_playback(path)