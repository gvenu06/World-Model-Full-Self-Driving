from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.envs.metadrive_env import MetaDriveEnv
import numpy as np
import cv2
from metadrive.engine.interface import Interface
from panda3d.core import WindowProperties


Interface._render_contact_result = lambda self, contacts: None  # remove surface label for visujal appeal type

show_cam1 = False
refocus_in = 0 #how many steps it will refocus in so its not instant and works

def toggle_cam1():
    global show_cam1, refocus_in
    show_cam1 = not show_cam1 #var for showing cam 1(front) is on
    if show_cam1:
        refocus_steps = 3
    else:
        cv2.destroyWindow("front camera")


def refocus(): #fighting macos to give main window to the sim
    props = WindowProperties()
    props.setForeground(True)
    env.engine.win.requestProperties(props)
    print("refocus")



env = MetaDriveEnv(dict(
    image_observation = True, #cars observation is an image
    sensors = dict(rgb_camera=(RGBCamera, 64, 64)), #using only RGB camera
    vehicle_config = dict(
        image_source="rgb_camera",
        show_navi_mark=False,
        show_dest_mark=False,
        show_line_to_dest=False,
        show_line_to_navi_mark=False,),
    use_render = True, #window pops up
    manual_control = True,
    out_of_road_done=False,
    on_continuous_line_done=False,
    show_logo = False,
    window_size=(1360, 883),
))


obs, info = env.reset() #starts sim and puts car at start of road, returns what the car observes and info
env.engine.accept("1", toggle_cam1)


while True:
    #env.step(steering, throttle)
    obs, r, term, trunc, info = env.step([0, 0])
    #obs - what car sees now
    #r-score for that step
    #term - did the episode end
    #trunc - did it end from time limit
    #info - extra deets

    frame = obs["image"][:, :, :, 0] #makes normal pic by removing 4th axis
    frame = (frame * 255).astype(np.uint8) #takes values from (0,1) to (0,255)

    resized_frame = cv2.resize(frame, (384, 384), interpolation=cv2.INTER_NEAREST) #resizes from 64x64 to 384x384 and uses interpolation to expand
   
    top_left_grass_det = (122, 275) #used to make square for grass detection
    bottom_right_grass_det = (262, 384)

    front_grass_det = resized_frame[top_left_grass_det[1]:, top_left_grass_det[0]:bottom_right_grass_det[0]] #bottom middle would be grass if on grass(used for color avg)

    avg_color = front_grass_det.mean(axis=(0,1))
    #print(avg_color)
    

    cv2.rectangle(resized_frame, top_left_grass_det, bottom_right_grass_det, (0,0,255), 3) #BGR
    #cv2.imshow("front camera", resized_frame) #show front camera with title
    cv2.waitKey(1)          # lets the window actually update
    if show_cam1:
        cv2.imshow("front camera", resized_frame)
        if refocus_in > 0: #basically counts down from 3 steps so that way the camera window can acc appear for it to work
            refocus_in -= 1
        if refocus_in == 0:
            refocus()



    

    if term or trunc: #if episode ends, restart
        obs, info = env.reset()



