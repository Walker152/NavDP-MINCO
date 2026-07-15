import requests
import numpy as np
import cv2
import io
import json
import time

REQUEST_TIMEOUT_S = 5.0

def navigator_reset(intrinsic=None,stop_threshold=-0.5,batch_size=1,port=8888,env_id=None,seed=None):
    print("http://localhost:%d/navigator_reset"%port)
    if env_id is None:
        url = "http://localhost:%d/navigator_reset"%port
        response = requests.post(url,json={'intrinsic':intrinsic.tolist(),
                                           'stop_threshold':stop_threshold,
                                           'batch_size':batch_size,
                                           'seed':seed},
                                 timeout=REQUEST_TIMEOUT_S)
    else:
        url = "http://localhost:%d/navigator_reset_env"%port
        response = requests.post(url,json={'env_id':env_id}, timeout=REQUEST_TIMEOUT_S)
    return json.loads(response.text)['algo']

def navigator_close(port=8888, intrinsic=None, stop_threshold=-0.5, batch_size=1):
    url = "http://localhost:%d/navigator_close"%port
    try:
        response = requests.post(url, json={}, timeout=REQUEST_TIMEOUT_S)
        data = json.loads(response.text)
        return bool(data.get("ok", False))
    except Exception:
        if intrinsic is None:
            raise
        # Backward-compatible fallback for an older NavDP server without /navigator_close:
        # reset closes the previous writer before opening the next one, finalizing the run video.
        navigator_reset(intrinsic=intrinsic, stop_threshold=stop_threshold, batch_size=batch_size, port=port)
        return True

def nogoal_step(rgb_images,depth_images,port=8888):
    concat_images = np.concatenate([img for img in rgb_images],axis=0)
    concat_depths = np.concatenate([img for img in depth_images],axis=0)
    url = "http://localhost:%d/nogoal_step"%port
    _, rgb_image = cv2.imencode('.jpg', concat_images)
    image_bytes = io.BytesIO()
    image_bytes.write(rgb_image)
    
    depth_image = np.clip(concat_depths*10000.0,0,65535.0).astype(np.uint16)
    _, depth_image = cv2.imencode('.png', depth_image)
    depth_bytes = io.BytesIO()
    depth_bytes.write(depth_image)
    
    files = {
        'image': ('image.jpg', image_bytes.getvalue(), 'image/jpeg'),
        'depth': ('depth.png', depth_bytes.getvalue(), 'image/png'),
    }
    data = {
        'depth_time':time.time(),
        'rgb_time':time.time(),
    }
    response = requests.post(url, files=files, data=data, timeout=REQUEST_TIMEOUT_S)
    trajectory = json.loads(response.text)['trajectory']
    all_trajectory = json.loads(response.text)['all_trajectory']
    all_value = json.loads(response.text)['all_values']
    return np.array(trajectory),np.array(all_trajectory),np.array(all_value)

def pointgoal_step(point_goals,rgb_images,depth_images,port=8888):
    concat_images = np.concatenate([img for img in rgb_images],axis=0)
    concat_depths = np.concatenate([img for img in depth_images],axis=0)
    url = "http://localhost:%d/pointgoal_step"%port
    _, rgb_image = cv2.imencode('.jpg', concat_images)
    image_bytes = io.BytesIO()
    image_bytes.write(rgb_image)
    
    depth_image = np.clip(concat_depths*10000.0,0,65535.0).astype(np.uint16)
    _, depth_image = cv2.imencode('.png', depth_image)
    depth_bytes = io.BytesIO()
    depth_bytes.write(depth_image)
    
    files = {
        'image': ('image.jpg', image_bytes.getvalue(), 'image/jpeg'),
        'depth': ('depth.png', depth_bytes.getvalue(), 'image/png'),
    }
    data = {
        'goal_data': json.dumps({
        'goal_x': point_goals[:,0].tolist(),
        'goal_y': point_goals[:,1].tolist()
        }),
        'depth_time':time.time(),
        'rgb_time':time.time(),
    }
    response = requests.post(url, files=files, data=data, timeout=REQUEST_TIMEOUT_S)
    trajectory = json.loads(response.text)['trajectory']
    all_trajectory = json.loads(response.text)['all_trajectory']
    all_value = json.loads(response.text)['all_values']
    if 'sub_pointgoal_pd' in json.loads(response.text):
        sub_pointgoal_pd = json.loads(response.text)['sub_pointgoal_pd']
        return np.array(trajectory),np.array(all_trajectory),np.array(all_value),sub_pointgoal_pd
    else:
        return np.array(trajectory),np.array(all_trajectory),np.array(all_value)

def imagegoal_step(image_goals,rgb_images,depth_images,port=8888):
    concat_images = np.concatenate([img for img in rgb_images],axis=0)
    concat_depths = np.concatenate([img for img in depth_images],axis=0)
    concat_goals = np.concatenate([img for img in image_goals],axis=0)
    
    url = "http://localhost:%d/imagegoal_step"%port
    _, rgb_image = cv2.imencode('.jpg', concat_images)
    image_bytes = io.BytesIO()
    image_bytes.write(rgb_image)
    
    _, goal_image = cv2.imencode('.jpg', concat_goals)
    goal_bytes = io.BytesIO()
    goal_bytes.write(goal_image)
    
    depth_image = np.clip(concat_depths*10000.0,0,65535.0).astype(np.uint16)
    _, depth_image = cv2.imencode('.png', depth_image)
    depth_bytes = io.BytesIO()
    depth_bytes.write(depth_image)
    
    files = {
        'image': ('image.jpg', image_bytes.getvalue(), 'image/jpeg'),
        'goal': ('goal.jpg', goal_bytes.getvalue(), 'image/jpeg'),
        'depth': ('depth.png', depth_bytes.getvalue(), 'image/png'),
    }
    data = {
        'depth_time':time.time(),
        'rgb_time':time.time(),
    }
    response = requests.post(url, files=files, data=data, timeout=REQUEST_TIMEOUT_S)
    trajectory = json.loads(response.text)['trajectory']
    all_trajectory = json.loads(response.text)['all_trajectory']
    all_value = json.loads(response.text)['all_values']
    return np.array(trajectory),np.array(all_trajectory),np.array(all_value)

