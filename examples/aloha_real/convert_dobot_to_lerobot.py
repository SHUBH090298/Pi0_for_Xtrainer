"""  
Script to convert Dobot X-Trainer hdf5 data to the LeRobot dataset v2.0 format.  
  
Example usage: python convert_dobot_to_lerobot.py --raw-dir /path/to/train_data --repo-id <org>/<dataset-name>  
"""  
  
import dataclasses  
from pathlib import Path  
import shutil  
from typing import Literal  
  
import h5py  
#from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME  
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  
from lerobot.common.constants import HF_LEROBOT_HOME as LEROBOT_HOME
#from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw  
import numpy as np  
import torch  
import tqdm  
import tyro  
import cv2  
  
  
@dataclasses.dataclass(frozen=True)  
class DatasetConfig:  
    use_videos: bool = True  
    tolerance_s: float = 0.0001  
    image_writer_processes: int = 10  
    image_writer_threads: int = 5  
    video_backend: str | None = None  
  
  
DEFAULT_DATASET_CONFIG = DatasetConfig()  
  
  
def create_empty_dataset(  
    repo_id: str,  
    robot_type: str,  
    mode: Literal["video", "image"] = "video",  
    *,  
    has_velocity: bool = False,  
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,  
) -> LeRobotDataset:  
    # 14 motors: 7 per arm (6 joints + 1 gripper)  
    motors = [  
        "left_waist",  
        "left_shoulder",  
        "left_elbow",  
        "left_forearm_roll",  
        "left_wrist_angle",  
        "left_wrist_rotate",  
        "left_gripper",  
        "right_waist",  
        "right_shoulder",  
        "right_elbow",  
        "right_forearm_roll",  
        "right_wrist_angle",  
        "right_wrist_rotate",  
        "right_gripper",  
    ]  
      
    # Camera names from your setup  
    cameras = [  
        "top",  
        "left_wrist",  
        "right_wrist",  
    ]  
  
    features = {  
        "observation.state": {  
            "dtype": "float32",  
            "shape": (len(motors),),  
            "names": [motors],  
        },  
        "action": {  
            "dtype": "float32",  
            "shape": (len(motors),),  
            "names": [motors],  
        },  
    }  
  
    if has_velocity:  
        features["observation.velocity"] = {  
            "dtype": "float32",  
            "shape": (len(motors),),  
            "names": [motors],  
        }  
  
    for cam in cameras:  
        features[f"observation.images.{cam}"] = {  
            "dtype": mode,  
            "shape": (3, 480, 640),  
            "names": ["channels", "height", "width"],  
        }  
  
    if Path(LEROBOT_HOME / repo_id).exists():  
        shutil.rmtree(LEROBOT_HOME / repo_id)  
  
    return LeRobotDataset.create(  
        repo_id=repo_id,  
        fps=50,  
        robot_type=robot_type,  
        features=features,  
        use_videos=dataset_config.use_videos,  
        tolerance_s=dataset_config.tolerance_s,  
        image_writer_processes=dataset_config.image_writer_processes,  
        image_writer_threads=dataset_config.image_writer_threads,  
        video_backend=dataset_config.video_backend,  
    )  
  
  
def get_cameras(hdf5_files: list[Path]) -> list[str]:  
    with h5py.File(hdf5_files[0], "r") as ep:  
        return list(ep["/observations/images"].keys())  
  
  
def has_velocity(hdf5_files: list[Path]) -> bool:  
    with h5py.File(hdf5_files[0], "r") as ep:  
        return "/observations/qvel" in ep  
  
  
def load_raw_images_per_camera(ep: h5py.File, cameras: list[str], is_compressed: bool) -> dict[str, np.ndarray]:  
    imgs_per_cam = {}  
    for camera in cameras:  
        if is_compressed:  
            # Decompress JPEG images  
            imgs_array = []  
            for data in ep[f"/observations/images/{camera}"]:  
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)  
                # Convert BGR to RGB  
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
                imgs_array.append(img)  
            imgs_array = np.array(imgs_array)  
        else:  
            # Load uncompressed images directly  
            imgs_array = ep[f"/observations/images/{camera}"][:]  
            # Ensure RGB format  
            if imgs_array.shape[-1] == 3:  
                imgs_array = imgs_array[..., ::-1]  # BGR to RGB if needed  
  
        imgs_per_cam[camera] = imgs_array  
    return imgs_per_cam  
  
  
def load_raw_episode_data(  
    ep_path: Path,  
    cameras: list[str],  
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None]:  
    with h5py.File(ep_path, "r") as ep:  
        # Check if images are compressed  
        is_compressed = ep.attrs.get("compress", False)  
          
        state = torch.from_numpy(ep["/observations/qpos"][:].astype(np.float32))  
        action = torch.from_numpy(ep["/action"][:].astype(np.float32))  
  
        velocity = None  
        if "/observations/qvel" in ep:  
            velocity = torch.from_numpy(ep["/observations/qvel"][:].astype(np.float32))  
  
        imgs_per_cam = load_raw_images_per_camera(ep, cameras, is_compressed)  
  
    return imgs_per_cam, state, action, velocity  
  
  
def populate_dataset(  
    dataset: LeRobotDataset,  
    hdf5_files: list[Path],  
    cameras: list[str],  
    task: str,  
    episodes: list[int] | None = None,  
) -> LeRobotDataset:  
    if episodes is None:  
        episodes = range(len(hdf5_files))  
  
    for ep_idx in tqdm.tqdm(episodes):  
        ep_path = hdf5_files[ep_idx]  
  
        imgs_per_cam, state, action, velocity = load_raw_episode_data(ep_path, cameras)  
        num_frames = state.shape[0]  
  
        for i in range(num_frames):  
            frame = {  
                "observation.state": state[i],  
                "action": action[i], 
                "task" : task, 
            }  
  
            for camera, img_array in imgs_per_cam.items():  
                frame[f"observation.images.{camera}"] = img_array[i]  
  
            if velocity is not None:  
                frame["observation.velocity"] = velocity[i]  
  
            dataset.add_frame(frame)  
  
        dataset.save_episode()  
  
    return dataset  
  
  
def port_dobot(  
    raw_dir: Path,  
    repo_id: str,  
    raw_repo_id: str | None = None,  
    task: str = "dobot_Pick_Place",  
    *,  
    episodes: list[int] | None = None,  
    push_to_hub: bool = False,  
    robot_type: str = "dobot_xtrainer",  
    mode: Literal["video", "image"] = "image",  
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,  
):  
    if (LEROBOT_HOME / repo_id).exists():  
        shutil.rmtree(LEROBOT_HOME / repo_id)  
  
    if not raw_dir.exists():  
        if raw_repo_id is None:  
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")  
        download_raw(raw_dir, repo_id=raw_repo_id)  
  
    hdf5_files = sorted(raw_dir.glob("episode_init_*.hdf5"))  
      
    if not hdf5_files:  
        raise ValueError(f"No episode_init_*.hdf5 files found in {raw_dir}")  
  
    cameras = get_cameras(hdf5_files)  
    print(f"Found cameras: {cameras}")  
  
    dataset = create_empty_dataset(  
        repo_id,  
        robot_type=robot_type,  
        mode=mode,  
        has_velocity=has_velocity(hdf5_files),  
        dataset_config=dataset_config,  
    )  
      
    dataset = populate_dataset(  
        dataset,  
        hdf5_files,  
        cameras,  
        task=task,  
        episodes=episodes,  
    )  
      
    #dataset.consolidate()  
  
    if push_to_hub:  
        dataset.push_to_hub()  
  
  
if __name__ == "__main__":  
    tyro.cli(port_dobot)