import os
import cv2
import argparse
from glob import glob
from files import file_choices


def progress2video(path: str, name: str, fps: int, remove: bool):
    """Generate video from model exported epoch snapshot.

    Parameters
    ----------
    path : str
        Where the snapshots in.
    name : str
        Output video file name, must have .mp4.
    fps : int
        Frame per second.
    remove : bool
        Whether to remove the snapshots in the `path` folder.
    """
    img_array = []
    size = None
    filelist = sorted(glob(os.path.join(path, "*.png")))
    for filename in filelist:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])

    if remove:
        for file in filelist:
            os.remove(file)
    
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Where the snapshots in.")        
    parser.add_argument('--out', type=lambda s:file_choices(("mp4"),s), default="progress.mp4", help="Output video file name, must have .mp4 or .avi.")
    parser.add_argument('--fps', type=int, default=2, help="Frame per second.")
    parser.add_argument('--remove', action="store_true", help="Whether to remove the snapshots in the `path` folder.")
    args = parser.parse_args()
    progress2video(args.path, args.out, args.fps, args.remove)