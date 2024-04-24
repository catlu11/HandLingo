import torch
import numpy as np
from I3D.transform_video import get_video_tensor
from I3D.pytorch_i3d import InceptionI3d

NUM_CLASSES = 100
WEIGHTS_PATH = 'I3D/FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt'
CLASS_LIST = 'wlasl_class_list.txt'

class Sign2TextModel:
    
    def __init__(self):
        # Init model
        i3d = InceptionI3d(400, in_channels=3)
        i3d.replace_logits(NUM_CLASSES)
        i3d.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
        self.model = i3d
        
        # Load labels
        class_list = open(CLASS_LIST, "r")
        lines = class_list.readlines()
        self.labels = [l.split('\t')[-1] for l in lines]
        class_list.close()

    def get_prediction(self, video_path):
        inputs = get_video_tensor(video_path)
        per_frame_logits = self.model(inputs)
        predictions = torch.max(per_frame_logits, dim=2)[0]
        out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
        return self.labels[out_labels[-1]]
