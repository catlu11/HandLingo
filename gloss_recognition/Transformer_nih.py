import pandas as pd
import torch
from torch.utils.data import Dataset
from .Transformer.extract_features import get_landmarks, extract_keyframes
from .Transformer.transformer import Transformer

def read_feature_file(file):
    df = pd.read_csv(file)
    max_frame = max(df['frame_number'])
    frames = []
    for i in range(max_frame):
        df_coord = df[df['frame_number'] == i].drop(['frame_number', 'landmark'], axis=1)
        if len(df_coord) == 0:
            continue
        frames.append(sum(df_coord.values.tolist(), []))
    return frames

def transform(features):
    result = torch.tensor(features)
    return torch.stack([result], dim=0)

class Sign2TextModel:
    
    def __init__(self):
        model = Transformer()
        model.load_state_dict(torch.load("gloss_recognition/Transformer/best_epoch.pt")['model_state_dict'])
        self.model = model
        self.model.eval()

        self.class_query = torch.stack([torch.zeros(1, 118)], dim=0)

        class_list = open("gloss_recognition/wlasl_class_list.txt", "r")
        lines = class_list.readlines()
        self.labels = [l.split('\t')[-1] for l in lines]
        class_list.close()

    def get_prediction(self, video_path):
        keyframes = extract_keyframes(video_path)
        get_landmarks(video_path, "gloss_recognition/Transformer/temp.csv", keyframes)
        input = transform(read_feature_file("gloss_recognition/Transformer/temp.csv"))
        pred = self.model(input, self.class_query)
        class_id = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
        return self.labels[class_id]