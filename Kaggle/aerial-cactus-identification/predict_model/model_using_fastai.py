import pandas as pd
import numpy as np

from pathlib import Path
from fastai import *
from fastai.vision import *
import torch


print(os.listdir("../input/"))

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv("../input/sample_submission.csv")
data_folder = Path("../input")

print(train_df.shape)

print(train_df.head())

test_img = ImageList.from_df(test_df, path=data_folder, folder='test')
trfm = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)
train_img = (ImageList.from_df(train_df, path=data_folder, folder='train')
             .split_by_rand_pct(0.01)
             .label_from_df()
             .add_test(test_img)
             .transform(trfm, size=128)
             .databunch(path='.', bs=64, device= torch.device('cuda:0'))
             .normalize(imagenet_stats))

learn = cnn_learner(train_img, models.densenet161, metrics=[error_rate, accuracy])