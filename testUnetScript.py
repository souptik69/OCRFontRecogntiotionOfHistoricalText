from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image
from pytorch_lightning import seed_everything
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset.unet.UnetDataModule import UnetDataModule
from model.unet.UnetModel import UnetModel
from utils.CustomCollate import collate_unet
from utils.metrics.LineDetectionMetricUtils import compute_line_detection_from_batch
from utils.constants import *

print("Cuda available?", torch.cuda.is_available())

RANDOM_SEED = 47

torch.backends.cudnn.deterministic = True
seed_everything(RANDOM_SEED)

def run(args):
    if args.books is not None:
        books = args.books.split(",")

    dataModule = UnetDataModule(root_dir=args.root_dir, batch_size=args.batch_size, multiple_books=books, return_coords=True)
    dataModule.prepare_data()
    dataModule.setup()

    # dataset = UnetDataset(root_dir=args.root_dir, multiple_books=books, return_coords=True)
    # dl = DataLoader(dataModule.test_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False,
    #                 collate_fn=collate_unet)

    model = UnetModel.load_from_checkpoint(args.unet_checkpoint, strict=False)
    model.eval()
    model = model.cuda()
    toPIL = transforms.ToPILImage()
    sigmoid = torch.nn.Sigmoid()

    thresholds_to_test = np.arange(0.,1.01, 0.1)
    thresholds_to_test = [0.5]

    ROC = {}
    for t in tqdm(thresholds_to_test):
        predictions = []
        targets = []
        accuracies = []
        for idx, batch in enumerate(tqdm(dataModule.test_dataloader())):
            X, Y, height, width, content = batch[IMAGE], batch[MASK], batch["original_height"], batch["original_width"], \
                                           batch["content_infos"]

            x = toPIL(X[0])
            # mask = toPIL(Y[0]).convert("RGBA")
            # # y = toPIL(Y[0])
            # x = trans_paste(mask, x, .7)
            # x.show()
            # y.show()
            pred = model(X.cuda()).cpu()
            p = (sigmoid(pred[0])>0.5).float()
            p = toPIL(p).convert("RGBA")
            p = trans_paste(p, x, 0.6)
            # p.show()
            # p.save("tmp/att-unet-balanced.png")

            out = compute_line_detection_from_batch(X, pred, content, height, width, seg_threshold=t)
            predictions.append(out["prediction"])
            targets.append(out["target"])
            accuracies.append(out["accuracy"])

        acc = torch.mean(torch.stack(accuracies))
        print("mean recipient line recognition rate", acc)
        predictions = torch.cat(predictions)
        targets = torch.cat(targets)
        print(targets.sum(), len(targets))
        out = precision_recall_fscore_support(targets,predictions, average='binary')
        add = (*out, acc)
        ROC[t] = add
        # print(out)
        global_acc = torch.true_divide((predictions == targets).long().sum(), len(predictions))
        # print("global acc", global_acc)

    # # Save it
    # precision = []
    # recall = []
    # for key, val in ROC.items():
    #     precision.append(val[0])
    #     recall.append(val[1])
    #     if key==0.5:
    #         print(val)
    # save_np = np.stack((precision, recall))
    # print(save_np.shape)
    # np.save("tmp/np/unet_balanced.npy",save_np)

def trans_paste(fg_img,bg_img,alpha=1.0,box=(0,0)):
    fg_img_trans = Image.new("RGBA",fg_img.size)
    fg_img_trans = Image.blend(fg_img_trans,fg_img,alpha)
    bg_img.paste(fg_img_trans,box,fg_img_trans)
    return bg_img

def parse_arguments():
    args = ArgumentParser()
    args.add_argument('--root_dir', type=str, default="/cluster/mayr/nbb/vanilla/")
    args.add_argument('--batch_size', type=int, default=4)
    args.add_argument('--model_dir', type=str, default="tmp/models/unet20211216test0")
    args.add_argument('--books', type=str, default="Band2,Band3,Band4")
    args.add_argument('--unet_checkpoint', type=str, default="tmp/models/unet20211216test0/unet-epoch=0-val_loss=0.00.ckpt")
    return args.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run(args)