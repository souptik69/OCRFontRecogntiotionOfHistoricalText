import os
from argparse import ArgumentParser
from tqdm import tqdm
import torch
from torch import nn
from torchvision import transforms

from dataset.entityTransformer.ETDataModule import ETDataModule
from model.EntityTransformer.entity_module import EntityModule
from model.unet.UnetModel import UnetModel
from utils.alphabet import Alphabet
from utils.constants import *
from utils.efficiency import runF
from utils.metrics.LineDetectionMetricUtils import convert_segmentation_to_line_prediction
from utils.metrics.TextMetricUtils import cer, wer


def run(args):
    os.makedirs(args.save_dir, exist_ok=True)
    fname_pred = "raw_pred_"+args.name
    check_fname = os.path.join("/tmp",fname_pred+".pkl.gz")
    print(check_fname)
    if not os.path.isfile(check_fname) or args.recompute:
        if args.books is not None:
            books = args.books.split(",")

        dataModule = ETDataModule(root_dir=args.root_dir, batch_size=args.batch_size, multiple_books=books,
                                  line_height=args.line_height, mode="both")
        dataModule.prepare_data()
        dataModule.setup(stage="test")
        test_data = dataModule.test_dataloader()

        unet = UnetModel.load_from_checkpoint(args.unet_checkpoint, strict=False).eval()
        et = EntityModule.load_from_checkpoint(args.et_checkpoint, strict=False).eval()

        out = runF(fname=fname_pred, overwrite=args.recompute, func=predict, unet=unet, et=et, dl=test_data)
    # get raw predictions of UNet and ET
    else:
        out = runF(fname=fname_pred, overwrite=args.recompute, func=predict, unet=None, et=None, dl=None)


    # calculate unet pred for each line:

    for key, value in out.items():
        original_image = value["original_image"]
        unet_pred = value["unet_pred"]
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(size=original_image.size),
                                        transforms.ToTensor()])
        # transform = transforms.Resize(size=original_image.size)
        # print(unet_pred.shape)
        value["unet_pred_line"] = convert_segmentation_to_line_prediction(prediction=transform(value["unet_pred"]),
                                                                          content_infos=value["content_infos"],
                                                                          threshold=0.5)

    # calculate metrics
    ## calculate et metrics
    et_pred_pure, et_pred_processed, gt_recipient = list(), list(), list()
    pred_str, tgt_str = list(), list()
    for key, value in out.items():
        gt_pred_recipient = []
        for c in value["content_infos"]:
            gt_pred_recipient.append(c.is_recipient)
        value["gt_recipient"] = torch.LongTensor(gt_pred_recipient)
        value["pred_recipient_pure"], value["pred_recipient_processed"] = postprocess_recipient_predictions(value["pred_recipient"], pred_probs=value["pred"])
        et_pred_pure.append(value["pred_recipient_pure"])
        et_pred_processed.append(value["pred_recipient_processed"])
        gt_recipient.append(value["gt_recipient"])
        pred_str += value["pred_str"]
        tgt_str += value["tgt_str"]

    et_pred_pure = torch.cat(et_pred_pure)
    et_pred_processed = torch.cat(et_pred_processed)
    gt_recipient = torch.cat(gt_recipient)

    et_pure_acc = torch.true_divide((et_pred_pure==gt_recipient).sum(),len(gt_recipient))
    et_processed_acc = torch.true_divide((et_pred_processed==gt_recipient).sum(),len(gt_recipient))
    print(et_pure_acc, et_processed_acc, len(et_pred_pure))
    print("cer", cer(tgt_str, pred_str))
    print("wer", wer(tgt_str, pred_str))
    print(len((1-(et_pred_processed==gt_recipient).int()).nonzero()))

    ## calculate unet metrics
    unet_pred = list()
    for key, value in out.items():
        unet_pred.append(value["unet_pred_line"])
    unet_pred = torch.cat(unet_pred)

    unet_pred_acc = torch.true_divide((unet_pred == gt_recipient).sum(), len(gt_recipient))
    print("Unet result", unet_pred_acc)

    ## combine them
    ## calculate combined metrics


    # visualize results (be able to turn on and off different visualization renderings)
    visualize = False
    if visualize:
        toPIL = transforms.ToPILImage()
        for key, value in out.items():
            unet_img = toPIL(value["unet_pred"])
            unet_mask = toPIL(value["unet_mask"])
            unet_img.show()
            unet_mask.show()
            break


def postprocess_recipient_predictions(pred_recipient, pred_probs):
    alphabet = Alphabet(dataset="NBB", mode="s2s_recipient")
    sm = nn.Softmax(dim=-1)
    idx_no_recipient, idx_recipient, idx_pad = alphabet.toPosition[END_OF_SEQUENCE_BODY], alphabet.toPosition[END_OF_SEQUENCE_RECIPIENT], alphabet.toPosition[PAD]
    pure_pred_recipient, processed_pred_recipient = list(), list()
    for idx, p in enumerate(pred_recipient):
        if p == torch.LongTensor([idx_recipient]):
            pure_pred_recipient.append(1)
            processed_pred_recipient.append(1)
        elif p == torch.LongTensor([idx_no_recipient]):
            pure_pred_recipient.append(0)
            processed_pred_recipient.append(0)
        elif p==torch.LongTensor([idx_pad]):
            pure_pred_recipient.append(-1)
            probs = sm(pred_probs[idx]).sum(0)
            if probs[idx_recipient]>probs[idx_no_recipient]:
                processed_pred_recipient.append(1)
            else:
                processed_pred_recipient.append(0)
        else:
            print("found",p)
    assert len(pure_pred_recipient)==len(processed_pred_recipient)
    return torch.LongTensor(pure_pred_recipient), torch.LongTensor(processed_pred_recipient)


def predict(unet, et, dl):
    # don't forget cuda | dl needs content_infos
    with torch.no_grad():
        if torch.cuda.is_available():
            unet = unet.cuda()
            et = et.cuda()
        pred_dict = dict()
        sigmoid = torch.nn.Sigmoid()
        for batch in tqdm(dl):
            # visual pred
            image_unet, mask_unet = batch[IMAGE], batch[MASK]
            if torch.cuda.is_available():
                image_unet, mask_unet = image_unet.cuda(), mask_unet.cuda()
            unet_preds = sigmoid(unet(image_unet)).cpu()
            mask_unet = mask_unet.cpu()
            content_infos = batch["content_infos"]
            original_image = batch["original_image"]
            # semantic pred
            et_preds = []
            alphabet = Alphabet(dataset="NBB", mode="s2s_recipient")
            end_tokens = [alphabet.toPosition[END_OF_SEQUENCE_BODY], alphabet.toPosition[END_OF_SEQUENCE_RECIPIENT]]
            for lb_idx, line_batch in enumerate(batch["line_batch"]):
                et_pred = et_inference(et, line_batch, alphabet).cpu()
                et_preds.append(et_pred)
                pred_str, pred_recipient = alphabet.batch_logits_to_string_list(torch.argmax(et_pred, dim=2).long(), stopping_logits=end_tokens)
                tgt_str, tgt_recipient = alphabet.batch_logits_to_string_list(line_batch[S2S_TEXT][:, 1:].cpu(),
                                                                  stopping_logits=end_tokens)
                content_str = []
                for c in content_infos[lb_idx]:
                    content_str.append(c.content)
                # check if tgt texts are equal
                check_tgt_texts(tgt_str, content_str)
                # add all infos
                output = {"unet_pred": unet_preds[lb_idx],
                          "unet_mask": mask_unet[lb_idx],
                          "pred_str": pred_str,
                          "pred_recipient": pred_recipient,
                          "tgt_str": tgt_str,
                          "pred": et_pred,
                          "content_infos": content_infos[lb_idx],
                          "original_image": original_image[lb_idx]}
                pred_dict[content_infos[lb_idx][0].png_file_name] = output
        return pred_dict

def check_tgt_texts(l1, l2):
    for t1, t2 in zip(l1,l2):
        if t1 != t2:
            raise ValueError(f"Error for: t1:{t1}, t2:{t2}")

def et_inference(et, batch, alphabet):
    x, tgt, tgt_key_padding_mask, tgt_mask = batch[LINE_IMAGE],batch[S2S_TEXT], batch[TGT_KEY_PADDING_MASK], batch[TGT_MASK]
    if torch.cuda.is_available():
        x, tgt, tgt_key_padding_mask, tgt_mask = x.cuda(), tgt.cuda(), tgt_key_padding_mask.cuda(), tgt_mask.cuda()
    pred_logits = (torch.ones(size=tgt.shape) * alphabet.toPosition[PAD]).long()
    pred = torch.ones(size=(*tgt.shape, len(alphabet.toPosition)))
    pred_logits[:, 0] = (torch.ones(size=pred_logits[:, 0].shape) * alphabet.toPosition[START_OF_SEQUENCE]).long()
    if x.is_cuda:
        pred = pred.cuda()
        pred_logits = pred_logits.cuda()
    for i in range(1, tgt.shape[-1]):
        out, attention = et(x, tgt=pred_logits[:, :i])
        pred_logits[:, i] = torch.argmax(out, dim=2).long()[:, -1]
        pred[:, i - 1] = out[:, -1, :]
    return pred[:,:-1]


def parse_arguments():
    args = ArgumentParser()
    args.add_argument('--name', type=str, required=True)
    args.add_argument('--recompute', action='store_true')
    args.add_argument('--root_dir', type=str, default="/cluster/mayr/nbb/vanilla/")
    args.add_argument('--batch_size', type=int, default=16)
    # ET params
    args.add_argument('--n_head', type=int, default=1)
    args.add_argument('--line_height', type=int, default=64)
    args.add_argument('--hidden_size', type=int, default=256)
    args.add_argument('--dropout', type=float, default=0.1)
    args.add_argument('--save_dir', type=str, default="../../tmp/prediction")
    args.add_argument('--books', type=str, default="Band3,Band4")
    args.add_argument('--lr', type=float, default=0.0001)
    args.add_argument('--eps', type=float, default=0.4)
    args.add_argument('--noise_teacher_forcing', type=float, default=0.2)
    args.add_argument('--et_checkpoint', type=str, default="../../tmp/models/et_lre-4_e25.ckpt")
    # UNet params
    args.add_argument('--unet_checkpoint', type=str, default="../../tmp/models/att_unet_bin_20211223/unet-epoch=27-val_loss=0.03.ckpt")
    # args.add_argument('--unet_architecture', type=str, default="")

    return args.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run(args)