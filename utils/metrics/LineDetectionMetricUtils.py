import torch
from torchvision import transforms


def convert_segmentation_to_line_prediction(prediction, content_infos, threshold=0.5):
    prediction = (prediction >= threshold).long()
    line_predictions = []
    for info in content_infos:
        line_predictions.append(
            torch.mean(prediction[:, info.vpos:(info.vpos + info.height), info.hpos: (info.hpos + info.width)].float()))
    if len(line_predictions) == 0:
        print("Something went wrong.")
    return torch.stack(line_predictions)


def compute_line_detection_from_batch(imgs, preds, content_infos, heights, widths, seg_threshold=0.0):
    sigmoid = torch.nn.Sigmoid()
    toPIL = transforms.ToPILImage()
    preds = sigmoid(preds)
    for img, pred, content, height, width in zip(imgs, preds, content_infos, heights, widths):
        if len(content) == 0:
            continue
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((height, width)),
            transforms.ToTensor(),
        ])
        pred = transform(pred)
        out = compute_line_detection_accuracy(pred, content, seg_threshold=seg_threshold)
        return out


def compute_line_detection_accuracy(prediction, target, probs=True, seg_threshold=0.5, line_threshold=0.5):
    r""" Computes the line detection rate whether the input line is a recipient or not.

    Args:
        prediction: either segmentation mask or line-wise predictions
        target: all lines of the page with its labels and line coords
        probs: whether the predictions are probabilities are logits
        seg_threshold: used when probabilities are given and has to be used to calculate the logits
        line_threshold: used when probabilities are given and has to be applied to get the logit per line

    Return:
        score: accuracy for the whole page
        n_lines: number of lines used for computing the accuracy (lateron can be used to better weight the accuracy of each page for a global result)

    Approach:
        - Check whether probs=True
        - Check whether input is a segmentation mask or line-wise predictions. Use shape of prediciton as criterium:
            first channel has to have length 1, (folllowing dimensions have to be higher than a specific value) and the length of the shape is 3
            - if true:
                - use seg_threshold
                - transfer to line-wise predictions
        - compute logits of line-wise predictions
        - compute score
    """
    if probs:
        if len(prediction.shape) == 3 and prediction.shape[0] == 1:
            prediction = convert_segmentation_to_line_prediction(prediction, target, threshold=seg_threshold)

        prediction = (prediction > line_threshold).long()

    target_labels = [t.is_recipient for t in target]
    target_labels = torch.Tensor(target_labels).long()

    assert len(prediction) == len(target_labels)

    accuracy = torch.true_divide((prediction == target_labels).long().sum(), len(prediction))

    return {"accuracy": accuracy, "n_lines": len(prediction), "prediction": prediction, "target": target_labels}
