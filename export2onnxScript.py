import torch

from dataset.img2seq.Img2SeqDataModule import Img2SeqDataModule
from model.LinRecClassifier.LinRecClassifierCnn import LinRecClassifierCnn
from model.LinRecClassifier.LinRecClassifierRnn import LinRecClassifierRnn
from model.LinRecClassifier.LinRecModel import LinRecModel
from model.htr.ctc_model import CTCModule
from model.htr.gcrnn import GCRNN


def export():
    #data_module = Img2SeqDataModule(img_height=64, batch_size=32, use_custom_augmentation=True,
    #                                content_file="gt.txt", img_dir="",
    #                                root_dir="C:\\Git\\Uni\\nbb-recipient-detection\\tmp\\dataset\\vanilla",
    #                                books="Band2,Band3,Band4", recipient_information=True)
#
    #data_module.prepare_data()
    #data_module.setup()
#
    #test_data = data_module.test_dataloader()

    model = LinRecClassifierRnn()

    batch = torch.rand(32, 102).long()

    torch.onnx.export(model,  # model being run
                      batch,  # model input (or a tuple for multiple inputs)
                      "lin_rec_rnn_model.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=False,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    print("Done??")

if __name__ == "__main__":
    export()