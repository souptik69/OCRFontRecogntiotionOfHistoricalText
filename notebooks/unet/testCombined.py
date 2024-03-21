import torch
from pytorch_lightning import seed_everything

from dataset.combined.CombinedDataModule import CombinedDataModule
from model.combined.CombinedModel import CombinedModel
import pytorch_lightning as pl

print("Cuda available?", torch.cuda.is_available())

RANDOM_SEED = 47
torch.backends.cudnn.deterministic = True
seed_everything(RANDOM_SEED)


def run():
    combinedDataModule = CombinedDataModule()
    combinedDataModule.prepare_data()
    combinedDataModule.setup()
    test_data = combinedDataModule.test_dataloader()

    model_base_path = "./../../tmp/models"

    params = {
        "unet_path": f"{model_base_path}/att-unet_unbalanced.ckpt",
        "unet_arch": "att_unet",
        "lin_rec_path": f"{model_base_path}/text_class_cnn.ckpt",
        "lin_rec_arch": "cnn",
        "htr_model_path": f"{model_base_path}/htr_ctc.ckpt",
        "plot_path": "new_test"
    }

    model = CombinedModel(params)
    trainer = pl.Trainer(gpus=1, weights_summary="full")

    trainer.test(model=model, test_dataloaders=test_data, verbose=True)

if __name__ == "__main__":
    run()
