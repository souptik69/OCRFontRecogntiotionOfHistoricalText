from dataset.FontDetection.ETDataset import FontDataset
import zipfile
from utils.alphabet import Alphabet
from torch.utils.data import DataLoader
import torch
from utils.SubSequentMask import subsequent_mask
from utils.CustomCollate import collate_s2s_font
# zip_file_path = '/home/hpc/iwi5/iwi5144h/nbb-recipient-detection/SingleMultiple.zip'  # Replace with the actual path to your zip file
# multiple_fonts = True
# split = 'valid'  # Change this to the desired split ('train', 'valid', or 'test')
# augment = True  
# dataset = FontDataset(zip_file_path,image_size = 256,split=split, augment=augment)

# # Print the total number of samples in the dataset
# print(f"Total number of samples in the '{split}' split: {len(dataset)}")

# Initialize the Alphabet class
# A = Alphabet(dataset="NBB", mode="attention")
# print(len(A.toPosition))
# for i in range(len(A.toPosition)):
#     print('Character',i,'=',A.toCharacter[i])

# print(A.toPosition['`'])
# print(A.toCharacter[9])
    # # unique_font_classes = set()

# # Iterate through the dataset
# for idx in range(len(dataset)):
#     try:
#         sample = dataset[idx]
#         cf = sample['CF']

#         # Iterate through individual elements in the 'CF' array
#         for font_class in cf:
#             unique_font_classes.add(font_class)

#     except Exception as e:
#         print(f"Error processing Sample {idx + 1}: {str(e)}")

# # Print all unique font classes
# print("Unique Font Classes:", unique_font_classes)


# Iterate through the dataset
# missing_characters = set()
# for idx in range(len(dataset)):
#     try:
#         sample = dataset[idx]
#         text = sample['TEXT']
#         # print(f"Sample {idx + 1} TEXT: {text}")

#         # Check if the 'TEXT' characters are in the Alphabet's dictionary
#         for char in text:
#             if char not in A.toPosition:
#                missing_characters.add(char)

#     except Exception as e:
#         print(f"Error processing Sample {idx + 1}: {str(e)}")
# # Print missing characters only once
# if missing_characters:
#     print("Missing Characters:", missing_characters)
# else:
#     print("All characters are present in the Alphabet's dictionary.")

# Choose an index to access a specific sample from the dataset
# sample_index = 0  # Change this to the desired sample index

# try:
#     # Access a sample from the dataset
#     sample = dataset[sample_index]
#     # line_img, text, cf = sample
#     # Print information about the accessed sample
#     print(f"Sample {sample_index + 1}:")
#     # print(line_img.shape)
#     # print(text)
#     # print(cf)
#     print(f"Image Shape: {sample['LINE_IMAGE'].shape}")
#     print(f"Text: {sample['TEXT']}")
#     print(f"CF: {sample['CF']}")

# #     # You can perform further processing or analysis on this sample as needed
# except IndexError:
#     print(f"Sample index {sample_index} is out of range for the dataset.")

# def custom_collate(batch):
#     keys = batch[0].keys()
#     output = dict()
#     A = Alphabet(dataset="NBB", mode="attention")
#     ys =list()
#     cf =list()
#     for item in batch:
#         logits= A.string_to_logits(item['TEXT'])
#         cf_tensor = torch.LongTensor(item['CF'])
#         logits = torch.cat([torch.LongTensor([A.toPosition['<sos>']]), logits, torch.LongTensor([A.toPosition['<eos>']])])
#         cf_tensor = torch.cat([torch.LongTensor([A.toPosition['<sos>']]), cf_tensor, torch.LongTensor([A.toPosition['<eos>']])])
#         cf.append(cf_tensor)
#         ys.append(logits)
#     ys_padded = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=A.toPosition['<pad>'])
#     ys_masking = torch.eq(ys_padded,torch.ones(ys_padded.shape,dtype=torch.long)*torch.LongTensor([A.toPosition['<pad>']]))
#     cf_padded = torch.nn.utils.rnn.pad_sequence(cf, batch_first=True, padding_value=A.toPosition['<pad>'])
#     cf_masking = torch.eq(cf_padded,torch.ones(cf_padded.shape,dtype=torch.long)*torch.LongTensor([A.toPosition['<pad>']]))
#     # output['TGT_KEY_PADDING_MASK'] = ys_masking
#     # output['TGT_MASK'] = subsequent_mask(ys_masking.shape[-1]-1)
#     ys_mask= subsequent_mask(ys_masking.shape[-1]-1)
#     # output['S2S']=ys_padded
#     # output['CF_TGT_KEY_PADDING_MASK'] = cf_masking
#     # output['CF_TGT_MASK'] = subsequent_mask(cf_masking.shape[-1]-1)
#     cf_mask = subsequent_mask(cf_masking.shape[-1]-1)
#     # output['CF']=cf_padded
#     max_seq_len = max(ys_padded.size(1), cf_padded.size(1))
#     if ys_padded.size(1) < max_seq_len:
#         ys_padded = torch.cat((ys_padded, torch.zeros(ys_padded.size(0), max_seq_len - ys_padded.size(1))), dim=1)
#     if cf_padded.size(1) < max_seq_len:
#         cf_padded = torch.cat((cf_padded, torch.zeros(cf_padded.size(0), max_seq_len - cf_padded.size(1))), dim=1)
#     joined_tensor = torch.cat((ys_padded.unsqueeze(2), cf_padded.unsqueeze(2)), dim=2)
#     output['S2S'] = joined_tensor

#     max_seq_len1 = max(ys_masking.size(1), cf_masking.size(1))
#     if ys_masking.size(1) < max_seq_len1:
#         ys_masking = torch.cat((ys_masking, torch.zeros(ys_masking.size(0), max_seq_len1 - ys_masking.size(1))), dim=1)
#     if cf_masking.size(1) < max_seq_len1:
#         cf_masking = torch.cat((cf_masking, torch.zeros(cf_masking.size(0), max_seq_len1 - cf_masking.size(1))), dim=1)
#     joined_tensor_1 = torch.cat((ys_masking.unsqueeze(2), cf_masking.unsqueeze(2)), dim=2)
#     output['TGT_KEY_PADDING_MASK'] = joined_tensor_1

#     max_seq_len2 = max(ys_mask.size(1), cf_mask.size(1))
#     # print(f"ys_mask shape before concatenation: {ys_mask.shape}")
#     # print(f"cf_mask shape before concatenation: {cf_mask.shape}")
#     if ys_mask.size(1) < max_seq_len2:
#         ys_mask = torch.cat((ys_mask, torch.zeros(ys_mask.size(0), max_seq_len2 - ys_mask.size(1))), dim=1)
#     if cf_mask.size(1) < max_seq_len2:
#         cf_mask = torch.cat((cf_mask, torch.zeros(cf_mask.size(0), max_seq_len2 - cf_mask.size(1))), dim=1)
#     # print(f"ys_mask shape after concatenation: {ys_mask.shape}")
#     # print(f"cf_mask shape after concatenation: {cf_mask.shape}")
#     # assert ys_mask.size(1) == cf_mask.size(1)
#     max_batch_size = max(ys_mask.size(0), cf_mask.size(0))

#     # Ensure that both tensors have the same number of rows (batch size) by padding them
#     if ys_mask.size(0) < max_batch_size:
#         ys_mask = torch.cat((ys_mask, torch.zeros(max_batch_size - ys_mask.size(0), ys_mask.size(1))), dim=0)

#     if cf_mask.size(0) < max_batch_size:
#         cf_mask = torch.cat((cf_mask, torch.zeros(max_batch_size - cf_mask.size(0), cf_mask.size(1))), dim=0)

#     # print(f"ys_mask shape after padding: {ys_mask.shape}")
#     # print(f"cf_mask shape after padding: {cf_mask.shape}")

#     # Now both ys_mask and cf_mask have the same number of rows (batch size)
#     assert ys_mask.size(0) == cf_mask.size(0), "Batch sizes don't match after padding"

#     # Continue with the concatenation as before
#     joined_tensor_2 = torch.cat((ys_mask.unsqueeze(2), cf_mask.unsqueeze(2)), dim=2)
#     output['TGT_KEY_MASK'] = joined_tensor_2
#     for key in keys:
#         if key == 'LINE_IMAGE':
#             images = [item[key].permute(2, 0, 1) for item in batch]
#             # print(f'Shape of {key}: {images[0].shape}')
#             images_padded = torch.nn.utils.rnn.pad_sequence(images)
#             output[key] = images_padded.permute(1, 2, 3, 0)


    
#     return output



# train_data =DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8,collate_fn=collate_s2s_font)
# for batch in train_data:
#     # Access the S2S data from the batch
#     s2s_data = batch['S2S']
#     tgt_mask = batch['TGT_KEY_PADDING_MASK']
#     tgt = batch['TGT_KEY_MASK']
#     img = batch['LINE_IMAGE']
#     # cf = batch['CF']
#     # cf_tgt_mask = batch['CF_TGT_KEY_PADDING_MASK']
#     # cf_tgt = batch['CF_TGT_MASK']
#     # joined = batch['JOINED_TENSOR']
#     # Print the shape of the S2S data
#     print('***')
#     print(f"S2S Data Shape: {s2s_data.shape}") 
#     # print(f"CF Data Shape: {cf.shape}") 
#     print(f"TGT PADDING MASK Data Shape: {tgt_mask.shape}") 
#     # print(f"CF_TGT PADDING MASK Data Shape: {cf_tgt_mask.shape}") 
#     print(f"TGT MASK: {tgt.shape}") 
#     print(f"IMG_SHAPE: {img.shape}")
#     # print(f"CF_TGT MASK: {cf_tgt.shape}") 
#     # print(f"JOINED_TENSOR:{joined.shape}")
#     print('***')

# Import your FontDataModule class
# from dataset.FontDetection.ETDataModule import FontDataModule
# from utils.noisy_teacher_forcing import NoisyTeacherFont
# from utils.alphabet import Alphabet
# import torch
# from torch import nn
# from model.EntityTransformer.module_factory import produce_1d_PE
# # Instantiate your data module
# dataModule = FontDataModule(
#     zip_file_path="/home/hpc/iwi5/iwi5144h/nbb-recipient-detection/data.zip",
#     line_height=64,
#     batch_size=16,
#     augment=False
# )

# # Prepare and set up the data module
# dataModule.prepare_data()
# dataModule.setup()

# # Access the train and validation data loaders
# train_data = dataModule.train_dataloader()
# val_data = dataModule.val_dataloader()
# pe1d = produce_1d_PE(d_model=2048, dropout=0.1)
# # Print some information
# print("Train DataLoader Length:", len(train_data))
# print("Validation DataLoader Length:", len(val_data))

# # Iterate through a batch to check the data (optional)
# A = Alphabet(dataset="NBB", mode="attention")
# font_class_vocabulary = list(range(14))
# for batch in train_data:
#     print("Batch Shape:", batch['LINE_IMAGE'].shape)
#     print("Sample Text:", batch['S2S'][0])  # Print text of the first sample
#     print('First Padded tensor:', batch['TGT_KEY_PADDING_MASK'][0])
#     print('First Masked tensor:', batch['TGT_KEY_MASK'][0])
#     # print("Sample target:", batch['TGT_KEY_MASK'][0])
#     noisy_teacher = NoisyTeacherFont(character_vocab_size=len(A.toPosition), font_class_count=len(font_class_vocabulary), noise_prob = 0.1)
#     tgt = batch['S2S']
#     print("target Shape:", tgt.shape)
#     tgt = noisy_teacher(tgt[:,:-1])
#     print("NoisyTeacherEnforced:", tgt[0])
#     print("NoisyTeacherEnforced:", tgt.shape)
#     char_logits = tgt[:, :, 0]
#     font_classes = tgt[:, :, 1]
#     embedder = nn.Embedding(num_embeddings=len(A.toPosition), embedding_dim=2048)
#     font_embedder = nn.Embedding(num_embeddings=len(font_class_vocabulary), embedding_dim=2048)
#     char_embedded = pe1d(embedder(char_logits)).permute(1, 0, 2)
#     font_embedded = pe1d(font_embedder(font_classes)).permute(1, 0, 2)
#     combined_embedded = torch.cat((char_embedded, font_embedded), dim=-1)
#     # font_embeddings = font_embedder(tgt[:, :, 1])
#     # combined_embeddings = embedder(tgt[:, :, 0]) + font_embeddings
#     print("Embedding Shape:", combined_embedded.shape)
#     break  # Exit the loop after checking the first batch

# You can similarly check the validation data loader
# from model.EntityTransformer.entity_model import EntityModelFont
# import torch
# from torchvision import transforms

# if __name__ == "__main__":
#     vocab_size = 89
#     font_classes = 14  # You mentioned 12 font classes
#     hidden_size = 2048
#     n_head = 1
#     dropout = 0.1

#     m = EntityModelFont(vocab_size=vocab_size, font_classes=font_classes, hidden_size=hidden_size, n_head=n_head, dropout=dropout)
    
#     # Generate some random data for testing
#     x = torch.randn(10, 1, 128, 2048)
#     # y = torch.randint(low=0, high=vocab_size, size=(10, 50, 2)).long()  # 2 for character and font class
#     char_values = torch.randint(low=0, high=vocab_size, size=(10, 50, 1)).long()
#     font_values = torch.randint(low=0, high=font_classes, size=(10, 50, 1)).long()
#     y = torch.cat((char_values, font_values), dim=2)
#     # Forward pass through the model
#     pred_char, pred_font, attention_map = m(x, y)
    
#     # Check the shapes of the predictions
#     print("Character Prediction Shape:", pred_char.shape)
#     print("Font Class Prediction Shape:", pred_font.shape)
    
#     # Check the attention map shape
#     print("Attention Map Shape:", attention_map.shape)
    
#     # Normalize and visualize the attention map (assuming batch size is 1)
#     attention_map -= attention_map.min(1, keepdim=True)[0]
#     attention_map /= attention_map.max(1, keepdim=True)[0]
    
#     toPIL = transforms.ToPILImage()
#     attention_img = toPIL(attention_map[0].unsqueeze(0))
#     print("Normalized Attention Map Shape:", attention_map.shape)
# import os
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from dataset.FontDetection.ETDataModule import FontDataModule
# from model.EntityTransformer.entity_module import EntityModuleFont
# from utils.alphabet import Alphabet
# import pytorch_lightning as pl


# data_module = FontDataModule(zip_file_path="/home/hpc/iwi5/iwi5144h/nbb-recipient-detection/SingleMultiple.zip",line_height=64, batch_size=32, augment=True)
# data_module.prepare_data()
# data_module.setup()
# train_data = data_module.train_dataloader()
# val_data = data_module.val_dataloader()

# A = Alphabet(dataset="NBB", mode="attention")
# font_class_vocabulary = list(range(15))
# model = EntityModuleFont(vocab_size=len(A.toPosition),
#                          font_classes=len(font_class_vocabulary), hidden_size=512,
#                          n_head=1,dropout=0.1, lr=0.0001, eps=0.4, 
#                          noise_teacher_forcing=0.2)
# checkpoint_callback = ModelCheckpoint(dirpath=os.path.join("/home/hpc/iwi5/iwi5144h/nbb-recipient-detection/SingleMultipleFontModel","SingleMultipleFonts"), filename='et-{epoch}',
#                                           monitor="val/total_loss", save_top_k=1, mode="min")
# early_stopping = EarlyStopping(monitor="val/total_loss", patience=30)

# trainer = pl.Trainer(devices=1, num_nodes=1,callbacks=[checkpoint_callback, early_stopping])
# trainer.tune(model, train_dataloader=train_data)
# trainer.fit(model, train_data, val_data)


# data_module = FontDataModule(zip_file_path="/home/hpc/iwi5/iwi5144h/nbb-recipient-detection/SingleMultiple.zip", image_size= 256, batch_size=16, augment=False)
# data_module.prepare_data()
# data_module.setup(stage="test")
# # train_data = data_module.train_dataloader()
# # val_data = data_module.val_dataloader()
# test_data = data_module.test_dataloader()

# A = Alphabet(dataset="NBB", mode="attention")
# font_class_vocabulary = list(range(14))
# model = EntityModuleFont.load_from_checkpoint('/home/hpc/iwi5/iwi5144h/nbb-recipient-detection/SingleMultipleFontModel/SingleMultipleFonts6/et-epoch=11.ckpt', strict=False)

# # checkpoint_callback = ModelCheckpoint(dirpath=os.path.join("/home/hpc/iwi5/iwi5144h/nbb-recipient-detection/Saved_model_CheckPoint","MultipleFonts"), filename='et-{epoch}',
# #                                           monitor="val/total_loss", save_top_k=1, mode="min")
# # early_stopping = EarlyStopping(monitor="val/total_loss", patience=10)

# trainer = pl.Trainer(devices=1, num_nodes=1)
# # trainer.tune(model, train_dataloader=train_data)
# # trainer.fit(model, train_data, val_data)
# trainer.test(model=model, dataloaders= test_data)

# from utils.alphabet import Alphabet
# import torch
# A = Alphabet(dataset="NBB", mode="attention")
# batch = torch.LongTensor([[0,15,10,3,3,14,3,3],[20,4,5,6,14,2,2,2]])
# out = A.batch_logits_to_list_font(batch, stopping_logits=[14])
# print(out)

import numpy as np
from utils.metrics.TextMetricUtils import font_acc
from utils.metrics.TextMetricUtils import int_cer
font_gt = [np.array([1,2,3]), np.array([4, 4,5]), np.array([2, 2])]
font_pred = [np.array([1, 2, 3]), np.array([4, 4,5,]), np.array([2, 2])]

#Calculate accuracy
accuracy = font_acc(font_gt, font_pred)
cer = int_cer(font_gt,font_pred)
print('CER:', cer.item())
# Print the accuracy
print("Accuracy:", accuracy.item()) 
# print("Hello World,

# import wandb
# api = wandb.Api()
# run = api.run("/fontdetection/SingleMultipleFonts8/runs/wvdtaj94")
# print(run.history())