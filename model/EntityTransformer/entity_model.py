from PIL import Image
import torch
from torch import nn
import pytorch_lightning as pl
from torchvision import transforms
from model.EntityTransformer.module_factory import produce_feature_extractor, produce_adaptive_2d_PE, produce_transformer, produce_1d_PE
from utils.alphabet import Alphabet
class EntityModel(pl.LightningModule):
    def __init__(self, vocab_size, hidden_size=2048, n_head=1, dropout=0.1):
        super(EntityModel, self).__init__()
        ## ENCODER
        # cut R50 as encoder (1/32 scaling in height and width direction)
        self.feature_extractor = produce_feature_extractor()
        # conv2d to match the dimension size of the transformer encoder
        self.conv_hidden = nn.Conv2d(512, hidden_size, (1,1), stride=(1,1))
        # adaptive 2d positional encoding (most complex step!!)
        self.pe2d = produce_adaptive_2d_PE(d_model=hidden_size)
        # flatten -> only has to be done in the forward part
        # one transformer for all
        self.transformer = produce_transformer(d_model=hidden_size, n_head=n_head, dim_feedforwad=hidden_size, dropout=dropout)

        ## DECODDER
        # embedding and 1d pe
        self.embedder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.pe1d = produce_1d_PE(d_model=hidden_size, dropout=dropout)
        # linear layer to output alphabet
        self.predictor = nn.Linear(hidden_size,vocab_size)

    def forward(self, x, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        # visual
        features = self.conv_hidden(self.feature_extractor(x))
        features_pe = self.pe2d(features)
        s = features_pe.shape
        seq_in = features_pe.reshape(s[0],s[1],-1).permute(2,0,1)
        # text
        y_emb = self.pe1d(self.embedder(tgt)).permute(1, 0, 2)
        transformer_out = self.transformer(src=seq_in, tgt=y_emb, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        seq_out = transformer_out[0].permute(1,0,2)
        attention_map = transformer_out[1]
        pred = self.predictor(seq_out)
        return pred, attention_map

#### Font Detection ####


class EntityModelFont(pl.LightningModule):
    def __init__(self, vocab_size, font_classes, hidden_size=2048, n_head=1, dropout=0.1):
        super(EntityModelFont, self).__init__()
          ## ENCODER
        # cut R50 as encoder (1/32 scaling in height and width direction)
        self.feature_extractor = produce_feature_extractor()
        # conv2d to match the dimension size of the transformer encoder
        self.conv_hidden = nn.Conv2d(512, hidden_size, (1,1), stride=(1,1))
        # adaptive 2d positional encoding (most complex step!!)
        self.pe2d = produce_adaptive_2d_PE(d_model=hidden_size)
        # flatten -> only has to be done in the forward part
        # one transformer for all
        self.transformer = produce_transformer(d_model=hidden_size, n_head=n_head, dim_feedforwad=hidden_size, dropout=dropout)

        ## DECODDER
        # Character embedding and Font Embedding and 1d pe
        # A = Alphabet(dataset="NBB", mode="attention")
        self.embedder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=int(hidden_size/2))
        self.font_embedder = nn.Embedding(num_embeddings=font_classes, embedding_dim=int(hidden_size/2))
        self.pe1d = produce_1d_PE(d_model=int(hidden_size/2), dropout=dropout)
        # linear layer to output alphabet
        self.char_predictor = nn.Linear(int(hidden_size/2),vocab_size)
        # linear layer to output font class
        self.font_predictor = nn.Linear(int(hidden_size/2),font_classes)

    def forward(self, x, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        # visual
        features = self.conv_hidden(self.feature_extractor(x))
        features_pe = self.pe2d(features)
        s = features_pe.shape
        seq_in = features_pe.reshape(s[0],s[1],-1).permute(2,0,1)
        # text and font
        char_logits = tgt[:, :, 0]
        font_classes = tgt[:, :, 1]
        with torch.autograd.detect_anomaly():
            char_embedded = self.pe1d(self.embedder(char_logits)).permute(1, 0, 2)
        font_embedded = self.pe1d(self.font_embedder(font_classes)).permute(1, 0, 2)
        y_emb = torch.cat((char_embedded, font_embedded), dim=-1)
        # print(y_emb.shape)
        # y_emb = self.pe1d(self.embedder(tgt)).permute(1, 0, 2)
        transformer_out = self.transformer(src=seq_in, tgt=y_emb, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        # print(transformer_out[0].size())
        char_seq_out = transformer_out[0][:,:,:y_emb.shape[-1]//2].permute(1,0,2)
        font_seq_out = transformer_out[0][:,:, -y_emb.shape[-1]//2:].permute(1,0,2)
        attention_map = transformer_out[1]
        pred_char = self.char_predictor(char_seq_out)
        pred_font = self.font_predictor(font_seq_out)
        return pred_char,pred_font, attention_map

    


#### Font Detection ####

if __name__ == "__main__":
    # m = EntityModel(vocab_size=89)
    # x = torch.randn(10,1,128,2048)
    # y = torch.randint(low=0, high=89, size=(10,50)).long()
    # out, attention_map = m(x,y)
    # print(out.shape)
    # attention_map -= attention_map.min(1,keepdim=True)[0]
    # attention_map /= attention_map.max(1,keepdim=True)[0]
    # toPIL = transforms.ToPILImage()
    # attention_img = toPIL(attention_map[0].unsqueeze(0))
    # print(attention_map.shape) #torch.Size([10, 50, 256])
    vocab_size = 89
    font_classes = 12  # You mentioned 12 font classes
    hidden_size = 2048
    n_head = 1
    dropout = 0.1

    m = EntityModelFont(vocab_size=vocab_size, font_classes=font_classes, hidden_size=hidden_size, n_head=n_head, dropout=dropout)
    
    # Generate some random data for testing
    x = torch.randn(10, 1, 128, 2048)
    y = torch.randint(low=0, high=vocab_size, size=(10, 50, 2)).long()  # 2 for character and font class
    
    # Forward pass through the model
    pred_char, pred_font, attention_map = m(x, y)
    
    # Check the shapes of the predictions
    print("Character Prediction Shape:", pred_char.shape)
    print("Font Class Prediction Shape:", pred_font.shape)
    
    # Check the attention map shape
    print("Attention Map Shape:", attention_map.shape)
    
    # Normalize and visualize the attention map (assuming batch size is 1)
    attention_map -= attention_map.min(1, keepdim=True)[0]
    attention_map /= attention_map.max(1, keepdim=True)[0]
    
    toPIL = transforms.ToPILImage()
    attention_img = toPIL(attention_map[0].unsqueeze(0))
    print("Normalized Attention Map Shape:", attention_map.shape)
