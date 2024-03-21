import string

import torch
import numpy as np
from utils.constants import *


def convert_position_to_text(pos, d):
    out = []
    for p in pos:
        out.append(d[p])
    out = "".join(out)
    return out

class Alphabet():
    def __init__(self, dataset="IAM", mode="ctc"):
        if mode == "ctc":
            extra = ['<blank>','<pad>']
        elif mode== "attention":
            extra = ['<sos>','<eos>','<pad>']
        elif mode == "s2s_recipient":
            extra = [START_OF_SEQUENCE, END_OF_SEQUENCE_RECIPIENT, END_OF_SEQUENCE_BODY, PAD]
        self.seperators = None
        lower = string.ascii_lowercase
        upper = string.ascii_uppercase
        numbers = string.digits
        punctuation = [' ', '.', ',',"'",'-','"','#','(',')',':',';','?','*','!','/','&','+']
        arrows = [u"\u2190", u"\u2191", u"\u2192", u"\u2193", u"\u2194"]
        if dataset=="IAM":
            types = [extra,lower,upper,numbers, punctuation]
        elif dataset=="NBB":
            nbb_characters = ['ſ','ʒ','Ʒ','Ü','ü','Ö','ö','Ä','ä','ò','æ','ꝛ','̃','à','â','ͤ','ô','ù','œ','€','è','á','’','Æ','ᵱ','́','⁊','ç','ꝰ','Ꝝ', '´','ï','¹','¶','ę','⁵','û','⁹','ó','ꝓ','⁶','î','í','̍','²','⁰','ì','†','ű','̍''ú','É','ł','ζ','ꝗ','ú','Ó', 'ↄ','^','ꝟ', 'È','⁴','½','³', '⁷','⁸','đ', '‡', 'Œ','&', 'ë','é', 'ů','°', 'ê', 'ß','ÿ','`'] 
            self.seperators = ['.',':',',','/','~','§','%','-','*',' ','@', '',']','[',';','·','”','"', '‘','“',';',"'",'?', '(', ')','_', '+', '\\', '!','„', '|']
            types = [extra, lower, upper, numbers, nbb_characters, self.seperators]

        self.toPosition = {}
        self.toCharacter = {}
        id = 0
        for t in types:
            for char in t:
                self.toPosition[char] = id
                self.toCharacter[id] = char
                id += 1

    def string_to_logits(self, x_in):
        out = []
        for i in x_in:
            out.append(self.toPosition[i])
        return torch.LongTensor(out)

    def logits_to_string(self, x_in):
        out = []
        for i in x_in:
            out.append(self.toCharacter[int(i)])
        return "".join(out)

    # TODO: clean this shit up!!!
    def batch_logits_to_string_list(self, x_in, stopping_logits:list=None):
        text = []
        classification = []
        for b in x_in:
            if stopping_logits is None:
                text.append(self.logits_to_string(b))
                classification.append(torch.Tensor([-1]))
            else:
                stops = []
                for s in stopping_logits:
                    stop = torch.where(b==s)[0]
                    if len(stop)==0:
                        stop = torch.LongTensor([len(b)])

                    stops.append(stop[0])
                end_idx = torch.min(torch.stack(stops))
                text.append(self.logits_to_string(b[:end_idx]))
                if end_idx==len(b):
                    end_classifier = torch.LongTensor([3])
                    classification.append(end_classifier)
                else:
                    end_classifier = torch.argmin(torch.stack(stops))
                    classification.append(torch.Tensor([stopping_logits[end_classifier]]))
        return text, torch.stack(classification)
    

    ### Font detection ###

    def batch_logits_to_string_list_font(self, x_in, stopping_logits:list=None):
        text = []
        for b in x_in:
            if stopping_logits is None:
                text.append(self.logits_to_string(b))
            else:
                stops = []
                for s in stopping_logits:
                    stop = torch.where(b==s)[0]
                    if len(stop)==0:
                        stop = torch.LongTensor([len(b)])
                    stops.append(stop[0])
                end_idx = torch.min(torch.stack(stops))
                text.append(self.logits_to_string(b[:end_idx]))
        return text
    
    def batch_logits_to_list_font(self, x_in, stopping_logits:list=None):
        font = []
        for b in x_in:
            if stopping_logits is None:
                font.append(b)
            else:
                stops = []
                for s in stopping_logits:
                    stop = torch.where(b==s)[0]
                    if len(stop)==0:
                        stop = torch.LongTensor([len(b)])
                    stops.append(stop[0])
                end_idx = torch.min(torch.stack(stops))
                font.append(b[:end_idx].numpy())
        return font

    ### Font detection ###





if __name__ == "__main__":
    A = Alphabet(dataset="NBB", mode="s2s_recipient")
    logits = A.string_to_logits("bla")
    print(A.logits_to_string(logits))
    batch = torch.LongTensor([[0,15,10,3,3,32,3,3],[20,4,5,6,2,2,2,2]])
    print(batch.shape)
    print(A.batch_logits_to_string_list(batch))
    print("###############")
    logits = torch.cat([logits, torch.LongTensor([A.toPosition[END_OF_SEQUENCE_RECIPIENT]])])
    print(logits)
    print("###############")
    print("###############")
    out = A.batch_logits_to_string_list(batch, stopping_logits=[A.toPosition[END_OF_SEQUENCE_RECIPIENT], A.toPosition[END_OF_SEQUENCE_BODY]])
    print(out)