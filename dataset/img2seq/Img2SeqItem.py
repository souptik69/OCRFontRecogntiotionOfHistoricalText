import torch
import re


class Img2SeqItem:

    def __init__(self, img_file_name: str, content: str, recipient):
        cleared_content = Img2SeqItem._clear_content(content)
        self.recipient = recipient
        self.img_file_name, self.content = img_file_name, cleared_content.strip()

    def get_feature_vector(self, alphabet_dict: dict, max_len: int):
        # 1 is the value for padding tokens ...
        feature_vector = torch.ones(max_len, dtype=torch.long)

        for i, char in enumerate(self.content):
            feature_vector[i] = alphabet_dict[char]

        return feature_vector

    @staticmethod
    def _clear_content(t):
        t = re.sub(u'\u0057\u0308', 'W', t)
        t = re.sub(u'\u0077\u0308', 'w', t)
        t = re.sub(u'\u1e84', 'W', t)
        t = re.sub(u'\u1e85', 'w', t)
        t = re.sub(u'\u0056\u0308', 'V', t)
        t = re.sub(u'\u0076\u0308', 'v', t)
        t = re.sub(u'\u00eb', 'e', t)
        t = re.sub(u'\u0065\u0308', 'e', t)
        t = re.sub(u'\u00cb', 'E', t)
        t = re.sub(u'\u0045\u0308', 'E', t)
        t = re.sub("=", '', t)
        t = re.sub(u'\u0055\u0308', 'Ü', t)
        t = re.sub(u'\u0075\u0308', 'ü', t)
        t = re.sub('\t', '', t)
        t = re.sub(u'y\u0308', 'y', t)
        t = re.sub(u'\u00ff', 'y', t)
        t = re.sub(u'o\u0308', 'ö', t)
        t = re.sub(u'a\u0308', 'ä', t)
        t = re.sub('\+', '', t)
        t = re.sub('\?', '', t)
        t = re.sub('\(', '', t)
        t = re.sub('\)', '', t)
        t = re.sub("[`']", '', t)
        t = re.sub('\$#[^\$#]*#\$', '', t)
        t = re.sub('-\$', '', t)
        t = re.sub('\$-', '', t)
        t = re.sub('&', '%', t)
        t = re.sub(u'\u00e5', 'ä', t)
        t = re.sub('ß', 'ſ', t)
        return t
