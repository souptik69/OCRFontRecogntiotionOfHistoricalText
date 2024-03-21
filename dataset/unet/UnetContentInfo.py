import torch
from utils.clean_txt import clean_string

class UnetContentInfo:

    def __init__(self, text_line_tag, png_file_name, index):
        self.index = index
        string_tag = text_line_tag.find("{http://www.loc.gov/standards/alto/ns-v2#}String")

        self.content = clean_string(string_tag.attrib["CONTENT"])

        self.is_recipient = True if text_line_tag.attrib["RECIPIENT"] == "True" else False

        self.png_file_name = png_file_name

        self.height = int(string_tag.attrib["HEIGHT"])
        self.width = int(string_tag.attrib["WIDTH"])
        self.hpos = int(string_tag.attrib["HPOS"])
        self.vpos = int(string_tag.attrib["VPOS"])

        self._feature_vector = None

    def __repr__(self):
        return f"""{self.content}
Image file: {self.png_file_name}
Recipient: {self.is_recipient}
Heigth: {self.height} - Width: {self.width}
HPos: {self.hpos} - VPos: {self.vpos}"""

    # TODO check if this is used
    def get_feature_vector(self, alphabet_dict, max_len):
        if self._feature_vector is not None:
            return self._feature_vector

        feature_vector = torch.zeros(max_len, dtype=torch.long)
        for index, char in enumerate(self.content):
            feature_vector[index] = alphabet_dict[char]

        self._feature_vector = feature_vector
        return self._feature_vector


