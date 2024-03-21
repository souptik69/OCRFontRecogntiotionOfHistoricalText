from dataset.unet.UnetContentInfo import UnetContentInfo


class ApplyRecipientInfoToMask(object):
    def __init__(self, content_infos: UnetContentInfo):
        self.content_infos = content_infos

    def __call__(self, img):

        for info in self.content_infos:
            img[:, info.vpos:(info.vpos + info.height), info.hpos: (info.hpos + info.width)] = 1

        return img
