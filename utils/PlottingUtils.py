import matplotlib.pyplot as plt
from utils.metrics.TextConverter import convert_vector_to_text
from utils.constants import IMAGE, MASK, TEXT


def plot_img2seq_batch(data_loader, alphabet, num=5):
    batch = next(iter(data_loader))
    lookup_alphabet = {value: key for key, value in alphabet.items()}

    f, ax_array = plt.subplots(num, 1)
    for i in range(num):
        image = batch[IMAGE][i]
        text = batch[TEXT][i]
        readable_text = convert_vector_to_text(text, lookup_alphabet)
        _plot_single(ax_array[i], image, readable_text)

    plt.show()


def plot_unet_batch(data_loader, num=3):
    batch = next(iter(data_loader))

    f, ax_array = plt.subplots(num, 2, figsize=(12, 24))
    for i in range(num):
        image = batch[IMAGE][i]
        mask = batch[MASK][i]
        _plot_single(ax_array[i][0], image.permute(1, 2, 0), "Original")
        _plot_single(ax_array[i][1], mask, "Masked")


def _plot_single(ax, image, text):
    ax.set_title(text)
    ax.imshow(image.squeeze(), cmap='Greys_r')
    ax.axis('off')
