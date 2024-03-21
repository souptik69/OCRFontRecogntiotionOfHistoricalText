import os
from tqdm import tqdm
from application.helper.page_helper import load_page_and_draw, build_entries

def prepare_images(book = "Band6"):
    entries = build_entries(os.path.join("tmp","new_dataset"), book)

    for e in tqdm(entries):
        image = load_page_and_draw(e, precomputing=True)
        file_name = e.image_file.replace("png", "jpg")
        save_path = os.path.join("tmp","new_dataset",book,"app",file_name)
        image.save(save_path, quality=30)


if __name__ == '__main__':
    prepare_images()
