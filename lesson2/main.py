import fastbook
fastbook.setup_book()

from fastbook import *
from fastai.vision.widgets import *

def get_dog_images():
    images = search_images_ddg('dog')
    images = ['https://lh3.googleusercontent.com/-qnjC1af4pg4/TYfmcH-DrTI/AAAAAAAAD74/jVwpZWYi1Q8/s1600/dog+%252849%2529.jpg']
    dest = 'images/dog.jpg'
    download_url(images[0], dest)

    im = Image.open(dest)
    im.to_thumb(128, 128)

def populate_animal_images():
    animal_types = ['dog','cat','wolf']
    path = Path('animals')

    if not path.exists():
        path.mkdir()
        for i in range(0, len(animal_types)):
            print(i)
            print(animal_types[i])
            dest = (path / animal_types[i])
            dest.mkdir(exist_ok=True)
            results = search_images_ddg(animal_types[i])
            download_images(dest, urls=results)

    fns = get_image_files(path)
    failed = verify_images(fns)
    failed.map(Path.unlink)


def test_data_loaders():
    dogs = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(128))

    path = Path('animals')

    dogs = dogs.new(item_tfms=Resize(224), batch_tfms=aug_transforms(mult=2))
    dls = dogs.dataloaders(path)
    dls.train.show_batch(max_n=8, nrows=2, unique=True)

if __name__ == "__main__":
    # get_dog_images()
    # populate_animal_images()
    # test_data_loaders()



