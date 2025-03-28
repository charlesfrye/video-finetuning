## Quickstart

1. Create a [Modal](https://modal.com/) account
1. Compile ~5 pictures of a "character" you'd like the model to learn (e.g. your pet, yourself) 
1. Clone this repo
1. Run `modal run modal_jupyter.py` to open a notebook running on Modal
1. Provide the password, 1234
1. Open `training.ipynb`
1. Upload your pictures to a Modal Volume with the following command: `modal volume put finetune-video selfie1.jpg`
1. Run all cells in `training.ipynb` (training takes ~5m)
1. See sample outputted videos featuring your character in `outputs/<model_hash>/samples`
