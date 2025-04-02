## Data Prep

Create a folder inside `data/`, parallel to the sample data, `data/sample`.
You can name it whatever you want.

Place at least four images of yourself in that folder.
Images should be in `.png` or `.jpg` format.

Optionally, add captions in `.txt` files in that same folder.
They should look something like
`"[trigger] smiling at the camera, outdoor scene, close-up, selfie"`.
See the sample data for more example image-caption pairs.

## Training

Start up a JupyterLab server on Modal with

```bash
modal run train_from_notebook.py
```

Click the `modal.host` URL that appears in the output
to open Jupyter in the browser.
Provide the password, `1234`.

Open the training notebook, `training.ipynb`.

Read the notebook and run it. Change the dataset path to the folder you created --
which has been mounted on the remote cloud machine where the notebook is running.

Take note of the name given to your training run.
By default, it's a hash like `38c67a92f6ce87882044ab53bf94cce0`,
but you can customize it in the notebook.
This is your `finetune-id`.

## Inference

Test out your new fine-tuned model by running:

```bash
modal run inference.py --finetune-id {your-finetune-id} --num-frames 15
```

You can deploy the video generator onto Modal with

```bash
modal deploy inference.py
```

Modal is serverless, so this won't cost you any money when it isn't serving any traffic.

## Music video generation

Generate a music video starring yourself by running

```bash
modal run music_video_generator.py --finetune-id {your-finetune-id}
```

With the default settings, this will create a thirty second video in about five minutes
by running generation in parallel on seven H100s.

The music can be changed by passing in a different song via the `--mp3-file` argument
and the generated clips can be changed by passing a different list of prompts via the `--prompt-file` argument.
