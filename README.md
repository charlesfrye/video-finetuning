# Deploy a personalized music video generation service on Modal

Music videos are [cool](https://youtu.be/Cye-1RP5jso),
but unless you are famous or
[pay a lot of money](https://youtu.be/kfVsfOSbJY0),
you don't get to star in them.

Until now!

This repo includes all the code you need to deploy a custom
music video generator on [Modal](https://modal.com),
a serverless infrastructure platform for data, ML, and AI applications.

Below is a sample video, generated by Modal Developer Advocate
[`@charles_irl`](https://twitter.com/charles_irl).

https://github.com/user-attachments/assets/5bd90898-7251-4298-808f-6d58ed4c6b6f

And because Modal is
[generic serverless infrastructure](https://twitter.com/charles_irl/status/1819438860771663923),
you can customize this custom music video generator however you wish --
it's just code and containers!

## Setup

In the Python environment of your choosing,
run `pip install modal`.

If you run into trouble with Python environments,
we suggest using
[this Google Colab notebook](https://colab.research.google.com/github/charlesfrye/video-finetuning/blob/main/notebooks/self_contained.ipynb),
where we've set the environment up for you.
It's a bit of work to get used to running terminal commands in a notebook
if you haven't done that before, but the Python setup works and running the notebook in Colab is free!
All you need is a Google account.

Then, if you've never used Modal on the computer you're using,
run `modal setup` to create an account on Modal (if you don't have one)
and set up authentication.

## Data Prep

Create a folder inside `data/`, parallel to the sample data, `data/sample`.
You can name it whatever you want.

Place at least four images of yourself in that folder --
ideally eight or more.
Images should be in `.png` or `.jpg` format
and around 400 to 800 pixels on each side.
For best results, we recommend putting a variety of images,
in particular where you are wearing different clothes and making different faces,
and including some images that have other people in them.
But you can also just take a few photos of yourself right now!

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

Open the training notebook, `training.ipynb`.

Read the notebook and run it, following the instructions to edit cells as needed.

In particular, change the dataset path to the folder you created --
it has been mounted on the remote cloud machine where the notebook is running.

You can also directly upload data to the `/root/data` folder on the remote machine.
You can even edit caption files inside of JupyterLab!
This data will stick around between runs, and you can find it with

```bash
modal volume ls finetune-video-data
```

See the help for `modal volume` and its subcommands for details.

The notebook will kick off training, which takes a few minutes.
Take note of the name given to your training run.
By default, it's a hash like `38c67a92f6ce87882044ab53bf94cce0`,
but you can customize it in the notebook.
This is your `finetune-id`.

If you forget it, you can show all of your `finetune-id`s
by running

```bash
modal volume ls finetune-video-models
```

## Inference

Test out your new fine-tuned model by running:

```bash
modal run inference.py --finetune-id {your-finetune-id} --num-frames 15
```

You can also provide a `--prompt` to customize the generation.

You can deploy the video generator onto Modal with

```bash
modal deploy inference.py
```

Modal is serverless, so this won't cost you any money when it isn't serving any traffic.

## Music video generation

Once you've deployed an inference endpoint,
you can generate a music video starring yourself by running

```bash
modal run music_video_generator.py --finetune-id {your-finetune-id}
```

With the default settings, this will create a thirty second video in about five minutes
by running generation in parallel on seven H100s.

The music can be changed by passing in a different song via the `--mp3-file` argument.
The default is a Modal-themed song in `data/coding-up-a-storm.mp3`.
This song was created with [Suno](https://suno.com),
a music generation service -- that runs on Modal!
If you want to DIY music generation as well,
see [this example](https://modal.com/docs/examples/musicgen)
in the Modal docs.

The generated clips can be changed by passing a different list of prompts via the `--prompt-file` argument.
The default is a set of prompts created with OpenAI's GPT-4.5 system.
You can write your own or generate them with a language model.
If you want to serve your own language model,
see [this example](https://modal.com/docs/examples/vllm_inference)
in the Modal docs.
