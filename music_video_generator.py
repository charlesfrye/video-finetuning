from pathlib import Path

import modal


app = modal.App("music-video-generator")

here = Path(__file__).parent


@app.local_entrypoint()
def main(finetune_id, mp3_file=None, prompt_file=None):
    clip_duration = 5  # seconds

    generator = modal.Cls.from_name("finetune-video-generate", "VideoGenerator")(
        finetune_id=finetune_id
    )

    if mp3_file is None:
        mp3_file = here / "data" / "coding-up-a-storm.mp3"

    if prompt_file is None:
        prompt_file = here / "data" / "sample_prompts.txt"

    # load prompts and audio
    prompts = Path(prompt_file).read_text().splitlines()
    print(f"loaded prompt file: {prompt_file}")

    mp3_bytes = Path(mp3_file).read_bytes()
    print(f"loaded mp3 file: {mp3_file}")

    total_duration = int(get_duration.remote(mp3_bytes))
    print(f"\twith duration {total_duration}s")

    n_clips = (total_duration // clip_duration) + (total_duration % clip_duration != 0)

    assert n_clips <= len(prompts), (
        "not enough prompts for song of length {total_duration}s"
    )

    # generate video clips
    prompts = prompts[:n_clips]
    print(f"generating {n_clips} clip(s) of duration {clip_duration}s")
    videos_bytes = list(
        generator.run.map(
            prompts, kwargs={"num_frames": 15 * clip_duration}, order_outputs=False
        )
    )

    # concatenate clips and overlay audio
    video = combine.remote(videos_bytes, mp3_bytes)

    # save locally
    output_dir = Path("/tmp") / finetune_id
    output_dir.mkdir(exist_ok=True, parents=True)
    mp3_name = mp3_file.stem
    output_path = output_dir / (mp3_name + ".mp4")

    output_path.write_bytes(video)
    print(f"output written to {output_path}")


@app.function(image=modal.Image.debian_slim().pip_install("mutagen"))
def get_duration(mp3: bytes) -> int:
    from io import BytesIO

    from mutagen.mp3 import MP3

    audio = MP3(BytesIO(mp3))

    return audio.info.length


@app.function(
    image=modal.Image.debian_slim().apt_install("ffmpeg").pip_install("ffmpeg-python")
)
def combine(videos: list[bytes], audio: bytes) -> bytes:
    import tempfile

    import ffmpeg

    with tempfile.TemporaryDirectory() as tmpdir:
        # write out video inputs to files
        video_paths = []
        for i, chunk in enumerate(videos):
            path = Path(tmpdir) / f"chunk{i}.mp4"
            path.write_bytes(chunk)
            video_paths.append(path)

        # concatenate video inputs together
        video_inputs = [ffmpeg.input(video_path) for video_path in video_paths]
        video_concat = ffmpeg.concat(*video_inputs, v=1, a=0).node

        # write audio to file
        audio_path = Path(tmpdir) / "audio.mp3"
        audio_path.write_bytes(audio)

        # combine audio with concatenated video
        audio_input = ffmpeg.input(str(audio_path))
        output_path = Path(tmpdir) / "output.mp4"
        output = ffmpeg.output(
            video_concat[0],
            audio_input,
            str(output_path),
            vcodec="libx264",
            acodec="aac",
            shortest=None,
        )

        # execute pipeline
        output.run()

        return output_path.read_bytes()
