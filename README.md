# Fake Conversation Generator

This Python script generates fake conversations between a customer and an agent using OpenAI's language model and text-to-speech capabilities. The generated conversations are then packaged as vCon (Virtual Conversation) files and uploaded to an Amazon S3 bucket.

This code is available on GitHub: https://github.com/vcon-dev/vcon_faker


## Features

- Generates fake conversations based on user-defined prompts
- Synthesizes audio for each line of the conversation using OpenAI's text-to-speech model
- Creates vCon files containing conversation metadata, transcripts, and audio URLs
- Uploads vCon files and audio files to an Amazon S3 bucket
- Provides a user-friendly Streamlit interface for generating and managing conversations

## Prerequisites

Before running the script, ensure you have the following:

- Python 3.12 installed
- Required Python packages listed in the `requirements.txt` file
- OpenAI API key
- Amazon Web Services (AWS) access key and secret key
- An S3 bucket for storing the generated vCon files and audio files

## Setup

1. Clone the repository or download the script file.

2. Install the required Python packages by running the following command:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the same directory as the script and provide the following environment variables:
   ```
   AWS_KEY_ID=<your_aws_access_key>
   AWS_SECRET_KEY=<your_aws_secret_key>
   S3_BUCKET=<your_s3_bucket_name>
   OPENAI_API_KEY=<your_openai_api_key>
   OPENAI_MODEL=<openai_language_model>
   OPENAI_TTS_MODEL=<openai_text_to_speech_model>
   ```

   Replace `<your_aws_access_key>`, `<your_aws_secret_key>`, `<your_s3_bucket_name>`, and `<your_openai_api_key>` with your actual credentials. You can also specify the desired OpenAI language model and text-to-speech model.

## Usage

1. Run the script using the following command:
   ```
   streamlit run <script_name>.py
   ```

   Replace `<script_name>` with the actual name of the Python script file.

2. Access the Streamlit application in your web browser at `http://localhost:8501`.

3. Use the slider to select the number of conversations you want to generate.

4. Click the "Generate Conversation(s)" button to start generating the conversations.

5. The generated conversations will be displayed on the page, along with links to download the corresponding vCon files.

6. The conversation prompt can be edited in the sidebar to customize the generated conversations.

## Customization

- You can modify the `default_conversation_prompt` variable to change the default prompt used for generating conversations.

- The `fake_names.py` file contains lists of names, businesses, problems, and emotions used to generate random conversation details. You can update these lists to suit your needs.

- The script uses OpenAI's language model and text-to-speech model. You can change the models by updating the `OPENAI_MODEL` and `OPENAI_TTS_MODEL` environment variables in the `.env` file.

## Smart Trunk calls with local voices

`python -m vconfaker.smarttrunk` makes synthetic phone calls in Smart Trunk's call shape: a
caller phones an invented business on its Smart Trunk number and is forwarded. Each call is a
vCon (syntax `0.5.0`) with a two-channel recording that starts at answer with the recording
notice, the notice evidence as tags, a transcript per channel, and, on numbers that screen
calls, a screening outcome. Every vCon carries the three synthetic markers described in
[CONSENT.md](CONSENT.md).

Four packs in `vconfaker/smarttrunk/packs/`: car dealerships, medical and dental clinics,
municipal offices, and home services. Each names invented businesses (or towns, with their own
streets), their lines (555-0100 to 555-0149; callers use 555-0150 to 555-0199), the scenarios
and their mix. Every pack has calls about a recording (a copy, or a deletion), hang-ups,
wrong numbers, a screened robocall, after hours and voicemail. Calls vary in how they open and
how long they run (many are 15 to 45 s), and some go through a phone menu, are transferred to a
colleague, or have interruptions.

The stages, in one process:

1. **Script.** Claude (`claude-sonnet-5-5`, structured output) writes each call from a brief. A
   script is thrown away and written again if it names a person, business or place it was not
   given, a real company from the industry's block list (`REAL_NAMES` in `hygiene.py`), an
   email or web address, or a number outside the fictional range, or if it books something for
   the moment of the call.
2. **Voices.** Kokoro-82M through mlx-audio, on the machine, one voice per speaker.
3. **Phone audio.** ffmpeg: the notice at answer, any phone menu and key tone, hold music on
   holds, telephone band, light line noise, G.711 at 8 kHz, a stereo MP3 with the caller on the
   left channel and the business side on the right.
4. **Check.** Kokoro is released, then parakeet-mlx transcribes each channel's speech. A call
   fails if its word error rate is over 20%, if any line of three words or more was not heard,
   or if a number outside the fictional range comes out; it is voiced again with other voices.
   The recognizer's transcript, not the script, goes into the vCon.
5. **vCon.** Built with vcon-lib and validated with vcon-lib, the core-05 JSON schema in
   `tests/` and the governance check.

Set up a venv from `requirements-smarttrunk.txt` (Apple silicon), install `parakeet-mlx` and
ffmpeg, then from the repo root:

```
python -m vconfaker.smarttrunk --industry clinic --count 50 --out ~/synthetic-calls/v1/clinic \
    --corpus-root ~/synthetic-calls/v1 --usage-file ~/synthetic-calls/v1/api_usage.json --max-cost-usd 5
```

The output folder holds `vcons/`, `audio/`, `scripts/`, `transcripts/`, `manifest.json` (timings,
memory, token use and estimated cost) and an `index.html` to listen to each call;
`--corpus-root` also writes an overall `index.html` over every industry folder. Keep it outside
the repo. The run pauses while free memory is under `--min-free-mb` (2048), and stops writing
scripts once the shared usage file reaches `--max-cost-usd`. The MP3 URLs in the vCons use
`--media-base-url`; whatever publishes the audio sets the real one, and the `content_hash` lets
it check the file. Offline tests: `pytest tests/test_smarttrunk.py`.

## License

This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgements

- [OpenAI](https://openai.com/) for providing the language model and text-to-speech capabilities.
- [Streamlit](https://streamlit.io/) for the user-friendly web interface.
- [Amazon Web Services](https://aws.amazon.com/) for the S3 storage service.

Feel free to contribute to this project by submitting pull requests or reporting issues on the GitHub repository.