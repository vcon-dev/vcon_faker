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

## License

This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgements

- [OpenAI](https://openai.com/) for providing the language model and text-to-speech capabilities.
- [Streamlit](https://streamlit.io/) for the user-friendly web interface.
- [Amazon Web Services](https://aws.amazon.com/) for the S3 storage service.

Feel free to contribute to this project by submitting pull requests or reporting issues on the GitHub repository.