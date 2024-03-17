import boto3
from pydub import AudioSegment
from dotenv import load_dotenv
import os
import json
import random
import streamlit as st
import vcon
from datetime import datetime
import uuid
from fake_names import male_names, female_names, last_names, businesses, problems, emotions
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


from openai import OpenAI

# Get the environment variables from secrets.toml

AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = st.secrets["AWS_SECRET_KEY"]
S3_BUCKET = st.secrets["S3_BUCKET"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_MODEL = st.secrets["OPENAI_MODEL"]
OPENAI_TTS_MODEL = st.secrets["OPENAI_TTS_MODEL"]

print(f"Using model: {OPENAI_MODEL}, TTS model: {OPENAI_TTS_MODEL} and S3 bucket: {S3_BUCKET}")

client = OpenAI()
client.api_key = OPENAI_API_KEY


def generate_conversation(prompt, agent_name, customer_name, business, problem, emotion):
    prompt += f"\n\nIn this conversation, the agent's name is {agent_name} and the customer's name is {customer_name}.  "
    prompt += f"The conversation is about a problem related to a {business} .  "
    prompt += f"The problem is related to {problem}.  The customer is feeling {emotion}."

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt}
        ]
        )
    result = json.loads(completion.choices[0].message.content)
    return result.get("conversation", [])

default_conversation_prompt = """
Generate a fake conversation between a customer and an agent.  
The agent should introduce themselves, their company and give the customer their name. The agent should ask for the customer's name.
As part of the conversation, have the agent ask for two pieces of personal information.  
Spell out numbers. For example, 1000 should be said as one zero zero zero, not one thousand.
The conversation should be at least 10 lines long and be complete. At the end
of the conversation, the agent should thank the customer for their time and end the conversation.
Return the conversation formatted like the following example:

{'conversation': 
    [
    {'speaker': 'Agent', 'message': 'xxxxx'}, 
    {'speaker': 'Customer', 'message': "xxxxx."}, 
    {'speaker': 'Agent', 'message': "xxxxxx"}
    ] 
}
"""


# Set a slider to control the number of conversations to generate
# Generate the conversation based on the prompt trigger

col1, col2 = st.columns(2)
col1.title("Fake Conversation Generator")
col2.markdown("This app generates fake conversations between a customer and an agent.  The conversation is generated based on a prompt and includes the names of the agent and customer, the business, the problem, and the emotion of the customer.  The conversation is then synthesized into an audio file, a vCon is created then it is uploaded into S3.") 

num_conversations = col2.number_input("Number of Conversations to Generate", 1, 20, 1)
generate = col2.button("Generate Conversation(s)")
st.toast(f"Configured to use model: {OPENAI_MODEL}, TTS model: {OPENAI_TTS_MODEL} and S3 bucket: {S3_BUCKET}")

# Display the instructions in the sidebar
with st.sidebar:
    instructions = f"""    
    
    ## Instructions

    1. Use the slider to select the number of conversations to generate.
    2. Click the "Generate Conversation(s)" button to generate the conversations.
    3. The conversations will be generated and displayed below.
    4. Each conversation will include a link to download the vCon file.

    ## Conversation Prompt

    The conversation prompt is passed to the LLM to generate the conversation.  
    The conversation will be generated based on the prompt and the names of the agent and customer, 
    the business, the problem, and the emotion of the customer (all picked at random).

    This prompt can be edited to generate different conversations.

    """
    st.markdown(instructions)
    conversation_prompt = st.text_area("Conversation Prompt (Editable)", default_conversation_prompt, height=400)

if generate:
    completed_conversations = []
    progress_text = "Generating fake conversations. Please wait."
    total_bar = st.progress(0, text=progress_text)

    for i in range(num_conversations):
        # Generate the conversations and pick random names for the agent and customer
        agent_name = random.choice(male_names) + " " + random.choice(last_names)
        customer_name = random.choice(female_names) + " " + random.choice(last_names)
        business = random.choice(businesses)
        problem = random.choice(problems)
        emotion = random.choice(emotions)

        # create a random fake phone number for the agent, and one for the customer
        agent_phone = f"+1{random.randint(1000000000, 9999999999)}"
        customer_phone = f"+1{random.randint(1000000000, 9999999999)}"
        # Create a random fake email for the agent
        agent_email = f"{agent_name.replace(' ', '.').lower()}@{business.replace(' ', '').lower()}.com"
        # Create a random fake email for the customer
        customer_email = f"{customer_name.replace(' ', '.').lower()}@gmail.com"

        # Create a new vcon UUID
        vcon_uuid = str(uuid.uuid4())


        total_bar.progress((i+1)/num_conversations, text=f"Generating conversation {i+1} of {num_conversations}")
        this_bar = st.progress(0, text="Generating conversation transcript.")
        generated_conversation = generate_conversation(
            conversation_prompt,
            agent_name=agent_name,
            customer_name=customer_name,
            business=business,
            problem=problem,
            emotion=emotion)
        
        print(generated_conversation)
        this_bar.progress(0.2, text="Synthesizing conversation audio")
        # Process each line of the conversation
        audio_files = []  # Initialize an empty list to store audio files
        voices = ["alloy", "echo", "fable", "onyx", "nova",  "shimmer"]
        
        # Pick a different random voice for each speaker
        agent_voice = random.choice(voices)
        # Remove the agent voice from the list so we don't pick it again
        voices.remove(agent_voice)
        customer_voice = random.choice(voices)
        voices = {
            "Agent": agent_voice,
            "Customer": customer_voice
        }

        combined_audio = AudioSegment.silent(duration=0)
        for item in generated_conversation:
            role = item['speaker']
            speech_file_path = "_temp.mp3"
            response = client.audio.speech.create(
                input=item['message'],
                model=OPENAI_TTS_MODEL,
                response_format="mp3",
                voice=voices[role]
            )
            response.stream_to_file(speech_file_path)
            # Append the audio file to the list
            audio = AudioSegment.from_file(speech_file_path)
            combined_audio += audio
            # Remove the temporary audio file
            os.remove(speech_file_path)

        # Export the combined audio to a file
        combined_file = f"{vcon_uuid}.mp3"
        combined_audio.export(combined_file)

        this_bar.progress(0.4, text="Creating vCon and uploading conversations")

        # Calculate the duration of the audio file
        audio_duration = len(combined_audio) / 1000

        # Upload the audio file to S3 bucket
        s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
        bucket_name = S3_BUCKET

        # Create the bucket if it doesn't exist
        try:
            s3_client.create_bucket(Bucket=bucket_name)
        except s3_client.exceptions.BucketAlreadyOwnedByYou:
            pass

        # Upload the audio file to the S3 bucket, use the vcon_uuid as the file name
        # and put it in a path based on year, month and day
        year, month, day = datetime.now().isoformat().split("T")[0].split("-")

        # Make sure that the directory exists
        s3_client.put_object(Bucket=bucket_name, Key=f"{year}/")
        s3_client.put_object(Bucket=bucket_name, Key=f"{year}/{month}/")
        s3_client.put_object(Bucket=bucket_name, Key=f"{year}/{month}/{day}/")

        file_name = combined_file
        s3_path = f"{year}/{month}/{day}/{file_name}"
        this_bar.progress(0.6, text="uploading audio file")
        s3_client.upload_file(file_name, bucket_name, s3_path)

        # Get the URL of the uploaded file
        url = s3_client.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': s3_path})

        # Now create the vCon from this conversation
        # Create a vCon object from the generated conversation
        agent_party = {
                "tel": agent_phone,
                "meta": {
                    "role": "agent"
                },
                "name": agent_name,
                "mailto": agent_email
            }
        customer_party = {
                "tel": customer_phone,
                "meta": {
                    "role": "customer"
                },
                "name": customer_name,
                "email": customer_email
            }
        dialog_info = {
            "alg": "SHA-512",
            "url": url,
            "meta": {
                "direction": "in",
                "disposition": "ANSWERED"
            },
            "type": "recording",
            "start": datetime.now().isoformat(),
            "parties": [
                1,
                0
            ],
            "duration": audio_duration,
            "filename": file_name,
            "mimetype": "audio/mp3"
        }

        # Save the generation information to an attachment
        generation_info = {
            "type":"generation_info",
            "body": {
                "agent_name": agent_name,
                "customer_name": customer_name,
                "business": business,
                "problem": problem,
                "emotion": emotion,
                "prompt": conversation_prompt,
                "created_on": datetime.now().isoformat(),
                "model": OPENAI_MODEL
            }
        }

        # Save the transcript of the conversation as an analysis
        analysis_info = {
            "type": "transcript",
            "dialog": 0,
            "vendor": 'openai',
            "encoding": "json",
            "body": generated_conversation,
            "vendor_schema": {
                "model": OPENAI_MODEL,
                "prompt": conversation_prompt
            }
        }

        
        creation_time = datetime.now().isoformat()
        # Create the vCon object
        vcon_obj = vcon.Vcon(
            parties=[agent_party, customer_party],
            dialog=[dialog_info],
            attachments=[generation_info],
            analysis=[analysis_info],
            uuid=vcon_uuid,
            created_at=creation_time,
            updated_at=creation_time
        )
        vcon_obj.sign_dialogs()

        # Save the vCon object to a JSON file
        vcon_json = vcon_obj.to_json()
        vcon_file = f"{vcon_uuid}.vcon.json"
        with open(vcon_file, 'w') as file:
            file.write(vcon_json)

        # Upload the vCon JSON file to S3 bucket
        s3_path = f"{year}/{month}/{day}/{vcon_file}"

        # Upload the vCon file
        this_bar.progress(0.8, text="uploading vcon")
        s3_client.upload_file(vcon_file, bucket_name, s3_path)

        # Get the URL of the uploaded file
        vcon_url = s3_client.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': s3_path})


        completed_conversations.append({
            "vcon_uuid": vcon_uuid,
            "vcon_url": vcon_url,
            "creation_time": creation_time,
            "summary": f"Conversation between {agent_name} and {customer_name} about a {business} problem related to {problem}.  The customer is feeling {emotion}. "
        })           
        # Remove the temporary files
        os.remove(combined_file)
        os.remove(vcon_file)
        this_bar.empty()

    # Display the completed conversations
    total_bar.empty()
    st.markdown("## Completed Conversations")
    for conversation in completed_conversations:
        st.markdown(f"**Created at:** {conversation['creation_time']}")
        st.markdown(conversation["summary"])
        st.markdown(f"**vCon URL:** [Download vCon]({conversation['vcon_url']})")
        st.markdown("---")



