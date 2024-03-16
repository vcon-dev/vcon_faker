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

from openai import OpenAI
# Load environment variables from .env file
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
S3_BUCKET = os.getenv("S3_BUCKET", "fakevcons")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

client = OpenAI()
client.api_key = OPENAI_API_KEY

# Initialize the Polly client with credentials
polly_client = boto3.client(
    'polly',
    region_name='us-west-2',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

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


st.markdown("# Virtual Conversation Faker")
# Set a slider to control the number of conversations to generate
col1, col2 = st.columns(2)
with col1:
    st.markdown("This app generates a fake conversation between a customer and an agent.  The conversation is generated based on a prompt and includes the names of the agent and customer, the business, the problem, and the emotion of the customer.  The conversation is then synthesized into an audio file using Amazon Polly. ")
with col2:
    # Generate the conversation based on the prompt trigger
    num_conversations = st.slider("Number of Conversations to Generate", 1, 20, 1)
    generate = st.button("Generate Conversation(s)")

# Display the conversation prompt

# Display the instructions in the sidebar
with st.sidebar:
    st.markdown("### Instructions")
    st.markdown("1. Edit the conversation prompt as needed.")
    st.markdown("2. Click the 'Generate Conversation' button to generate a conversation based on the prompt.")
    st.markdown("3. The conversation will be displayed and synthesized into an audio file.")
    st.markdown("4. The vCon will be uploaded to S3 and displayed locally.")

    # Reveal the fields for customizing the conversation if
    # the user wants to customize the conversation
    # if st.checkbox("Advanced: Customize Conversation"):
    #     st.markdown("### Conversation Topics, Names and Emotions")
    #     st.markdown("The following fields can be edited to customize the conversation.")
    #     agent_name = st.text_input("Agent Name", agent_name)
    #     customer_name = st.text_input("Customer Name", customer_name)
    #     business = st.selectbox("Business", businesses, index=businesses.index(business))
    #     problem = st.selectbox("Problem", problems, index=problems.index(problem))
    #     emotion = st.selectbox("Emotion", emotions, index=emotions.index(emotion))

    st.markdown("### Conversation Prompt")
    conversation_prompt = st.text_area("Conversation Prompt (Editable)", default_conversation_prompt, height=400)

if generate:
    completed_conversations = []
    for i in range(num_conversations):
        # Create a new vcon UUID
        vcon_uuid = str(uuid.uuid4())

        # Show the progress
        with st.status(f"Generating conversation: {i+1}"):
            generated_conversation = generate_conversation(
                conversation_prompt,
                agent_name=agent_name,
                customer_name=customer_name,
                business=business,
                problem=problem,
                emotion=emotion)
            st.write(generated_conversation)
            
        # Voice IDs for the agent and customer
        # Pick a random Polly voice for the agent and customer
        polly_voices = polly_client.describe_voices(Engine='standard', LanguageCode='en-US')['Voices']

        # Pick a random male voice for the agent and a random female voice for the customer
        agent_voice = random.choice([voice for voice in polly_voices if voice['Gender'] == 'Male'])
        customer_voice = random.choice([voice for voice in polly_voices if voice['Gender'] == 'Female'])
                                

        voices = {
            "Agent": agent_voice["Id"],
            "Customer": customer_voice["Id"]
        }

        with st.status("Synthesizing conversation audio..."):
            # Process each line of the conversation
            combined = AudioSegment.empty()  # Initialize an empty AudioSegment for combining the audio
            for item in generated_conversation:
                role = item['speaker']
                response = polly_client.synthesize_speech(
                    Text=item['message'],
                    OutputFormat='mp3',
                    VoiceId=voices[role]
                )

                # Save the audio stream to a temporary file
                temp_file = f"temp_{role}.mp3"
                with open(temp_file, 'wb') as file:
                    file.write(response['AudioStream'].read())

                # Load the temporary file as an AudioSegment and append it to the combined audio
                line_audio = AudioSegment.from_mp3(temp_file)
                combined += line_audio

                # Remove the temporary file
                os.remove(temp_file)


            # Export the combined audio to a single file
            combined_file = f"{vcon_uuid}.mp3"
            combined.export(combined_file, format='mp3')

            # Display the audio player
            st.audio(combined_file, format='audio/mp3')

        with st.status("Creating vCon and uploading conversation to S3..."):
            # Calculate the duration of the audio file
            audio_duration = len(combined) / 1000

            # Upload the audio file to S3 bucket
            s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
            bucket_name = S3_BUCKET

            # Create the bucket if it doesn't exist
            try:
                s3_client.create_bucket(Bucket=bucket_name)
            except s3_client.exceptions.BucketAlreadyOwnedByYou:
                pass

            file_name = combined_file
            s3_client.upload_file(file_name, bucket_name, file_name)

            # Get the URL of the uploaded file
            url = s3_client.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': file_name})

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
            vcon_file = f"{vcon_uuid}.json"
            with open(vcon_file, 'w') as file:
                file.write(vcon_json)

            # Upload the vCon JSON file to S3 bucket
            s3_client.upload_file(vcon_file, bucket_name, vcon_file)

            # Get the URL of the uploaded file
            vcon_url = s3_client.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': vcon_file})

            # Display the vCon in JSON format
            st.json(vcon_obj.to_dict())

        completed_conversations.append({
            "vcon_uuid": vcon_uuid,
            "vcon_url": vcon_url,
            "creation_time": creation_time
        })           
        # Remove the temporary files
        os.remove(combined_file)
        os.remove(vcon_file)

    # Display the completed conversations
    st.markdown("## Completed Conversations")
    for conversation in completed_conversations:
        st.markdown(f"### Conversation {conversation['vcon_uuid']}")
        st.markdown(f"**Created at:** {conversation['creation_time']}")
        st.markdown(f"**vCon URL:** [Download vCon]({conversation['vcon_url']})")
        st.markdown(f"**Audio URL:** [Download Audio]({url})")
        st.markdown("---")



