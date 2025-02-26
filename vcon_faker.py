# Standard library imports
import base64
import hashlib
import json
import logging
import os
import random
import uuid
import warnings
from datetime import datetime

# Third-party imports
import boto3
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
from vcon import Vcon
from vcon.party import Party
from vcon.dialog import Dialog

# Local imports
from fake_names import (
    male_names,
    female_names,
    last_names,
    businesses,
    problems,
    emotions,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

# Configure warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv()

# Get environment variables from secrets.toml
AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = st.secrets["AWS_SECRET_KEY"]
S3_BUCKET = st.secrets["S3_BUCKET"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_MODEL = st.secrets["OPENAI_MODEL"]
OPENAI_TTS_MODEL = st.secrets["OPENAI_TTS_MODEL"]

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize S3 client
s3_client = boto3.client(
    "s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY
)


def ensure_s3_bucket_exists(bucket_name):
    """Ensure the S3 bucket exists, create if it doesn't."""
    try:
        s3_client.create_bucket(Bucket=bucket_name)
        logger.info(f"Created new S3 bucket: {bucket_name}")
    except s3_client.exceptions.BucketAlreadyOwnedByYou:
        logger.debug(f"Using existing S3 bucket: {bucket_name}")


def upload_to_s3(file_path, bucket, s3_path):
    """Upload a file to S3 and return its URL."""
    s3_client.upload_file(file_path, bucket, s3_path)
    return s3_client.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": s3_path}
    )


def get_s3_path(filename):
    """Generate S3 path based on current date."""
    year, month, day = datetime.now().isoformat().split("T")[0].split("-")
    return f"{year}/{month}/{day}/{filename}"


def create_vcon_object(
    agent_name,
    customer_name,
    agent_phone,
    customer_phone,
    agent_email,
    customer_email,
    url,
    filename,
    signature,
    audio_duration,
    business_name,
    business,
    problem,
    emotion,
    generation_prompt,
    conversation,
    generate_audio=False
):
    """Create and return a vCon object with all components.
    
    Args:
        agent_name (str): Name of the agent
        customer_name (str): Name of the customer
        agent_phone (str): Phone number of the agent
        customer_phone (str): Phone number of the customer
        agent_email (str): Email of the agent
        customer_email (str): Email of the customer
        url (str): URL of the conversation
        filename (str): Filename of the conversation
        signature (str): Signature of the conversation
        audio_duration (float): Duration of the conversation in seconds
        business_name (str): Name of the business
        business (str): Type of business
        problem (str): The problem/situation being discussed
        emotion (str): Customer's emotional state
        generation_prompt (str): The base prompt template for conversation generation
        conversation (list): List of conversation turns with speaker and message
        generate_audio (bool): Whether to generate audio files. Defaults to False.

    Returns:
        Vcon: The created vCon object
    """

    # Ensure all strings are properly escaped for JSON
    def sanitize_for_json(text):
        if not isinstance(text, str):
            return text
        # Replace any problematic characters
        return text.replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")

    # Create sanitized copies of all string inputs
    safe_agent_name = sanitize_for_json(agent_name)
    safe_customer_name = sanitize_for_json(customer_name)
    safe_business_name = sanitize_for_json(business_name)
    safe_business = sanitize_for_json(business)
    safe_problem = sanitize_for_json(problem)
    safe_emotion = sanitize_for_json(emotion) if emotion else None
    safe_prompt = sanitize_for_json(generation_prompt)

    # Create customer ID
    customer_id = f"{customer_phone}_{customer_email}_1100"

    # Create parties with IDs
    agent_party = Party(
        id=agent_email,
        name=safe_agent_name,
        tel=agent_phone,
        mailto=agent_email,
        role="agent",
        meta={
            "role": "agent",
            "extension": "2212",
            "cxm_user_id": "891"
        }
    )

    customer_party = Party(
        id=customer_id,
        name=safe_customer_name,
        tel=customer_phone,
        mailto=customer_email,
        role="customer",
        meta={
            "role": "customer"
        }
    )

    # Create dialog with updated metadata
    dialog_info = Dialog(
        type="recording" if generate_audio else "text",
        start=datetime.now().isoformat(),
        parties=[1, 0],  # Agent is 1, Customer is 0
        url=url if generate_audio else None,
        filename=filename if generate_audio else None,
        mimetype="audio/x-wav" if generate_audio else "text/plain",
        alg="SHA-512" if generate_audio else None,
        signature=signature if generate_audio else None,
        duration=audio_duration if generate_audio else None,
        meta={
            "disposition": "ANSWERED",
            "direction": "out",
            "agent_selected_disposition": "VM Left",
            "is_dealer_manually_set": False,
            "engaged": False
        }
    )

    # Create vCon object
    vcon = Vcon.build_new()
    vcon.add_party(customer_party)  # Add customer first (index 0)
    vcon.add_party(agent_party)     # Add agent second (index 1)
    vcon.add_dialog(dialog_info)

    # Build transcript from conversation
    transcript_text = ""
    for turn in conversation:
        if isinstance(turn, dict):
            message = sanitize_for_json(turn.get("message", ""))
            transcript_text += message + "\n\n"

    # Add transcript analysis
    transcript_info = {
        "type": "transcript",
        "dialog": 0,
        "vendor": "deepgram" if generate_audio else "text",
        "body": {
            "transcript": transcript_text.strip(),
            "confidence": 0.99,
            "detected_language": "en"
        },
        "encoding": "none"
    }

    # Add summary analysis
    summary_info = {
        "type": "summary",
        "dialog": 0,
        "vendor": "openai",
        "body": f"In this conversation, {safe_agent_name} from {safe_business_name} discusses {safe_problem} with {safe_customer_name}. The agent provides assistance and information about {safe_business}.",
        "encoding": "none"
    }


    # Add diarized analysis
    diarized_info = {
        "type": "diarized",
        "dialog": 0,
        "vendor": "openai",
        "body": transcript_text.strip(),
        "encoding": "none"
    }

    # Add all analyses
    vcon.add_analysis(**transcript_info)
    vcon.add_analysis(**summary_info)
    vcon.add_analysis(**diarized_info)

    # Add attachments
    vcon.add_attachment(
        type="bria_call_ended",
        body={
            "email": agent_email,
            "extension": "2212",
            "isDealerManuallySet": False,
            "dealerId": 1100,
            "dealerName": safe_business_name,
            "agentName": safe_agent_name,
            "agentSelectedDisposition": "VM Left",
            "customerNumber": customer_phone,
            "direction": "out",
            "duration": audio_duration if generate_audio else 0,
            "state": "ANSWERED",
            "received_at": datetime.now().isoformat()
        },
        encoding="none"
    )


    # Validate the entire vCon object
    is_valid, errors = vcon.is_valid()
    if not is_valid:
        logger.error(f"vCon validation failed: {errors}")
        raise ValueError(f"Invalid vCon object: {errors}")

    return vcon


def process_conversation(
    business,
    business_name,
    problem,
    emotion,
    generation_prompt,
    progress_bar,
    generate_audio=False
):
    """Process a single conversation and return its details.
    
    Args:
        business (str): Type of business
        business_name (str): Name of the business
        problem (str): The problem/situation being discussed
        emotion (str): Customer's emotional state
        generation_prompt (str): The base prompt template for conversation generation
        progress_bar: Streamlit progress bar object
        generate_audio (bool): Whether to generate audio files. Defaults to False.
    """
    try:
        # Generate random identities
        agent_name = f"{random.choice(male_names)} {random.choice(last_names)}"
        customer_name = f"{random.choice(female_names)} {random.choice(last_names)}"
        agent_phone = f"+1{random.randint(1000000000, 9999999999)}"
        customer_phone = f"+1{random.randint(1000000000, 9999999999)}"
        agent_email = (
            f"{agent_name.replace(' ', '.').lower()}"
            f"@{business.replace(' ', '').lower()}.com"
        )
        customer_email = f"{customer_name.replace(' ', '.').lower()}" "@gmail.com"

        # Generate conversation
        conversation = generate_conversation(
            generation_prompt,
            agent_name,
            customer_name,
            business,
            problem,
            emotion,
            business_name,
        )

        if not conversation:
            raise ValueError("Failed to generate conversation")

        vcon_uuid = str(uuid.uuid4())
        audio_url = None
        audio_signature = None
        audio_duration = 0
        combined_file = None

        if generate_audio:
            progress_bar.progress(0.4, text="Generating audio...")
            # Generate audio
            combined_file = f"{vcon_uuid}.mp3"
            combined_audio = AudioSegment.silent(duration=0)

            voices = {
                "Agent": random.choice(["alloy", "echo", "fable"]),
                "Customer": random.choice(["onyx", "nova", "shimmer"]),
            }

            for item in conversation:
                if not isinstance(item, dict) or "message" not in item:
                    continue

                speech_file = "_temp.mp3"
                response = client.audio.speech.create(
                    model=OPENAI_TTS_MODEL,
                    voice=voices[item["speaker"]],
                    input=item["message"],
                    response_format="mp3",
                )

                response.stream_to_file(speech_file)
                audio_segment = AudioSegment.from_file(speech_file)
                combined_audio += audio_segment
                os.remove(speech_file)

            # Save combined audio
            combined_audio.export(combined_file)
            audio_duration = len(combined_audio) / 1000

            # Calculate audio signature
            with open(combined_file, "rb") as f:
                content = f.read()
                audio_signature = base64.urlsafe_b64encode(
                    hashlib.sha512(content).digest()
                ).decode("utf-8")

            # Upload to S3
            progress_bar.progress(0.6, text="Uploading files...")
            ensure_s3_bucket_exists(S3_BUCKET)

            s3_audio_path = get_s3_path(combined_file)
            audio_url = upload_to_s3(combined_file, S3_BUCKET, s3_audio_path)

        # Create and save vCon
        vcon = create_vcon_object(
            agent_name,
            customer_name,
            agent_phone,
            customer_phone,
            agent_email,
            customer_email,
            audio_url,
            combined_file,
            audio_signature,
            audio_duration,
            business_name,
            business,
            problem,
            emotion,
            generation_prompt,
            conversation,
            generate_audio=generate_audio
        )

        vcon_file = f"{vcon_uuid}.vcon.json"
        with open(vcon_file, "w") as f:
            f.write(vcon.to_json())

        # Upload vCon to S3
        s3_vcon_path = get_s3_path(vcon_file)
        vcon_url = upload_to_s3(vcon_file, S3_BUCKET, s3_vcon_path)

        # Cleanup temporary files
        if combined_file and os.path.exists(combined_file):
            os.remove(combined_file)
        os.remove(vcon_file)

        return {
            "vcon_uuid": vcon_uuid,
            "vcon_url": vcon_url,
            "creation_time": datetime.now().isoformat(),
            "summary": (
                f"Conversation between {agent_name} and {customer_name} "
                f"about {business_name}, a {business}, related to {problem}. "
                f"{'The customer is ' + emotion if emotion else ''}"
            ),
        }
    except Exception as e:
        logger.error(f"Error processing conversation: {str(e)}")
        raise


def generate_conversation(
    prompt, agent_name, customer_name, business, problem, emotion, business_name
):
    """Generate a conversation between an agent and customer using OpenAI.

    Args:
        prompt (str): The base prompt template for conversation generation
        agent_name (str): Name of the agent
        customer_name (str): Name of the customer
        business (str): Type of business
        problem (str): The problem/situation being discussed
        emotion (str): Customer's emotional state
        business_name (str): Name of the business

    Returns:
        list: List of conversation turns with speaker and message
    """
    logger.info(
        f"Generating conversation for {agent_name} and {customer_name} "
        f"about {business_name} ({business})"
    )
    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant designed to output JSON.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    result = json.loads(completion.choices[0].message.content)
    logger.info(
        f"Generated conversation with {len(result.get('conversation', []))} turns"
    )
    return result.get("conversation", [])


default_conversation_prompt = """
Generate a fake conversation between a customer and an agent.
The agent should introduce themselves, their company and give the customer
their name. The agent should ask for the customer's name.
As part of the conversation, have the agent ask for two pieces of
personal information.  Spell out numbers. For example, 1000 should be
said as one zero zero zero, not one thousand. The conversation should be
at least 10 lines long and be complete. At the end
of the conversation, the agent should thank the customer for their time
and end the conversation. Return the conversation formatted 
like the following example:

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
st.title("Fake Conversation Generator")

col1, col2 = st.columns(2)

# select business from a dropdown
business = col2.selectbox("Select Business", businesses)
problem = col2.selectbox("Select Problem", problems)
business_name = col2.text_input("Business Name", "a random business")
col1.markdown(
    "This app generates fake conversations between a customer and \
            an agent. The conversation is generated based on a prompt and \
            includes the names of the agent and customer, the business, \
            the problem, and the emotion of the customer.  The conversation \
            is then synthesized into an audio file, a vCon is created then \
            it is uploaded into S3."
)

add_emotion = col2.checkbox("Add emotion to conversation.")
generate_audio = col2.checkbox("Generate audio files", value=False)
num_conversations = col2.number_input("Number of Conversations to Generate", 1, 100, 1)
generate = col2.button("Generate Conversation(s)")
st.toast(
    f"Configured to use model: {OPENAI_MODEL}, TTS model: {OPENAI_TTS_MODEL} and S3 bucket: {S3_BUCKET}"
)

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
    conversation_prompt = st.text_area(
        "Conversation Prompt (Editable)", default_conversation_prompt, height=400
    )

if generate:
    completed_conversations = []
    progress_text = "Generating fake conversations. Please wait."
    total_bar = st.progress(0, text=progress_text)

    for i in range(num_conversations):
        logger.info(f"Generating conversation {i+1} of {num_conversations}")
        this_bar = st.progress(0, text="Processing conversation...")

        # Select random business and problem if needed
        current_business = (
            random.choice(businesses)
            if business == "Pick Random Business Type"
            else business
        )
        current_problem = (
            random.choice(problems) if problem == "random situation" else problem
        )
        current_emotion = random.choice(emotions) if add_emotion else None

        # Build generation prompt
        current_prompt = (
            f"{conversation_prompt}\n\n"
            f"The conversation is about {business_name} "
            f"(a {current_business}) and is about {current_problem}. "
            f"{'The customer is feeling ' + current_emotion + '.' if current_emotion else ''}"
        )

        try:
            conversation_details = process_conversation(
                current_business,
                business_name,
                current_problem,
                current_emotion,
                current_prompt,
                this_bar,
                generate_audio=generate_audio
            )
            completed_conversations.append(conversation_details)
        except Exception as e:
            logger.error(f"Error processing conversation: {str(e)}")
            st.error(f"Failed to generate conversation {i+1}: {str(e)}")

        total_bar.progress((i + 1) / num_conversations)
        this_bar.empty()

    total_bar.empty()

    # Display results
    st.markdown("## Completed Conversations")
    for conv in completed_conversations:
        st.markdown(f"**Created at:** {conv['creation_time']}")
        st.markdown(conv["summary"])
        st.markdown(f"**vCon URL:** [Download vCon]({conv['vcon_url']})")
        st.markdown("---")
