import base64
import hashlib
import json
import logging
import os
import random
import uuid
from datetime import datetime, timedelta, time as dt_time

# Third-party imports
import boto3
import streamlit as st
from openai import OpenAI
from pydub import AudioSegment
from vcon import Vcon
from vcon.party import Party
from vcon.dialog import Dialog
import pytz

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

# Get environment variables from secrets.toml
AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = st.secrets["AWS_SECRET_KEY"]
DEFAULT_S3_BUCKET = st.secrets["S3_BUCKET"]  # Changed to DEFAULT_S3_BUCKET
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_MODEL = st.secrets["OPENAI_MODEL"]
OPENAI_TTS_MODEL = st.secrets["OPENAI_TTS_MODEL"]

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
s3_client = boto3.client(
    "s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY
)


@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_available_openai_models():
    """
    Fetch available OpenAI models or return default list if API call fails.
    Cache results to avoid repeated API calls.
    
    Returns:
        tuple: Two lists containing chat models and TTS models
    """
    try:
        # Attempt to get models from the OpenAI API
        models = client.models.list()
        model_ids = [model.id for model in models.data]
        
        # Separate models into categories
        chat_models = model_ids.copy()
        tts_models = [m for m in model_ids if "tts" in m.lower()]
        
        # Sort models by name
        chat_models.sort()
        tts_models.sort()
        
        logger.info(f"Retrieved {len(chat_models)} models and {len(tts_models)} TTS models from OpenAI API")
        return chat_models, tts_models
    except Exception as e:
        logger.warning(f"Failed to fetch models from OpenAI API: {e}")
        # Default model lists if API call fails
        default_chat_models = ["o3-mini"]
        default_tts_models = ["tts-1", "tts-1-hd"]
        return default_chat_models, default_tts_models


@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour with no spinner
def ensure_s3_bucket_exists(bucket_name, _s3_client):
    """
    Ensure the S3 bucket exists, create if it doesn't.
    Results are cached to minimize AWS API calls.
    
    Args:
        bucket_name (str): Name of the S3 bucket
        s3_client: Boto3 S3 client
        
    Returns:
        bool: True if bucket exists or was created successfully
    """
    try:
        # First check if bucket exists
        s3_client.head_bucket(Bucket=bucket_name)
        logger.debug(f"Using existing S3 bucket: {bucket_name}")
        return True
    except s3_client.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        if error_code == '404' or error_code == 'NoSuchBucket':
            # Bucket doesn't exist, try to create it
            try:
                # For buckets in us-east-1, don't specify LocationConstraint
                region = boto3.session.Session().region_name
                if region == 'us-east-1':
                    s3_client.create_bucket(Bucket=bucket_name)
                else:
                    s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
                logger.info(f"Created new S3 bucket: {bucket_name}")
                return True
            except Exception as create_error:
                logger.error(f"Failed to create S3 bucket {bucket_name}: {create_error}")
                return False
        else:
            # Some other error occurred
            logger.error(f"Error accessing S3 bucket {bucket_name}: {e}")
            return False


def upload_to_s3(file_path, bucket, s3_path, s3_client):
    """
    Upload a file to S3 and return its URL.
    
    Args:
        file_path (str): Local path to the file
        bucket (str): S3 bucket name
        s3_path (str): Path within the S3 bucket
        s3_client: Boto3 S3 client
        
    Returns:
        str: Presigned URL for the uploaded file
    """
    s3_client.upload_file(file_path, bucket, s3_path)
    return s3_client.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": s3_path}
    )


def get_s3_path(filename, conversation_date=None):
    """
    Generate S3 path based on specified date or current date.
    
    Args:
        filename (str): Filename to include in the path
        conversation_date (datetime, optional): Date to use for path structure
        
    Returns:
        str: S3 path with year/month/day/filename structure
    """
    if conversation_date is None:
        conversation_date = datetime.now()
        
    year, month, day = conversation_date.strftime("%Y-%m-%d").split("-")
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
    generate_audio=False,
    conversation_type="voice",
    conversation_date=None,
    timezone="US/Eastern",
    business_hours=None
):
    """
    Create and return a vCon object with all components.
    
    Args:
        agent_name (str): Name of the agent
        customer_name (str): Name of the customer
        agent_phone (str): Phone number of the agent
        customer_phone (str): Phone number of the customer
        agent_email (str): Email of the agent
        customer_email (str): Email of the customer
        url (str): URL of the conversation audio
        filename (str): Filename of the conversation audio
        signature (str): Signature of the conversation audio
        audio_duration (float): Duration of the conversation in seconds
        business_name (str): Name of the business
        business (str): Type of business
        problem (str): The problem/situation being discussed
        emotion (str): Customer's emotional state
        generation_prompt (str): The base prompt template used for conversation generation
        conversation (list): List of conversation turns with speaker and message
        generate_audio (bool): Whether audio files were generated
        conversation_type (str): Type of conversation - "voice" or "messaging"
        conversation_date (datetime, optional): Date and time of the conversation
        timezone (str): Timezone for the conversation
        business_hours (dict, optional): Start and end hours for business

    Returns:
        Vcon: The created vCon object
    """
    # Set default business hours if not provided
    if business_hours is None:
        business_hours = {"start": 9, "end": 17}  # 9 AM to 5 PM
        
    # Set conversation date if not provided
    if conversation_date is None:
        conversation_date = datetime.now()
    
    # Ensure conversation_date is timezone-aware
    tz = pytz.timezone(timezone)
    if conversation_date.tzinfo is None:
        conversation_date = tz.localize(conversation_date)
    
    # Track the last dialog timestamp for setting created_at later
    last_dialog_time = conversation_date
    
    # Ensure all strings are properly escaped for JSON
    def sanitize_for_json(text):
        if not isinstance(text, str):
            return text
        return text.replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")

    # Create sanitized copies of all string inputs
    safe_agent_name = sanitize_for_json(agent_name)
    safe_customer_name = sanitize_for_json(customer_name)
    safe_business_name = sanitize_for_json(business_name)
    safe_business = sanitize_for_json(business)
    safe_problem = sanitize_for_json(problem)
    safe_emotion = sanitize_for_json(emotion) if emotion else None

    # Create vCon object
    vcon = Vcon.build_new()
    
    try:   
        # Rewrite the vcon object to include the created_at field
        vcon_dict = vcon.to_dict()
        vcon_dict['created_at'] = last_dialog_time.isoformat()
        vcon = Vcon(vcon_dict)
    except Exception as e:
        logger.error(f"Error rewriting vcon object: {e}")
        raise

    # Different approaches based on conversation type
    if conversation_type == "messaging":
        # For messaging, create a simplified structure
        agent_party = Party(
            id=agent_email,
            name=safe_agent_name,
            tel="",
            mailto=agent_email,
            role="agent"
        )

        customer_party = Party(
            id=customer_phone,
            name=customer_phone,
            tel=customer_phone,
            mailto=customer_email,
            role="contact"
        )

        # Add parties
        vcon.add_party(agent_party)     # Agent first (index 0)
        vcon.add_party(customer_party)  # Customer second (index 1)
        
        # Generate message timestamps
        base_time = conversation_date
        
        # Add dialog entries for each message
        for i, turn in enumerate(conversation):
            if not isinstance(turn, dict) or "message" not in turn:
                continue
                
            # Determine the party index based on speaker
            party_index = 1 if turn["speaker"] == "Customer" else 0
            
            # Add time between messages (2-5 minutes)
            if i > 0:
                base_time += timedelta(minutes=random.randint(2, 5), seconds=random.randint(0, 59))
            
            # Create Dialog object
            dialog_entry = Dialog(
                type="text",
                start=base_time.isoformat(),
                parties=[party_index],
                originator=party_index,
                mimetype="text/plain",
                body=turn["message"]
            )
            
            vcon.add_dialog(dialog_entry)
            last_dialog_time = base_time  # Update the last dialog time
        
    else:
        # Traditional voice conversation approach
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

        # Add parties
        vcon.add_party(customer_party)  # Add customer first (index 0)
        vcon.add_party(agent_party)     # Add agent second (index 1)

        # Create dialog with updated metadata
        dialog_info = Dialog(
            type="recording" if generate_audio else "text",
            start=conversation_date.isoformat(),
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
        
        # Add dialog
        vcon.add_dialog(dialog_info)
        last_dialog_time = conversation_date  # For voice, use the conversation start time

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

 
    # Set the vCon created_at to 15 seconds after the last dialog
    vcon_created_time = last_dialog_time + timedelta(seconds=15)
    
    vcon_dict = vcon.to_dict()
    vcon_dict['created_at'] = vcon_created_time.isoformat()
    vcon = Vcon(vcon_dict)

    # Validate the entire vCon object
    is_valid, errors = vcon.is_valid()
    if not is_valid:
        logger.error(f"vCon validation failed: {errors}")
        raise ValueError(f"Invalid vCon object: {errors}")

    return vcon


def generate_conversation(
    prompt, agent_name, customer_name, business, problem, emotion, business_name, model=OPENAI_MODEL
):
    """
    Generate a conversation between an agent and customer using OpenAI.

    Args:
        prompt (str): The base prompt template for conversation generation
        agent_name (str): Name of the agent
        customer_name (str): Name of the customer
        business (str): Type of business
        problem (str): The problem/situation being discussed
        emotion (str): Customer's emotional state
        business_name (str): Name of the business
        model (str): OpenAI model to use for generation

    Returns:
        list: List of conversation turns with speaker and message
    """
    logger.info(
        f"Generating conversation for {agent_name} and {customer_name} "
        f"about {business_name} ({business}) using model {model}"
    )
    
    # Update the prompt to include specific agent and customer names
    enhanced_prompt = (
        f"{prompt}\n\n"
        f"The conversation is about {business_name} "
        f"(a {business}) and is about {problem}. "
        f"{'The customer is feeling ' + emotion + '.' if emotion else ''}\n\n"
        f"Important: The agent's name MUST be {agent_name} and the customer's name MUST be {customer_name}."
    )
    
    completion = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant designed to output JSON. Make sure the agent name is consistent throughout the conversation.",
            },
            {"role": "user", "content": enhanced_prompt},
        ],
    )
    
    result = json.loads(completion.choices[0].message.content)
    conversation = result.get("conversation", [])
    
    # Post-process the conversation to ensure name consistency
    processed_conversation = []
    for turn in conversation:
        if not isinstance(turn, dict) or "message" not in turn:
            continue
            
        message = turn["message"]
        # For the first agent message, ensure they introduce themselves with the correct name
        if turn["speaker"] == "Agent" and len(processed_conversation) < 2:
            # Check if the message has the agent introducing themselves
            if "my name is" in message.lower() or "this is" in message.lower():
                # If a different name is used, replace it with the correct agent name
                parts = message.split("my name is", 1) if "my name is" in message.lower() else message.split("this is", 1)
                if len(parts) > 1:
                    first_part = parts[0] + ("my name is" if "my name is" in message.lower() else "this is")
                    name_part = parts[1].split(",", 1) if "," in parts[1] else parts[1].split(".", 1)
                    if len(name_part) > 1:
                        message = f"{first_part} {agent_name}{name_part[1]}"
                    else:
                        message = f"{first_part} {agent_name}"
        
        processed_conversation.append({"speaker": turn["speaker"], "message": message})
    
    logger.info(
        f"Generated conversation with {len(processed_conversation)} turns"
    )
    return processed_conversation


def generate_random_business_datetime(start_date, end_date, business_hours, timezone="US/Eastern"):
    """
    Generate a random datetime within business hours in the specified date range.
    
    Args:
        start_date (datetime): Start of date range
        end_date (datetime): End of date range
        business_hours (dict): Dict with 'start' and 'end' hours (24-hour format)
        timezone (str): Timezone name
        
    Returns:
        datetime: Random datetime within business hours and date range
    """
    # Ensure dates are datetime objects
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
    if isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
    # Get timezone
    tz = pytz.timezone(timezone)
    
    # Make sure dates are timezone-aware
    if start_date.tzinfo is None:
        start_date = tz.localize(start_date)
    if end_date.tzinfo is None:
        end_date = tz.localize(end_date)
    
    # Calculate random date between start and end
    date_range_days = (end_date - start_date).days
    if date_range_days < 0:
        raise ValueError("End date must be after start date")
    
    random_day = random.randint(0, date_range_days)
    random_date = start_date + timedelta(days=random_day)
    
    # Set time within business hours
    business_start = business_hours.get('start', 9)  # Default 9 AM
    business_end = business_hours.get('end', 17)     # Default 5 PM
    
    # Generate random time within business hours
    random_hour = random.randint(business_start, business_end - 1)
    random_minute = random.randint(0, 59)
    random_second = random.randint(0, 59)
    
    # Create datetime with random business hour
    random_datetime = tz.localize(datetime(
        year=random_date.year,
        month=random_date.month,
        day=random_date.day,
        hour=random_hour,
        minute=random_minute,
        second=random_second
    ))
    
    return random_datetime


def process_conversation(
    business,
    business_name,
    problem,
    emotion,
    generation_prompt,
    progress_bar,
    s3_client,
    s3_bucket,
    generate_audio=False,
    conversation_type="voice",
    start_date=None,
    end_date=None,
    business_hours=None,
    timezone="US/Eastern"
):
    """
    Process a single conversation and return its details.
    
    Args:
        business (str): Type of business
        business_name (str): Name of the business
        problem (str): The problem/situation being discussed
        emotion (str): Customer's emotional state
        generation_prompt (str): The base prompt template for conversation generation
        progress_bar: Streamlit progress bar object
        s3_client: Boto3 S3 client
        s3_bucket (str): S3 bucket name
        generate_audio (bool): Whether to generate audio files
        conversation_type (str): Type of conversation - "voice" or "messaging"
        start_date (datetime, optional): Start of date range for conversations
        end_date (datetime, optional): End of date range for conversations
        business_hours (dict, optional): Dict with 'start' and 'end' hours (24-hour format)
        timezone (str): Timezone for generated conversations
        
    Returns:
        dict: Conversation details including vCon UUID, URL, creation time, and summary
        
    Raises:
        ValueError: If conversation generation fails
        Exception: For other processing errors
    """
    try:
        # Set default business hours if not provided
        if business_hours is None:
            business_hours = {"start": 9, "end": 17}  # 9 AM to 5 PM
            
        # Set default date range if not provided (last 30 days)
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
            
        # Generate random conversation date within business hours
        conversation_date = generate_random_business_datetime(
            start_date, end_date, business_hours, timezone
        )
        
        # Generate random identities
        agent_name = f"{random.choice(male_names)} {random.choice(last_names)}"
        customer_name = f"{random.choice(female_names)} {random.choice(last_names)}"
        agent_phone = f"+1{random.randint(1000000000, 9999999999)}"
        customer_phone = f"+1{random.randint(1000000000, 9999999999)}"
        agent_email = (
            f"{agent_name.replace(' ', '.').lower()}"
            f"@{business.replace(' ', '').lower()}.com"
        )
        customer_email = f"{customer_name.replace(' ', '.').lower()}@gmail.com"

        # Generate conversation
        conversation = generate_conversation(
            generation_prompt,
            agent_name,
            customer_name,
            business,
            problem,
            emotion,
            business_name,
            model=OPENAI_MODEL
        )

        if not conversation:
            raise ValueError("Failed to generate conversation")
            
        # Validate conversation for name consistency before proceeding
        for turn in conversation:
            if turn["speaker"] == "Agent" and "my name is" in turn["message"].lower():
                if agent_name not in turn["message"]:
                    logger.warning(f"Agent name mismatch in dialog. Fixing...")
                    # Fix will be applied in generate_conversation function
        
        vcon_uuid = str(uuid.uuid4())
        audio_url = None
        audio_signature = None
        audio_duration = 0
        combined_file = None

        if generate_audio and conversation_type == "voice":
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
            bucket_exists = ensure_s3_bucket_exists(s3_bucket, s3_client)
            if not bucket_exists:
                raise ValueError(f"Failed to create or access S3 bucket: {s3_bucket}")

            s3_audio_path = get_s3_path(combined_file, conversation_date)
            audio_url = upload_to_s3(combined_file, s3_bucket, s3_audio_path, s3_client)

        # Create and save vCon
        vcon_object = create_vcon_object(
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
            generate_audio=generate_audio,
            conversation_type=conversation_type,
            conversation_date=conversation_date,
            timezone=timezone,
            business_hours=business_hours
        )

        vcon_file = f"{vcon_uuid}.vcon.json"
        with open(vcon_file, "w") as f:
            f.write(vcon_object.to_json())

        # Upload vCon to S3
        s3_vcon_path = get_s3_path(vcon_file, conversation_date)
        vcon_url = upload_to_s3(vcon_file, s3_bucket, s3_vcon_path, s3_client)

        # Cleanup temporary files
        if combined_file and os.path.exists(combined_file):
            os.remove(combined_file)
        os.remove(vcon_file)

        return {
            "vcon_uuid": vcon_uuid,
            "vcon_url": vcon_url,
            "conversation_date": conversation_date.isoformat(),
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


# Update default prompts to emphasize name consistency
default_conversation_prompt = """
Generate a fake conversation between a customer and an agent.
The agent should introduce themselves, their company and give the customer
their name. The agent should ask for the customer's name.
As part of the conversation, have the agent ask for two pieces of
personal information. Spell out numbers. For example, 1000 should be
said as one zero zero zero, not one thousand. The conversation should be
at least 10 lines long and be complete. At the end
of the conversation, the agent should thank the customer for their time
and end the conversation. 

IMPORTANT: The agent MUST use EXACTLY the name provided to you in the prompt.
Do not make up a different name for the agent.

Return the conversation formatted like the following example:

{'conversation': 
    [
    {'speaker': 'Agent', 'message': 'xxxxx'}, 
    {'speaker': 'Customer', 'message': "xxxxx."}, 
    {'speaker': 'Agent', 'message': "xxxxxx"}
    ] 
}
"""

default_messaging_prompt = """
Generate a fake messaging conversation between a customer and an agent.
The customer should start the conversation with a problem or question,
and the agent should introduce themselves and their company when they respond.
The conversation should follow a natural messaging style - shorter messages,
more direct questions and answers.
As part of the conversation, have the agent gather necessary information
to resolve the customer's issue.
The conversation should be at least 8 messages long and be complete.
At the end, the agent should confirm the issue is resolved and offer
additional assistance if needed. 

IMPORTANT: The agent MUST use EXACTLY the name provided to you in the prompt.
Do not make up a different name for the agent.

Return the conversation formatted like the following example:

{'conversation': 
    [
    {'speaker': 'Customer', 'message': 'xxxxx'}, 
    {'speaker': 'Agent', 'message': "xxxxx."}, 
    {'speaker': 'Customer', 'message': "xxxxxx"}
    ] 
}
"""

# Main Streamlit app
def main():
    """Main Streamlit application function."""
    st.title("Fake Conversation Generator")

    col1, col2 = st.columns(2)
    with col1:
        # Business information selection
        business = col1.selectbox("Select Business", businesses)
        problem = col1.selectbox("Select Problem", problems)
        business_name = col1.text_input("Business Name", "a random business")
        
        # S3 bucket selection
        s3_bucket = col1.text_input("S3 Bucket Name", DEFAULT_S3_BUCKET)
        
        # Date range selection
        today = datetime.now().date()
        default_start = today - timedelta(days=30)
        
        st.write("Conversation Date Range")
        start_date = st.date_input("Start Date", default_start)
        end_date = st.date_input("End Date", today)
        
        # Timezone selection
        timezone_options = [
            "US/Eastern", "US/Central", "US/Mountain", "US/Pacific",
            "US/Alaska", "US/Hawaii", "Europe/London", "Asia/Tokyo"
        ]
        timezone = st.selectbox("Timezone", timezone_options, index=0)
        
        # Business hours
        st.write("Business Hours")
        col1a, col1b = st.columns(2)
        with col1a:
            business_start_hour = st.number_input("Start Hour (24h)", 0, 23, 9)
        with col1b:
            business_end_hour = st.number_input("End Hour (24h)", 0, 23, 17)
        
    with col2:
        # Get available models
        chat_models, tts_models = get_available_openai_models()
        
        # Set default model to o3-mini if available
        default_chat_model = "o3-mini"
        default_chat_model_index = 0
        if default_chat_model in chat_models:
            default_chat_model_index = chat_models.index(default_chat_model)
        elif OPENAI_MODEL in chat_models:
            default_chat_model_index = chat_models.index(OPENAI_MODEL)
        
        default_tts_model_index = 0
        if OPENAI_TTS_MODEL in tts_models:
            default_tts_model_index = tts_models.index(OPENAI_TTS_MODEL)
        
        # Add model selection dropdowns
        selected_model = col2.selectbox(
            "Select Chat Model", 
            chat_models, 
            index=default_chat_model_index
        )
        
        # Add conversation type selection with messaging as default
        conversation_type = col2.radio(
            "Conversation Type",
            ["messaging", "voice"],  # Changed order to make messaging first
            format_func=lambda x: "Text Messaging" if x == "messaging" else "Voice Call"
        )
        
        if conversation_type == "voice":
            generate_audio = col2.checkbox("Generate audio files", value=False)
            if generate_audio:
                selected_tts_model = col2.selectbox(
                    "Select TTS Model", 
                    tts_models, 
                    index=default_tts_model_index
                )
            else:
                selected_tts_model = OPENAI_TTS_MODEL
        else:
            generate_audio = False
            selected_tts_model = OPENAI_TTS_MODEL

        add_emotion = col2.checkbox("Add emotion to conversation")
        if conversation_type == "messaging" and generate_audio:
            generate_audio = False
            st.warning("Audio generation is not available for messaging conversations.")

        num_conversations = col2.number_input("Number of Conversations to Generate", 1, 100, 1)
        generate = col2.button("Generate Conversation(s)")
        
        # Display model and S3 information as toast message
        st.toast(
            f"Using model: {selected_model}, TTS model: {selected_tts_model} and S3 bucket: {s3_bucket}"
        )

    # Display the instructions in the sidebar
    with st.sidebar:
        instructions = f"""    
        ## Overview
        This app generates fake conversations between a customer and 
        an agent. The conversation is generated based on a prompt and 
        includes the names of the agent and customer, the business, 
        the problem, and optionally the emotion of the customer.
        
        For voice conversations, audio can be synthesized and a vCon (voice conversation)
        file is created and uploaded to S3.

        ## Instructions

        1. Select the conversation type (Voice Call or Text Messaging).
        2. Choose the number of conversations to generate.
        3. Click the "Generate Conversation(s)" button.
        4. The conversations will be displayed below.
        5. Each conversation will include a link to download the vCon file.

        ## Conversation Prompt

        The conversation prompt is passed to the LLM to generate the conversation.  
        You can edit this prompt to generate different types of conversations.
        """
        st.markdown(instructions)
        
        # Show different default prompts based on conversation type
        current_default_prompt = default_messaging_prompt if conversation_type == "messaging" else default_conversation_prompt
        conversation_prompt = st.text_area(
            "Conversation Prompt (Editable)", current_default_prompt, height=400
        )

    if generate:
        # Initialize S3 client with specified credentials
        s3_client = boto3.client(
            "s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY
        )
        
        # Check if bucket exists and create if needed
        if not ensure_s3_bucket_exists(s3_bucket, s3_client):
            st.error(f"Failed to access or create S3 bucket: {s3_bucket}. Please check your permissions or try another bucket name.")
            return
            
        completed_conversations = []
        progress_text = "Generating fake conversations. Please wait."
        total_bar = st.progress(0, text=progress_text)

        # Convert date inputs to datetime objects with timezone
        start_datetime = datetime.combine(start_date, dt_time.min)
        end_datetime = datetime.combine(end_date, dt_time.max)
        
        # Create business hours dictionary
        business_hours = {
            "start": business_start_hour,
            "end": business_end_hour
        }

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
                    s3_client,             # Pass S3 client
                    s3_bucket,             # Pass S3 bucket name
                    generate_audio=generate_audio,
                    conversation_type=conversation_type,
                    start_date=start_datetime,
                    end_date=end_datetime,
                    business_hours=business_hours,
                    timezone=timezone
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
            st.markdown(f"**Conversation date:** {conv['conversation_date']}")
            st.markdown(conv["summary"])
            st.markdown(f"**vCon URL:** [Download vCon]({conv['vcon_url']})")
            st.markdown("---")


if __name__ == "__main__":
    main()

