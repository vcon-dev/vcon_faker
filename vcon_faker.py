import boto3
from pydub import AudioSegment
from dotenv import load_dotenv
import os
import json
import random
from openai import OpenAI
# Load environment variables from .env file
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()
client.api_key = OPENAI_API_KEY


# Initialize the Polly client with credentials
polly_client = boto3.client(
    'polly',
    region_name='us-west-2',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)


# Example lists of names traditionally associated with a particular gender
male_names = ['John', 'James', 'Robert', 'Michael', 'William']
female_names = ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth']

# Example list of industries
industries = ['retail', 'finance', 'healthcare', 'technology', 'education', 'manufacturing', 
              'construction', 'hospitality', 'transportation', 'entertainment']

# Example list of problems
problems = ['billing', 'shipping', 'returns', 'customer service', 'product quality', 'inventory',
            'pricing', 'security', 'privacy', 'data management']

def generate_conversation(prompt):
    agent_name = random.choice(male_names)
    customer_name = random.choice(female_names)
    industry = random.choice(industries)
    problem = random.choice(problems)

    prompt += f"\n\nIn this conversation, the agent's name is {agent_name} and the customer's name is {customer_name}.  "
    prompt += f"The conversation is about a problem related to the {industry} industry.  "
    prompt += f"The problem is related to {problem}.  "
    print(prompt)

    completion = client.chat.completions.create(
        model="gpt-4-0125-preview",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt}
        ]
        )
    result = json.loads(completion.choices[0].message.content)
    print(result)
    return result.get("conversation", [])



conversation_prompt = """
Generate a fake conversation between a customer and an agent.  
The agent should introduce themselves, their company and give the customer their name. The agent should ask for the customer's name.
As part of the conversation, have the agent ask for two pieces of personal information.  
Spell out numbers. For example, 1000 should be said as one zero zero zero, not one thousand.
The conversation should be at least 10 lines long.
Return the conversation formatted like the following example:

{'conversation': 
    [
    {'speaker': 'Agent', 'message': 'xxxxx'}, 
    {'speaker': 'Customer', 'message': "xxxxx."}, 
    {'speaker': 'Agent', 'message': "xxxxxx"}
    ] 
}
"""

generated_conversation = generate_conversation(conversation_prompt)

print("Generated Conversation in Tuple Format:")
for item in generated_conversation:
    print((item["speaker"], item["message"]))


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
print(f"Agent voice: {agent_voice['Name']} ({agent_voice['Id']})")
print(f"Customer voice: {customer_voice['Name']} ({customer_voice['Id']})")

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

# Export the combined audio to a single file
combined_file = 'combined_conversation.mp3'
combined.export(combined_file, format='mp3')

print(f"Combined conversation saved to {combined_file}")



