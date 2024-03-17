# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Set the working directory
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Streamlit port (default is 8501)
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "vcon_faker.py"]
