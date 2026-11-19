"""Put dummy Streamlit secrets in place before vcon_faker is imported.

vcon_faker reads st.secrets and builds OpenAI and S3 clients at import time.
The dummy values never reach a network: tests call the vCon builder directly
and mock the one chat completion call.
"""
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SECRETS = """
AWS_ACCESS_KEY = "test"
AWS_SECRET_KEY = "test"
S3_BUCKET = "test-bucket"
OPENAI_API_KEY = "test"
OPENAI_MODEL = "test-model"
OPENAI_TTS_MODEL = "test-tts"
"""

_tmp = tempfile.mkdtemp(prefix="vcon-faker-tests-")
os.makedirs(os.path.join(_tmp, ".streamlit"))
with open(os.path.join(_tmp, ".streamlit", "secrets.toml"), "w") as f:
    f.write(_SECRETS)
os.chdir(_tmp)
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
sys.path.insert(0, _ROOT)
