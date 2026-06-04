"""vcon_faker engine: headless, provider-agnostic synthetic vCon generation.

Importable without Streamlit, OpenAI, or AWS. The Streamlit app and the CLI are
thin clients over these building blocks:

* ``config.Settings``            - typed configuration from the environment
* ``providers.llm``              - LLM chat (scripted | openai | ...)
* ``generators.conversation``    - scenario -> conversation turns
* ``vconbuild.builder``          - spec-compliant vCon assembly (IETF core-02)
* ``storage``                    - output sink (local | s3 | ...)

Run the CLI with ``python -m vconfaker generate --help``.
"""

__version__ = "0.2.0"
__all__ = ["__version__"]
