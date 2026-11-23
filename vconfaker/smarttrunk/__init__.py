"""Synthetic calls in Smart Trunk's call shape, with local voices.

A caller phones an invented business on its Smart Trunk number and is forwarded. Each call has
the recording notice at the start, the notice evidence, the recording, a transcript and, on
numbers that screen calls, a screening outcome. Every vCon is marked synthetic
(``vconfaker.governance``).

Stages: ``pack`` (scenarios and the call plan), ``writer`` (Claude writes the script,
``hygiene`` checks it), ``voice`` (Kokoro), ``telephone`` (ffmpeg), ``check`` (parakeet-mlx),
``build`` (the vCon), ``review`` (index.html), ``run`` (all of it, with metrics).
"""
