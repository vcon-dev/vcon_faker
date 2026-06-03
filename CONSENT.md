# Synthetic Data Statement

All vCons produced by **vcon_faker** are **wholly synthetic**. The conversations
are authored by a large language model and, for voice calls, the audio is
synthesized with text-to-speech. **No real person is represented**, no real call
was recorded, and no real personal data is processed.

This file is the human-readable companion to the machine-readable markers carried
inside every generated vCon. It is referenced from each vCon's `lawful_basis`
attachment via the `terms_of_service` field.

## How synthetic origin is declared in each vCon

Every vCon emitted by vcon_faker carries exactly three markers:

1. **A `lawful_basis` attachment** (`purpose: "lawful_basis"`; a vCon-level
   attachment with `party: 0`, `dialog: 0`, `encoding: "json"`), following
   [`draft-howe-vcon-lawful-basis`](https://datatracker.ietf.org/doc/draft-howe-vcon-lawful-basis/).
   For synthetic data the basis is `legitimate_interests` (research and
   development on synthetic test data), `expiration` is `null`, and explicit
   `purpose_grants` cover recording, transcription, analysis, and
   redistribution. A `proof_mechanism` of type `external_system` documents the
   synthetic pipeline. `"lawful_basis"` is listed in the vCon's top-level
   `extensions` array.

2. **A `tags` attachment** carrying `synthetic: "true"` alongside scenario
   metadata (business, problem, emotion, conversation type).

3. **`validation: "synthetic"` on every party.**

## What this is NOT

This statement does **not** describe, and vcon_faker does **not** emit, a
`synthetic_data_consent` (or `synthetic-data-consent`) attachment. That was an
earlier, retired design. Synthetic origin is declared solely through the three
markers above. There is no real consent record because there is no real person
to consent.

## Redistribution

Generated vCons are MIT-licensed. When redistributing, preserve the synthetic
markers above so downstream consumers can always tell the data is not real.
