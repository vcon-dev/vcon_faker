# Synthetic Data Statement

All vCons produced by **vcon_faker** are **wholly synthetic**. The conversations are written by
a large language model and, for voice calls, the audio is synthesized with text-to-speech. **No
real person is represented**, no real call was recorded, and no real personal data is processed.

This file is the human-readable companion to the machine-readable markers carried inside each
generated vCon. Each vCon's `lawful_basis` attachment points here through `terms_of_service`.

## How synthetic origin is declared in each vCon

Every vCon from the Smart Trunk call mode (`python -m vconfaker.smarttrunk`) carries exactly
three markers, written by `vconfaker/governance.py` for vCon syntax `0.5.0`
(draft-ietf-vcon-vcon-core-05):

1. **A `lawful_basis` attachment** (`purpose: "lawful_basis"`), following
   [`draft-howe-vcon-lawful-basis`](https://datatracker.ietf.org/doc/draft-howe-vcon-lawful-basis/).
   It covers the whole call, so it is scoped with `party: 0` and `session: 0`, and its body is
   the JSON object itself with `encoding: "json"`. For synthetic data the basis is
   `legitimate_interests` (research and development on synthetic test data, the basis
   documented for `vcon-dev/fake-vcons`), `expiration` is `null`, and `purpose_grants` cover
   recording, transcription, analysis and redistribution. A proof mechanism of type
   `external_system` names the synthetic pipeline. `"lawful_basis"` is listed in the vCon's
   top-level `extensions`.

2. **A `tags` attachment** whose body holds `synthetic:true`, beside scenario tags such as
   `industry:dealership` and `scenario:service_appointment`. Tags are `name:value` strings, the
   form vcon-lib's `add_tag` writes and the Smart Trunk portal reads.

3. **`validation: "synthetic"` on every party.**

No other basis is ever written. There is no real consent record because there is no real person
to consent: a call's recording notice is real audio in the synthetic recording, and the vCon
records that it played (the `annotation_announcement*` tags), but no `consent` basis or consent
receipt is made from it.

The Streamlit generator (`vcon_faker.py`) does not add these markers yet.

## What this is NOT

vcon_faker does not emit a `synthetic_data_consent` (or `synthetic-data-consent`) attachment.
That was an earlier, retired design.

## Redistribution

Generated vCons are MIT-licensed. When redistributing, keep the synthetic markers above so
downstream consumers can always tell the data is not real.
