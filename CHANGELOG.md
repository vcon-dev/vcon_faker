# Changelog

## 0.2.0

Port to vCon core-05 (syntax "0.5.0").

- Depend on `vcon>=1.0.0,<2` (was `vcon==0.5.0`).
- Generated vCons carry `vcon: "0.5.0"` and one Session per conversation. Session
  `parties` lists the participants (the originator first); Dialog `parties` lists
  the contributors.
- Dialog: `mimetype` is now `mediatype`; `alg` and `signature` are replaced by
  `content_hash` (`sha512-<base64url>`); `originator` is gone (it lives in the Session).
  Text Dialogs set `encoding: "none"`. Voice conversations without audio carry the
  transcript as the text Dialog body.
- Transcript analysis body is a plain string (core-05 schema); confidence and language
  moved to the analysis `meta`.
- No Events are emitted: the faker has never produced joins, drops, holds or transfers.
- No `lawful_basis` attachment is added: the faker has never modelled consent, and a
  consent record is not invented.
- Add pytest smoke test (`tests/`) validating generated vCons against the vendored
  core-05 JSON schema (`tests/appendix-c-json-schema.json`). The LLM call is mocked.
- Bump version to 0.2.0.
