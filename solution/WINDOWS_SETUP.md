# Running this on Windows (PowerShell)

## 1. Install Python
You need Python 3.9–3.12. If you don't have it: https://python.org (check
"Add python.exe to PATH" during install). Verify:

```powershell
python --version
```

## 2. Unzip this folder somewhere, then open PowerShell in it

```powershell
cd path\to\builderr-solution
```

## 3. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run PowerShell as Administrator
once and do:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 4. Install dependencies

```powershell
pip install -r requirements.txt
```

This pulls `faster-whisper` (which pulls `ctranslate2` + `av` for audio
decoding — no separate ffmpeg install needed) and `indic-transliteration`.
First install may take a couple of minutes.

## 5. First run downloads the models once (needs internet ONE time)

The first time you run the tool, `faster-whisper` will download the
`base` and `small` Whisper model weights (a few hundred MB total) from
Hugging Face and cache them under `%USERPROFILE%\.cache\huggingface`.
After that first download, everything runs fully offline.

## 6. Smoke test (no real audio needed, just checks the pipeline runs)

```powershell
python smoke_test.py
```

You should see `result.json` written and printed to the console.

## 7. Run it on a real clip

Get a real `.wav` clip — either record yourself (mixing Hindi/English is
the interesting case), or use the sample clips from the challenge repo's
`samples/` folder. Then:

```powershell
python -m solution.transcribe --input clip.wav --mode auto --output result.json
type result.json
```

Try the other modes too:

```powershell
python -m solution.transcribe --input clip.wav --mode fast --output result_fast.json
python -m solution.transcribe --input clip.wav --mode hinglish --output result_hinglish.json
python -m solution.transcribe --input clip.wav --mode verbatim --output result_verbatim.json
```

## 8. Proving it's actually offline (matches the challenge's rule)

Once models are cached (step 5), turn off networking and re-run to prove
there's no cloud dependency — e.g. disable Wi-Fi, or block the process with
Windows Firewall:

```powershell
# Run once with network enabled to warm up/cache models, then:
New-NetFirewallRule -DisplayName "BlockPythonSTT" -Direction Outbound `
    -Program (Get-Command python).Source -Action Block
python -m solution.transcribe --input clip.wav --mode auto --output result.json
# ... then remove the rule when done testing:
Remove-NetFirewallRule -DisplayName "BlockPythonSTT"
```

If it still produces a correct result.json with the firewall rule active,
you've proven `local_only: true` for real.

## 9. Comparing against the challenge's own preview/scorecard

Drop this `solution/` folder into the actual challenge repo you cloned
from GitHub (`builderr-ai/builderr-speech-to-text`) — it already expects
`solution/transcribe.py` in this exact location — then from that repo root:

```powershell
pip install -r requirements.txt
python preview.py
```

That runs the organizers' own preview scorer against your engine on their
public dev clips, exactly like admission scoring (just offline, on your
own machine).

## Notes on what's actually happening

- **Two models, not one**: a fast `base` Whisper pass drafts first. If it
  looks like the clip has Hindi in it (either the text still contains
  Devanagari, common Hinglish words show up, or the model's own
  "this-is-English" confidence is shaky), a slower `small` Whisper pass
  re-transcribes with more care. This is the router/finalizer idea the
  challenge brief describes — plain English stays fast, mixed clips get
  the slower, more faithful path.
- **Transcribe, never translate**: both passes use Whisper's `transcribe`
  task, never `translate`. This is what keeps "yeh file update kar do"
  as itself instead of quietly becoming "update this file."
- **Devanagari → Roman, not Devanagari → English**: if Whisper outputs
  Hindi words in Devanagari script, they get transliterated letter-for-
  letter into Roman script (so it still reads like normal Hinglish
  texting), not translated into different English words. `verbatim` mode
  skips this and leaves whatever script Whisper produced.
- **No hard-coded answers**: the "Hinglish hint words" list only affects
  *routing* (which model gets a second look), never the actual output
  text — the output always comes from the ASR model, never a lookup table.

## Being honest about where this stands

This gives you a working, correctly-contracted local engine that follows
every rule in the challenge (offline, CPU-runnable, commercial-friendly
MIT-licensed models, no hard-coded answers, repetition-loop guard, full
JSON audit trail). It is a solid starting point, not a guaranteed winner —
the challenge explicitly requires beating both the best open-source engine
*and* their own RambleFix benchmark on a hidden test set, which needs real
tuning against your own recordings (accents, speed of code-switching,
domain vocabulary) before you submit. Use `preview.py` from the actual
repo, look at where your Hinglish meaning score is weakest, and iterate —
that's the real work the challenge is paying for.
