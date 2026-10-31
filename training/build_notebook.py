"""Regenerates MissPearl_LipSync_Training.ipynb from the cell sources below."""
import json
from pathlib import Path

MD, CODE = "markdown", "code"
cells = [
    (MD, """# 💋 Miss Pearl — Face & Lip-Sync Training
**Pearl AI Labs · Kampala**

This notebook trains the three models behind Miss Pearl's face and saves everything to
**Google Drive → `MyDrive/MissPearl-LipSync/`**:

| # | Model | Learns from | Used for |
|---|---|---|---|
| 1 | **audio → viseme** network | ~6 h of speech rendered by Kokoro-82M in 24 voices with *exact* phoneme timing | lip-syncing any audio: ElevenLabs replies, uploaded files, a live microphone |
| 2 | **G2P** (letters → phonemes) | CMU Pronouncing Dictionary (125k words) | pronouncing & lip-syncing *any* word, even names and made-up words |
| 3 | **emotion** network | ~100k sentences labelled by a DistilRoBERTa emotion teacher | brows, eyelids, cheeks, smile — the "intelligent face" |

`Runtime → Run all`. Total ≈ 50 min on a T4 GPU. Every step skips work that is already in Drive, so re-running is cheap.
The browser runtime that consumes the exported models lives in `pearl/` in the same repo."""),
    (CODE, """#@title 1 · Runtime check
import os, sys, subprocess, time, json, shutil
print(sys.version)
!nvidia-smi --query-gpu=name,memory.total --format=csv || echo "⚠️ No GPU — use Runtime ▸ Change runtime type ▸ T4 GPU"

def run(cmd):
    \"\"\"Run a shell command, stream its output, fail the cell if it fails.\"\"\"
    print('$', cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='', flush=True)
    if p.wait() != 0:
        raise RuntimeError(f'command failed with exit code {p.returncode}: {cmd}')"""),
    (CODE, """#@title 2 · Connect Google Drive  (all data, checkpoints and models are stored in MyDrive/MissPearl-LipSync)
BASE = '/content/drive/MyDrive/MissPearl-LipSync'
try:
    from google.colab import drive
    drive.mount('/content/drive')
    os.makedirs(BASE, exist_ok=True)
    print('✅ Google Drive connected →', BASE)
except Exception as e:
    BASE = '/content/MissPearl-LipSync'
    os.makedirs(BASE, exist_ok=True)
    print('⚠️ Drive not mounted (', e, ') — saving to', BASE)
LOCAL_DATA = '/content/data'            # fast local disk for training
DRIVE_DATA = f'{BASE}/data'             # permanent copy of the corpus
EXPORT     = f'{BASE}/export'           # browser-ready models
CKPT       = f'{BASE}/checkpoints'
for d in (EXPORT, CKPT): os.makedirs(d, exist_ok=True)
os.environ['PYTHONUNBUFFERED'] = '1'"""),
    (CODE, """#@title 3 · Get the training code and install Kokoro TTS
REPO = 'https://github.com/ad-1astra/misspearl-render-backend.git'
if os.path.exists('/content/mp/.git'):
    run('git -C /content/mp pull -q')
else:
    run(f'git clone -q {REPO} /content/mp')
run('git -C /content/mp log --oneline -1')
run('apt-get -qq -y install espeak-ng > /dev/null 2>&1 || true')
run('pip install -q "kokoro>=0.9.4" soundfile || pip install -q --ignore-requires-python "kokoro>=0.9.4" soundfile')
run('pip install -q datasets transformers matplotlib')
T = '/content/mp/training'"""),
    (CODE, """#@title 4 · Smoke test: render 6 sentences and check phoneme alignment
run(f'python {T}/gen_audio_data.py --data /content/smoke --smoke')
from IPython.display import Audio, display
import glob
for f in sorted(glob.glob('/content/smoke/samples/*.wav'))[:2]:
    print(f); display(Audio(f))"""),
    (CODE, """#@title 5 · Start G2P training in the background (runs while speech is generated)
g2p_log = open('/content/g2p.log', 'w')
g2p_proc = subprocess.Popen(f'python {T}/train_g2p.py --export {EXPORT} --ckpt {CKPT} --cache {BASE}/cache --minutes 14',
                            shell=True, stdout=g2p_log, stderr=subprocess.STDOUT)
print('G2P training started, pid', g2p_proc.pid)"""),
    (CODE, """#@title 6 · Generate the phoneme-aligned speech corpus with Kokoro (≈20 min)
if os.path.exists(f'{DRIVE_DATA}/index.json') and not os.path.exists(f'{LOCAL_DATA}/index.json'):
    print('Restoring corpus from Drive …'); shutil.copytree(DRIVE_DATA, LOCAL_DATA, dirs_exist_ok=True)
run(f'python {T}/gen_audio_data.py --data {LOCAL_DATA} --max-utts 6000 --minutes 20')
if not os.path.exists(f'{DRIVE_DATA}/index.json'):
    print('Saving corpus to Drive …'); shutil.copytree(LOCAL_DATA, DRIVE_DATA, dirs_exist_ok=True)
print(open(f'{LOCAL_DATA}/durations.json').read()[:1500])"""),
    (CODE, """#@title 7 · Train the audio → viseme network (≈16 min)
run(f'python {T}/train_audio2viseme.py --data {LOCAL_DATA} --export {EXPORT} --ckpt {CKPT} --minutes 16')
from IPython.display import Image, display
if os.path.exists(f'{EXPORT}/metrics/audio2viseme_confusion.png'):
    display(Image(f'{EXPORT}/metrics/audio2viseme_confusion.png'))"""),
    (CODE, """#@title 8 · Wait for G2P training to finish
while g2p_proc.poll() is None:
    time.sleep(15)
print(open('/content/g2p.log').read()[-6000:])
if g2p_proc.returncode != 0:
    raise RuntimeError('G2P training failed — see log above')"""),
    (CODE, """#@title 9 · Train the emotion network (teacher labelling + distillation, ≈8 min)
run(f'python {T}/train_emotion.py --export {EXPORT} --ckpt {CKPT}')"""),
    (CODE, """#@title 10 · Package the models (saved to Drive) and download them
ZIP = f'{BASE}/misspearl_face_models.zip'
run(f'python {T}/package_export.py --data {LOCAL_DATA} --export {EXPORT} --zip {ZIP}')
shutil.copy(ZIP, '/content/misspearl_face_models.zip')
try:
    from google.colab import files
    files.download('/content/misspearl_face_models.zip')
except Exception as e:
    print('download skipped:', e)
print('\\n📁 Everything is in Google Drive →', BASE)
run(f'ls -la {BASE} {EXPORT}')"""),
]

nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {"colab": {"provenance": [], "gpuType": "T4", "name": "MissPearl_LipSync_Training.ipynb"},
                 "kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "language_info": {"name": "python"}, "accelerator": "GPU"},
    "cells": [],
}
for kind, src in cells:
    lines = src.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    cell = {"cell_type": kind, "metadata": {}, "source": source}
    if kind == CODE:
        cell.update({"execution_count": None, "outputs": []})
    nb["cells"].append(cell)

out = Path(__file__).with_name("MissPearl_LipSync_Training.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
