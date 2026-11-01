"""Regenerates MissPearl_Singing_Training.ipynb from the cell sources below."""
import json
from pathlib import Path

MD, CODE = "markdown", "code"
cells = [
    (MD, """# 🎤 Miss Pearl — Song Lip-Sync Training
**Pearl AI Labs · Kampala**

Teaches Miss Pearl to lip-sync **songs**: her mouth follows the *singing voice* — never the drums, bass or beat —
and stays closed when only the music plays. Everything is saved to **Google Drive → `MyDrive/MissPearl-LipSync/`**.

| Data | What it teaches |
|---|---|
| **Kokoro sings** ~8 h of lyric-like lines (public-domain poetry, everyday phrases, "oh / yeah / la la") with a melody, held notes and vibrato — every mouth shape has an exact time | what sung words look like on the lips |
| **The spoken corpus** from the first notebook (~6 h) | rap / spoken parts of songs |
| **~900 real songs** from the Free Music Archive (Creative Commons), split by **Demucs** into *voice* and *instrumental* | the difference between a voice and a beat; real singing |

The network (`sing2viseme`, ~700k params, ±0.63 s context) predicts per 10 ms frame the **15 visemes** of the voice and
**"is someone singing right now?"**. It is tested on voices and artists it never heard, against the speech model as a baseline,
and on any song you upload in step 4 (that song is only tested on, never trained on).

`Runtime → Run all` (T4 GPU). ≈ 1.5 h. Every step skips work already saved in Drive."""),
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
        raise RuntimeError(f'command failed with exit code {p.returncode}: {cmd}')

def restore(drive_dir, local_dir, marker):
    if os.path.exists(f'{drive_dir}/{marker}') and not os.path.exists(f'{local_dir}/{marker}'):
        print(f'Restoring {local_dir} from Drive …'); shutil.copytree(drive_dir, local_dir, dirs_exist_ok=True)

def save(local_dir, drive_dir, marker):
    if os.path.exists(f'{local_dir}/{marker}') and not os.path.exists(f'{drive_dir}/{marker}'):
        print(f'Saving {local_dir} to Drive …'); shutil.copytree(local_dir, drive_dir, dirs_exist_ok=True)"""),
    (CODE, """#@title 2 · Connect Google Drive  (MyDrive/MissPearl-LipSync)
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
EXPORT = f'{BASE}/export_singing'
CKPT = f'{BASE}/checkpoints'
TEST = f'{BASE}/test_songs'
for d in (EXPORT, CKPT, TEST): os.makedirs(d, exist_ok=True)
os.environ['PYTHONUNBUFFERED'] = '1'"""),
    (CODE, """#@title 3 · Get the code, Kokoro TTS and Demucs
if not os.path.exists('/content/mp/.git'):
    run('git clone -q https://github.com/ad-1astra/misspearl-render-backend.git /content/mp')
%run -i /content/mp/training/colab_setup.py
tv = subprocess.run([KPY, '-c', 'import torch; print(torch.__version__.split("+")[0])'], capture_output=True, text=True).stdout.strip()
if os.path.realpath(KPY) != os.path.realpath(sys.executable):
    run(f'uv pip install -q --python {KPY} demucs "torchaudio=={tv}" pyarrow huggingface_hub')
else:
    run('pip install -q demucs pyarrow huggingface_hub')
run(f'{KPY} -c "import demucs.pretrained, demucs.apply; print(\\'demucs ready\\')"')"""),
    (CODE, """#@title 4 · A song to test on (optional — tested on, never trained on)
AUDIO_EXT = ('.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac')
songs = [f for f in os.listdir(TEST) if f.lower().endswith(AUDIO_EXT)]
if not songs:
    print('Choose a song file to test Miss Pearl on, or press Cancel to skip.')
    try:
        from google.colab import files
        for name, data in files.upload().items():
            open(f'{TEST}/{name}', 'wb').write(data)
    except Exception as e:
        print('upload skipped:', e)
print('test songs:', os.listdir(TEST))"""),
    (CODE, """#@title 5 · Real songs → voice + instrumental stems with Demucs (runs in the background, ≈25 min)
restore(f'{BASE}/music', '/content/music', 'index.json')
music_log = open('/content/music.log', 'w')
music_proc = subprocess.Popen(f'{KPY} {T}/gen_song_data.py music --out /content/music --shards 0,1 --test-dir "{TEST}"',
                              shell=True, stdout=music_log, stderr=subprocess.STDOUT)
print('music preparation started, pid', music_proc.pid)"""),
    (CODE, """#@title 6 · The spoken corpus (from the first notebook, or rebuilt ≈20 min)
restore(f'{BASE}/data', '/content/data', 'index.json')
run(f'python {T}/gen_audio_data.py --data /content/data --max-utts 6000 --minutes 20')
save('/content/data', f'{BASE}/data', 'index.json')"""),
    (CODE, """#@title 7 · Kokoro sings: build the sung corpus (≈22 min)
restore(f'{BASE}/sung', '/content/sung', 'index.json')
run(f'python {T}/gen_song_data.py sung --data /content/sung --max-lines 9000 --minutes 22')
save('/content/sung', f'{BASE}/sung', 'index.json')
from IPython.display import Audio, display
import glob
for f in sorted(glob.glob('/content/sung/samples/*.wav'))[:3]:
    print(f); display(Audio(f))"""),
    (CODE, """#@title 8 · Wait for the song stems
while music_proc.poll() is None:
    time.sleep(20)
print(open('/content/music.log').read()[-4000:])
if music_proc.returncode != 0:
    raise RuntimeError('music preparation failed — see log above')
save('/content/music', f'{BASE}/music', 'index.json')"""),
    (CODE, """#@title 9 · Train the song lip-sync network (≈45 min)
run(f'python {T}/train_sing2viseme.py --speech /content/data --sung /content/sung --music /content/music '
    f'--teacher /content/mp/models/audio2viseme.json --export {EXPORT} --ckpt {CKPT} --minutes 45')"""),
    (CODE, """#@title 10 · Package the model (saved to Drive) and download it
ZIP = f'{BASE}/misspearl_singing_model.zip'
shutil.make_archive(ZIP[:-4], 'zip', EXPORT)
shutil.copy(ZIP, '/content/misspearl_singing_model.zip')
print(json.dumps(json.load(open(f'{EXPORT}/metrics/sing2viseme.json'))['test_songs'], indent=1))
try:
    from google.colab import files
    files.download('/content/misspearl_singing_model.zip')
except Exception as e:
    print('download skipped:', e)
print('\\n📁 Everything is in Google Drive →', BASE)
run(f'ls -la {EXPORT}')"""),
]

nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {"colab": {"provenance": [], "gpuType": "T4", "name": "MissPearl_Singing_Training.ipynb"},
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

out = Path(__file__).with_name("MissPearl_Singing_Training.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
