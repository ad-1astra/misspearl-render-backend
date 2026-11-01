# Run from the singing notebook with:  %run -i /content/mp/training/colab_setup_singing.py
# (after colab_setup.py). Installs Demucs for splitting songs into voice and instrumental.
#
# Demucs needs torchaudio, which is not published for every new PyTorch. If the
# notebook's own Python can't import it, Demucs gets an isolated env with a
# matching torch/torchaudio pair. Defines DPY = the Python that runs Demucs.
import os
import subprocess
import sys

DPY = sys.executable
_probe = "import demucs.pretrained, demucs.apply, pyarrow, huggingface_hub, scipy"
if subprocess.run([sys.executable, "-c", _probe], capture_output=True).returncode != 0:
    run("pip install -q uv")  # noqa: F821
    if not os.path.exists("/content/denv/bin/python"):
        run("uv venv -q -p 3.12 /content/denv")  # noqa: F821
    DPY = "/content/denv/bin/python"
    if subprocess.run([DPY, "-c", _probe], capture_output=True).returncode != 0:
        run(f'uv pip install -q --python {DPY} demucs "torch>=2.8,<2.10" "torchaudio>=2.8,<2.10" '  # noqa: F821
            "pyarrow huggingface_hub scipy soundfile numpy")
run(f"{DPY} -c \"import demucs.pretrained, demucs.apply, torch; "  # noqa: F821
    f"print('demucs ready — torch', torch.__version__, 'cuda', torch.cuda.is_available())\"")
print("✅ Demucs python:", DPY)
