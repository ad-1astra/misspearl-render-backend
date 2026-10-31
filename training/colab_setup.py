# Run from the notebook with:  %run -i /content/mp/training/colab_setup.py
# Expects `run()` from cell 1. Installs everything the training scripts need.
#
# Kokoro's G2P (misaki) depends on spaCy, which has no wheels for Colab's
# Python 3.13 and fails to build from source. So when the kernel is ≥3.13
# we build an isolated Python 3.12 env with `uv` just for Kokoro; the
# corpus generator re-launches itself in that env via $KOKORO_PY.
import os
import subprocess
import sys

REPO = "https://github.com/ad-1astra/misspearl-render-backend.git"
if os.path.exists("/content/mp/.git"):
    run("git -C /content/mp pull -q")  # noqa: F821
else:
    run(f"git clone -q {REPO} /content/mp")  # noqa: F821
run("git -C /content/mp log --oneline -1")  # noqa: F821
run("apt-get -qq -y install espeak-ng > /dev/null 2>&1 || true")  # noqa: F821

KPY = sys.executable
has_kokoro = subprocess.run([sys.executable, "-c", "import kokoro"], capture_output=True).returncode == 0
if not has_kokoro and sys.version_info >= (3, 13):
    run("pip install -q uv")  # noqa: F821
    if not os.path.exists("/content/kenv/bin/python"):
        run("uv venv -q -p 3.12 /content/kenv")  # noqa: F821
    run('uv pip install -q --python /content/kenv/bin/python "kokoro>=0.9.4" soundfile scipy datasets pip')  # noqa: F821
    KPY = "/content/kenv/bin/python"
elif not has_kokoro:
    run('pip install -q "kokoro>=0.9.4" soundfile')  # noqa: F821
run(f"{KPY} -c \"import kokoro, torch; print('kokoro ready, torch', torch.__version__, 'cuda', torch.cuda.is_available())\"")  # noqa: F821
run("pip install -q datasets transformers matplotlib")  # noqa: F821
os.environ["KOKORO_PY"] = KPY
T = "/content/mp/training"
print("✅ setup complete — Kokoro python:", KPY)
