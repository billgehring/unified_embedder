import os
import shutil
import subprocess
from typing import Optional


def _run_cmd(cmd: list[str], timeout: float = 5.0) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
            check=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as e:
        return 1, "", str(e)


def log_tesseract_diagnostics(logger, quick: bool = True) -> None:
    """Log best-effort diagnostics about the local Tesseract setup.

    - Detect binary path and version
    - Show TESSDATA_PREFIX and check for osd.traineddata
    - Optionally list available languages (quick=False)
    """
    try:
        tess_env_prefix = os.getenv("TESSDATA_PREFIX")
        tess_cmd_env = os.getenv("TESSERACT_CMD")
        which_tess = shutil.which(tess_cmd_env or "tesseract")

        logger.info(f"Tesseract diagnostics: cmd_env={tess_cmd_env}, resolved_cmd={which_tess}")
        logger.info(f"TESSDATA_PREFIX={tess_env_prefix}")

        if which_tess:
            rc, out, err = _run_cmd([which_tess, "--version"])
            if out:
                first_line = out.splitlines()[0]
                logger.info(f"tesseract --version: {first_line}")
            if err and rc != 0:
                logger.warning(f"tesseract --version stderr: {err}")

            if not quick:
                rc, out, err = _run_cmd([which_tess, "--list-langs"], timeout=10.0)
                if rc == 0 and out:
                    langs = [l for l in out.splitlines() if l and not l.lower().startswith("list of available")] 
                    logger.info(f"tesseract languages: {', '.join(langs[:20])}{' ...' if len(langs) > 20 else ''}")
                elif err:
                    logger.warning(f"--list-langs error: {err}")

        # Check tessdata directory for osd.traineddata
        if tess_env_prefix and os.path.isdir(tess_env_prefix):
            osd_path = os.path.join(tess_env_prefix, "osd.traineddata")
            eng_path = os.path.join(tess_env_prefix, "eng.traineddata")
            logger.info(
                f"tessdata present: osd={'yes' if os.path.isfile(osd_path) else 'no'}, "
                f"eng={'yes' if os.path.isfile(eng_path) else 'no'}"
            )
        else:
            logger.warning("TESSDATA_PREFIX not set or directory missing; OSD may fail.")
    except Exception as e:
        logger.warning(f"Tesseract diagnostics failed: {e}")

