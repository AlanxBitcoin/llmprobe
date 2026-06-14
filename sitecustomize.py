from __future__ import annotations

from pathlib import Path
import traceback


def _run() -> None:
    trigger = Path(r"C:\AI_Model\probe\data\cache\.run_build_attribute_groups_en_tmp")
    log = Path(r"C:\AI_Model\probe\data\cache\build_attribute_groups_en_tmp.log")
    if not trigger.exists():
        return
    try:
        from scripts.build_attribute_groups_en_tmp import build

        build()
        log.write_text("ok\n", encoding="utf-8")
    except Exception:
        log.write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        try:
            trigger.unlink()
        except Exception:
            pass


_run()
