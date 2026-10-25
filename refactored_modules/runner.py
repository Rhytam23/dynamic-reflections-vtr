"""Stage runner so one 'Run all' gets as far as it can and never wastes GPU time after a failure."""
import json
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence


class SkipStage(Exception):
    """Raise inside a stage to skip it with a reason (not a failure)."""


class StageRunner:
    """Run notebook stages in order.

    * an exception is printed and recorded, not raised, so later independent stages still run;
    * `needs=[...]` skips a stage unless the named stages finished OK;
    * `critical=True` aborts everything after it (nothing later can work without it).
    """

    def __init__(self):
        self.status: Dict[str, str] = {}
        self.detail: Dict[str, str] = {}
        self.seconds: Dict[str, float] = {}
        self.results: Dict[str, object] = {}
        self.aborted: Optional[str] = None

    def stage(self, name: str, critical: bool = False, needs: Sequence[str] = ()) -> Callable:
        def decorate(fn):
            print(f"\n{'=' * 8} {name} {'=' * 8}", flush=True)
            if self.aborted:
                return self._record(name, "skipped", f"aborted after critical failure in '{self.aborted}'", 0.0)
            unmet = [n for n in needs if self.status.get(n) != "ok"]
            if unmet:
                return self._record(name, "skipped", f"needs {unmet}", 0.0)
            t0 = time.time()
            try:
                self.results[name] = fn()
                self._record(name, "ok", "", time.time() - t0)
            except SkipStage as e:
                self._record(name, "skipped", str(e), time.time() - t0)
            except Exception as e:
                traceback.print_exc()
                self._record(name, "FAILED", f"{type(e).__name__}: {str(e)[:300]}", time.time() - t0)
                if critical:
                    self.aborted = name
            return self.results.get(name)
        return decorate

    def _record(self, name, status, detail, secs):
        self.status[name], self.detail[name], self.seconds[name] = status, detail, secs
        print(f"-> {status}{': ' + detail if detail else ''} ({secs / 60:.1f} min)", flush=True)

    def summary(self) -> str:
        w = max([len(n) for n in self.status] + [5])
        rows = [f"{n:<{w}}  {s:<8} {self.seconds[n] / 60:6.1f} min  {self.detail[n]}" for n, s in self.status.items()]
        return "\n".join(rows)

    def save(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({n: {"status": self.status[n], "detail": self.detail[n],
                                              "minutes": round(self.seconds[n] / 60, 2)} for n in self.status}, indent=2))

    def finish(self, disconnect: bool = False, summary_path: Optional[Path] = None):
        """Print the summary, save it, and optionally release the Colab runtime so no more units are used."""
        print("\n" + "=" * 20 + " SUMMARY " + "=" * 20 + "\n" + self.summary(), flush=True)
        if summary_path:
            self.save(summary_path)
        if disconnect:
            print("\nReleasing the Colab runtime in 10 s to stop GPU billing (outputs above stay visible).", flush=True)
            time.sleep(10)
            try:
                from google.colab import runtime
                runtime.unassign()
            except Exception as e:
                print(f"Could not auto-disconnect ({e}); use Runtime > Disconnect and delete runtime.")
