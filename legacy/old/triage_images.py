#!/usr/bin/env python3
"""
triage_images.py — quick keyboard-driven GUI to classify images into "good" and "bad" folders.

Features
- Simple Tkinter window to view one image at a time
- Keyboard shortcuts:
    g=good, b=bad, s=skip, u=undo, q=quit, ←/→=prev/next (wrap-around)
    f or / = set filter (comma-separated AND terms)
    c = clear filter
    G = change GOOD folder
    B = change BAD folder
    M = open Param Filter Panel (dropdowns per parameter; "Any" leaves param unfiltered)
    v (or double-click) = open current image in external viewer
- Filter supports decimals in input (e.g., AH=2.3) and normalizes to underscores (AH=2_3)
- Top bar shows parsed key=value pairs from filename (underscores between digits treated as decimals)
- Status bar (larger font) shows progress on line 1 and GOOD/BAD/filter on line 2
- Moves, copies, or symlinks images into target folders (default: move)
- CSV log with actions so you can resume later and audit decisions
- Resume behavior: already-triaged files in the log are skipped unless --ignore-log is set
- Optional shuffling and recursion

Dependencies
- Python 3.x
- Pillow (PIL): pip install pillow

Usage
    python triage_images.py /path/to/images --good GOOD --bad BAD --move
    python triage_images.py ./outputs --shuffle --recursive --copy
"""

from __future__ import annotations
import math
import argparse
import csv
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Literal, Dict, Set

from PIL import Image, ImageTk
import tkinter as tk
from tkinter import simpledialog, filedialog, ttk
import subprocess


Action = Literal["good", "bad", "skip", "undo"]

@dataclass
class Args:
    root: Path
    good_dir: Path
    bad_dir: Path
    mode: Literal["move", "copy", "symlink"]
    recursive: bool
    shuffle: bool
    extensions: Tuple[str, ...]
    log_path: Path
    ignore_log: bool
    start_from: Optional[str]
    window_width: int
    window_height: int

def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Keyboard GUI to triage images into 'good'/'bad' folders.")
    p.add_argument("root", type=Path, help="Folder containing images to review")
    p.add_argument("--good", dest="good_dir", type=Path, default=None, help="Destination folder for GOOD images (default: <root>/GOOD)")
    p.add_argument("--bad", dest="bad_dir", type=Path, default=None, help="Destination folder for BAD images (default: <root>/BAD)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--move", action="store_true", help="Move files (default)")
    g.add_argument("--copy", action="store_true", help="Copy files")
    g.add_argument("--symlink", action="store_true", help="Symlink files")
    p.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    p.add_argument("--shuffle", action="store_true", help="Shuffle the review order")
    p.add_argument("--ext", default="png,jpg,jpeg,jpe,tif,tiff,bmp,webp", help="Comma-separated list of file extensions to include")
    p.add_argument("--log", dest="log_path", type=Path, default=None, help="CSV log path (default: <root>/triage_log.csv)")
    p.add_argument("--ignore-log", action="store_true", help="Do not skip items already present in the log")
    p.add_argument("--start-from", type=str, default=None, help="Start from the first file whose path contains this substring")
    p.add_argument("--size", type=str, default="1280x800", help="Window size, e.g., 1600x1000")
    args = p.parse_args()

    root = args.root.expanduser().resolve()
    good_dir = (args.good_dir.expanduser().resolve() if args.good_dir is not None else (root / "GOOD").resolve())
    bad_dir = (args.bad_dir.expanduser().resolve() if args.bad_dir is not None else (root / "BAD").resolve())

    if args.copy:
        mode = "copy"
    elif args.symlink:
        mode = "symlink"
    else:
        mode = "move"

    try:
        w_str, h_str = args.size.lower().split("x")
        window_width, window_height = int(w_str), int(h_str)
    except Exception:
        window_width, window_height = 1280, 800

    log_path = args.log_path or (root / "triage_log.csv")
    exts = tuple("." + e.strip().lower().lstrip(".") for e in args.ext.split(","))

    return Args(
        root=root,
        good_dir=good_dir,
        bad_dir=bad_dir,
        mode=mode,  # type: ignore
        recursive=args.recursive,
        shuffle=args.shuffle,
        extensions=exts,
        log_path=log_path,
        ignore_log=args.ignore_log,
        start_from=args.start_from,
        window_width=window_width,
        window_height=window_height,
    )

def collect_images(root: Path, recursive: bool, extensions: Tuple[str, ...]) -> List[Path]:
    files: List[Path] = []
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in extensions:
                files.append(p)
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in extensions:
                files.append(p)
    files.sort()
    return files

def read_processed_from_log(log_path: Path) -> set[str]:
    processed: set[str] = set()
    if not log_path.exists():
        return processed
    try:
        with log_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row.get("src", ""))
    except Exception:
        pass
    return processed

def ensure_dirs(*paths: Path):
    for d in paths:
        d.mkdir(parents=True, exist_ok=True)

def atomic_move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))

def atomic_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))

def atomic_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)

def write_log_header_if_needed(log_path: Path):
    if not log_path.exists():
        with log_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ts", "action", "src", "dest", "mode"])

def append_log(log_path: Path, action: Action, src: Path, dest: Optional[Path], mode: str):
    with log_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([int(time.time()), action, str(src), str(dest) if dest else "", mode])

# --- Sticky filter helpers (module-level) ---
def _last_filter_path(root: Path) -> Path:
    """Persist last-used filter text per image folder."""
    return (root / ".triage_last_filter.txt").resolve()

def _load_last_filter(root: Path) -> str:
    p = _last_filter_path(root)
    try:
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        return ""

def _save_last_filter(root: Path, text: str) -> None:
    p = _last_filter_path(root)
    try:
        p.write_text(text, encoding="utf-8")
    except Exception:
        pass


def _parse_multi_filter_input(text: str) -> tuple[dict[str, list[str]], list[str]]:
    """
    Parse input like 'AH=2.35,2.5 / BH=1.0,1.1e0 / token, other'

    Returns:
    advanced:   {param_lower: [value1, value2, ...]}  (values normalized: 2.3 -> 2_3)
    free_terms: ['raw', 'tokens']  (legacy non k=v terms, comma-separated)

    Notes:
    - '/' separates parameter blocks
    - each block may be 'k=v1,v2,...' or just comma terms
    - values accept decimals and scientific; we normalize 2.3 → 2_3 for filename matching
    """
    advanced: dict[str, list[str]] = {}
    free_terms: list[str] = []

    # split top-level on '/'
    sections = [s.strip() for s in text.split('/') if s.strip()]
    for sec in sections:
        if '=' in sec:
            k, vs = sec.split('=', 1)
            k = k.strip().lower()
            vals = [v.strip() for v in vs.split(',') if v.strip()]
            # normalize 2.3 -> 2_3 for filename matching
            vals_norm = [re.sub(r'(?<=\d)\.(?=\d)', '_', v.lower()) for v in vals]
            if vals_norm:
                advanced.setdefault(k, [])
                for v in vals_norm:
                    if v not in advanced[k]:
                        advanced[k].append(v)
        else:
            # comma list of free tokens
            toks = [t.strip() for t in sec.split(',') if t.strip()]
            toks_norm = [re.sub(r'(?<=\d)\.(?=\d)', '_', t.lower()) for t in toks]
            free_terms.extend(toks_norm)

    return advanced, free_terms

class TriageApp:
    def __init__(self, args: Args):
        self.args = args

        # Master list of candidates (after initial filtering)
        all_files = collect_images(args.root, args.recursive, args.extensions)

        if not args.ignore_log:
            processed = read_processed_from_log(args.log_path)
            all_files = [f for f in all_files if str(f) not in processed]

        if args.shuffle:
            random.shuffle(all_files)

        if args.start_from:
            idx = next((i for i, p in enumerate(all_files) if args.start_from in str(p)), None)
            if idx is not None:
                all_files = all_files[idx:]

        self.all_files: list[Path] = all_files
        self.removed: set[Path] = set()  # files already triaged in this session

        # Comma-separated AND terms (normalized); match against filename
        self.active_filters: list[str] = []
        self.advanced_filters: Dict[str, List[str]] = {}  # param -> list[str]
        self.last_filter_text: str = _load_last_filter(self.args.root)  # NEW: sticky filter text

        # Parameter picker state (built from OptionMenus): dict[param] = selected string or None
        self.param_picker_selection: Dict[str, Optional[str]] = {}

        self.files: list[Path] = self._apply_filter()

        self.index = 0
        self.history: list[Tuple[Action, Path, Optional[Path]]] = []

        ensure_dirs(self.args.good_dir, self.args.bad_dir)
        write_log_header_if_needed(self.args.log_path)

        # GUI
        self.root = tk.Tk()
        self.root.title("Image Triage — g/b/s/u, ←/→ wrap, q, f/=filter, c=clear, G/B=change dest, M=param panel")
        self.root.geometry(f"{args.window_width}x{args.window_height}")

        # --- Top bar showing parsed parameters ---
        self.param_text = tk.StringVar(value="")
        self.param_label = tk.Label(
            self.root,
            textvariable=self.param_text,
            anchor="center",
            justify="center",
            font=("Helvetica", 20, "bold"),
            bg="black",
            fg="yellow",
            pady=8,
            wraplength=self.args.window_width - 10,
        )
        self.param_label.pack(fill=tk.X)

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        # Open current image in external viewer
        self.root.bind("<v>", lambda e: self.open_in_viewer())
        self.canvas.bind("<Double-Button-1>", lambda e: self.open_in_viewer())

        # Make the window’s X button behave like 'q'
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        # --- Status line (larger font + padding + high contrast) ---
        self.status = tk.StringVar(value="")
        self.label = tk.Label(
            self.root,
            textvariable=self.status,
            anchor="w",
            justify="left",
            font=("Helvetica", 18, "bold"),
            bg="black",
            fg="lime",
            padx=10,
            pady=6,
            wraplength=self.args.window_width - 40,
        )
        self.label.pack(fill=tk.X)

        # key bindings
        self.root.bind("<g>", lambda e: self.mark("good"))
        self.root.bind("<b>", lambda e: self.mark("bad"))
        self.root.bind("<s>", lambda e: self.skip())
        self.root.bind("<u>", lambda e: self.undo())
        self.root.bind("<q>", lambda e: self.quit())
        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())

        # filter & change-dest
        self.root.bind("<f>", lambda e: self.prompt_filter())
        self.root.bind("/",   lambda e: self.prompt_filter())
        self.root.bind("<c>", lambda e: self.clear_filter())
        self.root.bind("<G>", lambda e: self.change_dest("good"))
        self.root.bind("<B>", lambda e: self.change_dest("bad"))

        # NEW: parameter panel
        self.root.bind("<M>", lambda e: self.open_param_filter_panel())
        self.root.bind("<m>", lambda e: self.open_param_filter_panel())

        self.photo = None  # keep reference to avoid GC
        self.update_view()

        # Update on resize to keep image fit
        self.root.bind("<Configure>", self.on_resize)

    # ---------- ordering helpers ----------
    def _base_order(self) -> tuple[list[Path], dict[Path, int]]:
        """All remaining candidates in stable order + a rank map."""
        base = [p for p in self.all_files if p not in self.removed]
        ranks = {p: i for i, p in enumerate(base)}
        return base, ranks

    def _closest_index(self, anchor: Optional[Path], new_list: list[Path]) -> int:
        """If anchor is in new_list, go to it; else go to item whose base-rank is closest."""
        if not new_list:
            return 0
        if anchor in new_list:
            return new_list.index(anchor)
        base, ranks = self._base_order()
        if anchor not in ranks:
            return min(self.index, len(new_list) - 1)
        ar = ranks[anchor]
        best_i = min(range(len(new_list)), key=lambda i: abs(ranks.get(new_list[i], 10**9) - ar))
        return best_i

    # ---------- parsing helpers ----------
    


    def _parse_params_from_filename(self, name: str) -> dict[str, tuple[str, Optional[float]]]:
        """
        Return {key_lower: (value_str_clean, value_float_or_None)} parsed from filename.
        - Splits on "__"
        - Converts underscores between digits to decimal points (2_35 -> 2.35)
        - Strips one trailing '_<letters...>' junk (e.g., '_BHn6')
        - Detects numbers incl. scientific notation
        """
        num_re = re.compile(r'^[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?$')

        stem = Path(name).stem
        chunks = [c for c in stem.split("__") if c]
        out: dict[str, tuple[str, Optional[float]]] = {}

        for ch in chunks:
            if "=" not in ch:
                continue
            k, v = ch.split("=", 1)
            k = k.strip().lower()

            # 2_35 -> 2.35 only when underscore is between digits
            v = re.sub(r'(?<=\d)_(?=\d)', '.', v)
            v = v.strip().strip('_')
            # drop a trailing '_Junk'
            v = re.sub(r'_[A-Za-z]\w*$', '', v)

            if num_re.match(v):
                try:
                    out[k] = (v, float(v))
                except ValueError:
                    out[k] = (v, None)
            else:
                # keep non-numeric too (e.g., mode names), but float=None
                out[k] = (v, None)
        return out

    # ---------- parameter value discovery ----------
    def _discover_param_values(self) -> Dict[str, List[str]]:
        """
        Scan remaining (not yet triaged) files and build a map:
            key -> sorted list of unique value strings (display form, decimals '.', sci ok)
        Numeric-looking values are sorted numerically; otherwise lexicographically.
        """
        base_candidates = [p for p in self.all_files if p not in self.removed]
        values: Dict[str, Set[Tuple[str, Optional[float]]]] = {}
        for path in base_candidates:
            params = self._parse_params_from_filename(path.name)
            for k, (sval, fval) in params.items():
                values.setdefault(k, set()).add((sval, fval))

        result: Dict[str, List[str]] = {}
        for k, pairset in values.items():
            # Sort numeric-first if possible
            pairs = list(pairset)
            if all(fv is not None for _, fv in pairs) and pairs:
                pairs.sort(key=lambda t: t[1])  # sort by numeric value
            else:
                pairs.sort(key=lambda t: t[0])  # alphabetical on string
            result[k] = [s for (s, _) in pairs]
        return result

    # ---------- filtering ----------
    def _apply_filter(self) -> list[Path]:
        base = [p for p in self.all_files if p not in self.removed]

        # Combine Param Panel selection into strict k=v terms as before
        picker_terms: list[str] = []
        for k, v in self.param_picker_selection.items():
            if v is None:
                continue
            v_token = re.sub(r'(?<=\d)\.(?=\d)', '_', v)  # 2.3 -> 2_3
            picker_terms.append(f"{k}={v_token}")

        legacy_terms = list(self.active_filters) + picker_terms  # AND of these

        def matches_legacy_terms(path_like: Path, params: dict[str, tuple[str, Optional[float]]]) -> bool:
            # previous behavior: each term must be satisfied
            for term in legacy_terms:
                term = term.strip().lower()
                if "=" not in term:
                    # raw token must appear in filename
                    if term not in path_like.name.lower():
                        return False
                    continue
                k, v = term.split("=", 1)
                k = k.strip().lower()
                v = v.strip()
                if k not in params:
                    return False
                v_norm = re.sub(r'(?<=\d)_(?=\d)', '.', v)
                try:
                    fval = float(v_norm)
                    _, pval = params[k]
                    if pval is None or not math.isclose(pval, fval, rel_tol=1e-9, abs_tol=1e-12):
                        return False
                except ValueError:
                    pstr, _ = params[k]
                    if pstr.lower() != v_norm.lower():
                        return False
            return True

        def matches_advanced(path_like: Path, params: dict[str, tuple[str, Optional[float]]]) -> bool:
            # For each param in advanced_filters, the image must match ANY of the values for that param.
            for k, vals in self.advanced_filters.items():
                if k not in params:
                    return False
                pstr, pnum = params[k]
                # success if ANY value matches (numeric or string)
                ok_one = False
                for raw in vals:
                    v_norm_dot = re.sub(r'(?<=\d)_(?=\d)', '.', raw)  # back to dot for float try
                    try:
                        fval = float(v_norm_dot)
                        if pnum is not None and math.isclose(pnum, fval, rel_tol=1e-9, abs_tol=1e-12):
                            ok_one = True
                            break
                    except ValueError:
                        pass
                    if pstr.lower() == v_norm_dot.lower():
                        ok_one = True
                        break
                if not ok_one:
                    return False
            return True

        if legacy_terms or self.advanced_filters:
            filtered: list[Path] = []
            for p in base:
                params = self._parse_params_from_filename(p.name)
                if not params:
                    continue
                if legacy_terms and not matches_legacy_terms(p, params):
                    continue
                if self.advanced_filters and not matches_advanced(p, params):
                    continue
                filtered.append(p)
            base = filtered

        return base


    def open_in_viewer(self):
        p = self.current_file()
        if not p:
            self.set_status("No image selected.")
            return
        try:
            if sys.platform.startswith("linux"):
                for cmd in (["xdg-open", str(p)], ["gio", "open", str(p)]):

                    try:
                        subprocess.Popen(cmd,
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
                        self.set_status(f"Opened in external viewer: {p.name}")
                        return
                    except Exception:
                        continue
                self.set_status("Could not find xdg-open or gio on PATH.")
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(p)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.set_status(f"Opened in external viewer: {p.name}")
            elif sys.platform.startswith("win"):
                os.startfile(str(p))  # type: ignore[attr-defined]
                self.set_status(f"Opened in external viewer: {p.name}")
            else:
                self.set_status("Unsupported platform for opening viewer.")
        except Exception as e:
            self.set_status(f"Failed to open viewer: {e}")

    # ---------- Param Filter Panel ----------
    def open_param_filter_panel(self):
        """
        Modal dialog with one dropdown per discovered parameter.
        Each dropdown lists: Any (no filter), then the sorted values for that parameter.
        On Apply, updates param_picker_selection and re-filters.
        """
        anchor = self.current_file()

        values_map = self._discover_param_values()
        if not values_map:
            self.set_status("No parameters discovered in filenames.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Parameter Filter Panel — choose values (Any = unfiltered)")
        dialog.configure(bg="black")
        dialog.geometry("900x600")
        dialog.transient(self.root)
        dialog.grab_set()

        # Header
        header = tk.Label(
            dialog,
            text="Select a value for any parameter to filter; choose 'Any' to leave it unfiltered.",
            bg="black", fg="white", font=("Helvetica", 14)
        )
        header.pack(pady=10)

        # Scrollable area (in case there are many parameters)
        container = tk.Frame(dialog, bg="black")
        container.pack(fill="both", expand=True, padx=12, pady=8)

        canvas = tk.Canvas(container, bg="black", highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg="black")

        inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Build dropdowns
        var_by_key: Dict[str, tk.StringVar] = {}
        row = 0
        # Stable order: alphabetic by key
        for key in sorted(values_map.keys()):
            vals = values_map[key]
            frame = tk.Frame(inner, bg="black")
            frame.grid(row=row, column=0, sticky="ew", padx=6, pady=4)
            frame.columnconfigure(1, weight=1)

            label = tk.Label(frame, text=key, bg="black", fg="#9fe", font=("Helvetica", 13, "bold"))
            label.grid(row=0, column=0, sticky="w", padx=(2, 10))

            # Dropdown values: Any + actual values (shown with decimals)
            options = ["Any"] + vals
            sel = self.param_picker_selection.get(key)
            initial = "Any" if (sel is None or sel not in vals) else sel

            v = tk.StringVar(value=initial)
            var_by_key[key] = v

            om = ttk.OptionMenu(frame, v, initial, *options)
            om.grid(row=0, column=1, sticky="ew")

            row += 1

        # Buttons
        btns = tk.Frame(dialog, bg="black")
        btns.pack(pady=12)

        def do_apply():
            # Build new selection map
            new_sel: Dict[str, Optional[str]] = {}
            for k, v in var_by_key.items():
                val = v.get()
                if val == "Any":
                    new_sel[k] = None
                else:
                    new_sel[k] = val  # keep display form with dots and sci notation
            self.param_picker_selection = new_sel

            # Re-run filter and keep nearest index to anchor
            new_list = self._apply_filter()
            self.index = self._closest_index(anchor, new_list)
            self.files = new_list
            self.update_view()
            dialog.destroy()

        def do_clear_all():
            for v in var_by_key.values():
                v.set("Any")

        def do_cancel():
            dialog.destroy()

        apply_btn = tk.Button(btns, text="Apply", command=do_apply, font=("Helvetica", 14))
        clear_btn = tk.Button(btns, text="Clear all to Any", command=do_clear_all, font=("Helvetica", 14))
        cancel_btn = tk.Button(btns, text="Cancel", command=do_cancel, font=("Helvetica", 14))
        apply_btn.pack(side="left", padx=10)
        clear_btn.pack(side="left", padx=10)
        cancel_btn.pack(side="left", padx=10)

        dialog.bind("<Return>", lambda e: do_apply())
        dialog.bind("<Escape>", lambda e: do_cancel())

        # Focus
        dialog.focus_set()
        dialog.wait_window(dialog)

    def prompt_filter(self):
        anchor = self.current_file()  # remember where we are

        # --- Compute initial text for the dialog ---
        # Priority:
        #   1) Previously typed filter (self.last_filter_text), if any
        #   2) Rehydrate from current active selections
        #   3) Prefill from current image parameters
        initial = self.last_filter_text.strip()

        if not initial:
            if self.active_filters or self.advanced_filters:
                parts = []
                for k, vals in sorted(self.advanced_filters.items()):
                    shown = ",".join([re.sub(r'(?<=\d)_(?=\d)', '.', v) for v in vals])
                    parts.append(f"{k}={shown}")
                if self.active_filters:
                    parts.append(", ".join([re.sub(r'(?<=\d)_(?=\d)', '.', t) for t in self.active_filters]))
                initial = " / ".join(parts)
            else:
                if anchor is not None:
                    parsed = self.extract_params_from_filename(anchor.name)
                    if parsed:
                        initial = parsed.replace("   ", " / ")

        dialog = tk.Toplevel(self.root)
        dialog.title("Set Filters — use '/' to separate parameters; commas for multiple values")
        dialog.configure(bg="black")
        dialog.geometry("1100x240")
        dialog.transient(self.root)
        dialog.grab_set()

        label = tk.Label(
            dialog,
            text=("Examples:\n"
                "  AH=2.35,2.5 / BH=1.0,1.1e0 / kHA=1e12,1e13\n"
                "  (OR within each parameter; AND across parameters)\n"
                "You can also include plain terms (comma-separated) without '='."),
            bg="black", fg="white", font=("Helvetica", 12), justify="left"
        )
        label.pack(pady=8, anchor="w", padx=10)

        entry = tk.Entry(
            dialog,
            font=("Consolas", 16),
            width=140,
            bg="#222",
            fg="#00FF00",
            insertbackground="white"
        )
        entry.insert(0, initial)
        entry.pack(padx=10, pady=10, fill="x")

        result = {"text": None}

        def on_ok():
            result["text"] = entry.get()
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        button_frame = tk.Frame(dialog, bg="black")
        button_frame.pack(pady=10)
        ok_btn = tk.Button(button_frame, text="Apply Filter", command=on_ok, font=("Helvetica", 14))
        ok_btn.pack(side="left", padx=10)
        cancel_btn = tk.Button(button_frame, text="Cancel", command=on_cancel, font=("Helvetica", 14))
        cancel_btn.pack(side="left", padx=10)

        entry.focus()
        dialog.bind("<Return>", lambda e: on_ok())
        dialog.bind("<Escape>", lambda e: on_cancel())

        self.root.wait_window(dialog)
        s = result["text"]
        if s is None:
            return

        # Persist sticky text exactly as typed
        self.last_filter_text = s
        _save_last_filter(self.args.root, self.last_filter_text)

        # Parse into structures used by _apply_filter
        advanced, free_terms = _parse_multi_filter_input(s)
        self.advanced_filters = advanced
        self.active_filters = free_terms

        new_list = self._apply_filter()
        self.index = self._closest_index(anchor, new_list)
        self.files = new_list
        self.update_view()



    def clear_filter(self):
        anchor = self.current_file()
        self.active_filters = []
        self.advanced_filters = {}            # <— add this if you want full clear
        # keep param picker selections as-is; to clear those, use M → "Clear all to Any"
        new_list = self._apply_filter()
        self.index = self._closest_index(anchor, new_list)
        self.files = new_list
        self.update_view()


    # ---------- destination change ----------
    def change_dest(self, which: Literal["good", "bad"]):
        anchor = self.current_file()
        start_dir = str(self.args.good_dir if which == "good" else self.args.bad_dir)
        newdir = filedialog.askdirectory(title=f"Choose new {which.upper()} folder", initialdir=start_dir or "/")
        if not newdir:
            return
        new_path = Path(newdir).expanduser().resolve()
        ensure_dirs(new_path)
        if which == "good":
            self.args.good_dir = new_path
        else:
            self.args.bad_dir = new_path
        # recompute view & re-anchor
        new_list = self._apply_filter()
        self.index = self._closest_index(anchor, new_list)
        self.files = new_list
        self.set_status("Destination changed.")
        self.update_view()

    # ---------- GUI plumbing ----------
    def run(self):
        # Force a refresh once the window is realized, after Tk finishes layout.
        self.root.after(200, self._force_initial_refresh)
        self.root.mainloop()

    def _force_initial_refresh(self):
        """Ensure the first image is properly scaled after the window is drawn."""
        try:
            self.root.update_idletasks()
            self.update_view()
        except Exception:
            pass

    def on_resize(self, event):
        """Redraw only when the canvas size actually changes, not on every event."""
        if event.widget == self.root or event.widget == self.canvas:
            new_w, new_h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if not hasattr(self, "_last_size") or (new_w, new_h) != self._last_size:
                self._last_size = (new_w, new_h)
                self.root.after(100, self.update_view)

    def current_file(self) -> Optional[Path]:
        if 0 <= self.index < len(self.files):
            return self.files[self.index]
        return None

    def set_status(self, text: str):
        # Summarize both filter sources
        filt_terms = []

        # advanced multi-value blocks: show as k∈{...}
        if self.advanced_filters:
            parts = []
            for k, vals in sorted(self.advanced_filters.items()):
                shown = ",".join([re.sub(r'(?<=\d)_(?=\d)', '.', v) for v in vals])
                parts.append(f"{k}∈{{{shown}}}")
            if parts:
                filt_terms.append(" / ".join(parts))

        # legacy free terms
        if self.active_filters:
            filt_terms.append(", ".join([re.sub(r'(?<=\d)_(?=\d)', '.', t) for t in self.active_filters]))

        # Param picker strict k=v
        if any(v is not None for v in self.param_picker_selection.values()):
            parts = []
            for k, v in sorted(self.param_picker_selection.items()):
                if v is not None:
                    parts.append(f"{k}={v}")
            if parts:
                filt_terms.append(", ".join(parts))

        filt = ""
        if filt_terms:
            filt = " | filter= " + " | ".join(filt_terms)

        meta = f'GOOD→{self.args.good_dir.name} | BAD→{self.args.bad_dir.name}{filt}'
        self.status.set(f"{text}\n{meta}")


    # ---------- filename param parser (top bar) ----------
    def extract_params_from_filename(self, filename: str) -> str:
        """
        Parse key=value tokens from names like:
          '...__AH=2_3__A_trap=0_9__kHA=1e12__kHD=1e12_BHn6.png'
        Rules:
          - split by "__"
          - convert underscores between digits into decimals (2_35 -> 2.35)
          - strip one trailing '_<letters...>' junk (e.g., '_BHn6')
          - keep scientific notation intact
        """
        name = Path(filename).stem
        chunks = [c for c in name.split("__") if c]  # primary delimiter
        tokens, seen = [], set()
        num_pattern = re.compile(r'^[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?$')

        for chunk in chunks:
            if "=" not in chunk:
                continue
            k, v = chunk.split("=", 1)

            # convert 0_9 -> 0.9, 2_35 -> 2.35 (only underscores between digits)
            v = re.sub(r'(?<=\d)_(?=\d)', '.', v)

            # trim spaces/leading underscores
            v = v.strip().strip('_')

            # drop a single trailing "_junk" that starts with a letter (e.g., '_BHn6')
            v = re.sub(r'_[A-Za-z]\w*$', '', v)

            # if value is numeric (incl sci) or at least contains a digit, keep it
            if not num_pattern.match(v):
                if not re.search(r'\d', v):
                    continue  # pure junk, skip

            t = f"{k.strip()}={v}"
            if t not in seen:
                seen.add(t)
                tokens.append(t)

        return "   ".join(tokens)

    def fit_image_to_canvas(self, img: Image.Image) -> Image.Image:
        cwidth = self.canvas.winfo_width() or self.args.window_width
        cheight = self.canvas.winfo_height() or self.args.window_height - 24
        img_width, img_height = img.size
        scale = min(cwidth / img_width, cheight / img_height)
        if scale <= 0:
            scale = 1.0
        new_size = (max(1, int(img_width * scale)), max(1, int(img_height * scale)))
        return img.resize(new_size, Image.LANCZOS)

    def update_view(self):
        self.canvas.delete("all")
        p = self.current_file()
        if p is not None:
            self.param_text.set(self.extract_params_from_filename(p.name))
        else:
            self.param_text.set("")

        remaining_total = len([q for q in self.all_files if q not in self.removed])
        remaining_view = len(self.files) - self.index
        if p is None:
            self.set_status(f"Done. No more images to review. (remaining total: {remaining_total}, view: 0)")
            return

        try:
            img = Image.open(p)
            img = img.convert("RGB")
            img = self.fit_image_to_canvas(img)
            self.photo = ImageTk.PhotoImage(img)
            cwidth = self.canvas.winfo_width() or self.args.window_width
            cheight = self.canvas.winfo_height() or self.args.window_height - 24
            x = (cwidth - img.width) // 2
            y = (cheight - img.height) // 2
            self.canvas.create_image(x, y, anchor="nw", image=self.photo)
        except Exception as e:
            self.photo = None
            self.canvas.create_text(10, 10, anchor="nw", fill="white", text=f"Failed to open image:\n{p}\n{e}")

        self.set_status(f"[{self.index+1}/{len(self.files)} | remaining {remaining_view}/{remaining_total}]\n{p.name}")

    # ---------- actions ----------
    def _remove_current_from_lists(self, path: Path):
        # remove from current view and remember globally so it won't show again
        self.removed.add(path)
        if 0 <= self.index < len(self.files) and self.files[self.index] == path:
            self.files.pop(self.index)
        # keep index in bounds (wrap handled in next/prev)
        if self.index >= len(self.files):
            self.index = max(0, len(self.files) - 1)

    def mark(self, label: Literal["good", "bad"]):
        p = self.current_file()
        if p is None:
            return

        dest_dir = self.args.good_dir if label == "good" else self.args.bad_dir
        dest_path = dest_dir / p.name

        try:
            if self.args.mode == "move":
                atomic_move(p, dest_path)
            elif self.args.mode == "copy":
                atomic_copy(p, dest_path)
            else:
                atomic_symlink(p, dest_path)
            append_log(self.args.log_path, label, p, dest_path, self.args.mode)
            self.history.append((label, p, dest_path))
            self._remove_current_from_lists(p)
        except Exception as e:
            append_log(self.args.log_path, "skip", p, None, self.args.mode)
            self.set_status(f"ERROR moving file: {e}. Marked as skipped in log.")
            # still advance past this one to avoid getting stuck
            self.history.append(("skip", p, None))
            self.index = (self.index + 1) % max(1, len(self.files))

        self.update_view()

    def skip(self):
        p = self.current_file()
        if p is not None:
            append_log(self.args.log_path, "skip", p, None, self.args.mode)
            self.history.append(("skip", p, None))
            # Do not remove from global pool on skip; just advance (wrap)
            if self.files:
                self.index = (self.index + 1) % len(self.files)
        self.update_view()

    def prev_image(self):
        if not self.files:
            return
        self.index = (self.index - 1) % len(self.files)
        self.update_view()

    def next_image(self):
        if not self.files:
            return
        self.index = (self.index + 1) % len(self.files)
        self.update_view()

    def undo(self):
        if not self.history:
            self.set_status("Nothing to undo.")
            return
        last_action, src, dest = self.history.pop()
        try:
            if last_action in ("good", "bad") and dest is not None and dest.exists():
                if self.args.mode == "move":
                    atomic_move(dest, src)
                elif self.args.mode == "copy":
                    dest.unlink(missing_ok=True)
                else:
                    dest.unlink(missing_ok=True)
                # put the file back into pools
                if src in self.removed:
                    self.removed.remove(src)
                # recompute filtered view and try to jump back to restored item
                self.files = self._apply_filter()
                try:
                    self.index = self.files.index(src)
                except ValueError:
                    self.index = min(self.index, max(0, len(self.files) - 1))
            elif last_action == "skip":
                # go back one item
                if self.files:
                    self.index = (self.index - 1) % len(self.files)
        except Exception as e:
            self.set_status(f"Undo failed: {e}")
        self.update_view()

    def quit(self):
        self.root.quit()

def main():
    args = parse_args()
    app = TriageApp(args)
    app.run()

if __name__ == "__main__":
    main()
