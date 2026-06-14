"""
IIT Analyzer — interfaz gráfica
Análisis de Teoría de Información Integrada (QNodes / GeoMIP Recursivo).
"""
import re
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from runner import (
    Runner, ExecutionParams, RunResult,
    parse_tpm_text, load_tpm_csv,
    count_excel_rows, load_excel_rows, generate_tpm,
    letras_a_binario, export_results,
)
from validator import (
    ValidationError,
    validate_tpm, validate_bit_fields, validate_ks,
    validate_n, validate_seed, validate_excel_range,
)

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# ── paleta ────────────────────────────────────────────────────────────────────
BG      = "#FFFFFF"
BG2     = "#F6F6F6"
FG      = "#0A0A0A"
FG2     = "#999999"
BORDER  = "#E2E2E2"
PRI     = "#0A0A0A"
PRI_HOV = "#2A2A2A"
SEC_HOV = "#F0F0F0"

# ── tipografía ────────────────────────────────────────────────────────────────
MONO    = ("Consolas",  11)
MONO_SM = ("Consolas",   9)
SANS    = ("Segoe UI",  11)
SANS_SM = ("Segoe UI",   9)
SANS_XS = ("Segoe UI",   8)
TITLE   = ("Segoe UI",  20, "bold")
CAP     = ("Segoe UI",   8, "bold")

PAD         = 44
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ─────────────────────────────────────────────────────────────────────────────
class App(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("IIT Analyzer")
        self.geometry("900x960")
        self.minsize(820, 780)
        self.configure(fg_color=BG)

        # state vars
        self.mode  = ctk.StringVar(value="manual")
        self.algo  = ctk.StringVar(value="geomip")
        self.k_sel = {k: ctk.BooleanVar(value=True) for k in [2, 3, 4, 5]}

        # N var lives here so the trace survives mode switches
        self._n_var = ctk.StringVar(value="10")
        self._n_var.trace_add("write", self._on_n_change)

        # runner
        self._runner      = Runner()
        self._last_result = None        # RunResult más reciente
        self._all_results = []          # acumulado para exportar
        self._cancelled   = False       # suprime cb_error tras cancel manual

        # excel session
        self._excel_path = None
        self._csv_path   = None
        self._excel_rows = []
        self._excel_idx  = -1
        self._excel_N    = 10
        self._excel_seed = 73
        self._excel_tpm  = None
        self._excel_cond = ""
        self._excel_est  = ""
        self._excel_algo = "geomip"
        self._excel_ks   = [2, 3, 4, 5]

        self._build()

    # ── construcción ─────────────────────────────────────────────────────────

    def _build(self):
        scroll = ctk.CTkScrollableFrame(
            self, fg_color=BG,
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=FG2,
        )
        scroll.pack(fill="both", expand=True)
        C = scroll

        self._header(C)
        self._rule(C)
        self._section_input(C)
        self._rule(C)
        self._section_config(C)
        self._rule(C)
        self._section_actions(C)
        self._rule(C)
        self._section_results(C)

    # ── cabecera ──────────────────────────────────────────────────────────────

    def _header(self, p):
        f = ctk.CTkFrame(p, fg_color=BG, corner_radius=0)
        f.pack(fill="x", padx=PAD, pady=(36, 22))
        ctk.CTkLabel(f, text="IIT ANALYZER", font=TITLE, text_color=FG).pack(anchor="w")
        ctk.CTkLabel(f, text="Integrated Information Theory",
                     font=SANS_SM, text_color=FG2).pack(anchor="w", pady=(2, 0))

    # ── utilidades de layout ──────────────────────────────────────────────────

    def _rule(self, p):
        ctk.CTkFrame(p, fg_color=BORDER, height=1, corner_radius=0
                     ).pack(fill="x", padx=PAD, pady=(4, 18))

    def _cap(self, p, txt):
        ctk.CTkLabel(p, text=txt, font=CAP, text_color=FG2
                     ).pack(anchor="w", padx=PAD, pady=(0, 10))

    def _row(self, p, bg=BG, pady=2):
        f = ctk.CTkFrame(p, fg_color=bg, corner_radius=0)
        f.pack(fill="x", padx=PAD, pady=pady)
        return f

    def _entry(self, parent, placeholder="", width=None, mono=True):
        kw = dict(
            fg_color=BG, text_color=FG, border_color=BORDER,
            font=MONO if mono else SANS_SM,
            corner_radius=4, height=32, border_width=1,
            placeholder_text=placeholder,
            placeholder_text_color=FG2,
        )
        if width:
            kw["width"] = width
        return ctk.CTkEntry(parent, **kw)

    def _btn(self, parent, text, cmd, pri=True, width=148, height=38):
        return ctk.CTkButton(
            parent, text=text, command=cmd, width=width, height=height,
            font=("Segoe UI", 10, "bold") if pri else SANS_SM,
            fg_color=PRI if pri else BG,
            hover_color=PRI_HOV if pri else SEC_HOV,
            text_color=BG if pri else FG,
            corner_radius=4,
            border_width=1 if not pri else 0,
            border_color=BORDER,
        )

    # ── sección MODO DE ENTRADA ───────────────────────────────────────────────

    def _section_input(self, p):
        self._cap(p, "MODO DE ENTRADA")

        radio_row = self._row(p, pady=(0, 14))
        for txt, val in [("Manual", "manual"), ("CSV", "csv"), ("Excel", "excel")]:
            ctk.CTkRadioButton(
                radio_row, text=txt, variable=self.mode, value=val,
                command=self._on_mode_change,
                font=SANS, text_color=FG,
                fg_color=PRI, hover_color=PRI_HOV, border_color=BORDER,
            ).pack(side="left", padx=(0, 32))

        self._panel = ctk.CTkFrame(
            p, fg_color=BG2, corner_radius=6,
            border_width=1, border_color=BORDER,
        )
        self._panel.pack(fill="x", padx=PAD, pady=(0, 6))
        self._render_manual()

    def _on_mode_change(self):
        {"manual": self._render_manual,
         "csv":    self._render_csv,
         "excel":  self._render_excel}[self.mode.get()]()
        self._sync_next()

    def _clear_panel(self):
        for w in self._panel.winfo_children():
            w.destroy()

    # ── Panel Manual ──────────────────────────────────────────────────────────

    def _render_manual(self):
        self._clear_panel()
        P = self._panel

        ctk.CTkLabel(
            P,
            text="TPM  ·  filas separadas por salto de línea, valores por espacio o coma",
            font=SANS_XS, text_color=FG2,
        ).pack(anchor="w", padx=16, pady=(14, 4))

        self.tpm_box = ctk.CTkTextbox(
            P, height=88, fg_color=BG, text_color=FG, font=MONO,
            border_width=1, border_color=BORDER, corner_radius=4,
            activate_scrollbars=True,
        )
        self.tpm_box.pack(fill="x", padx=16, pady=(0, 10))

        self._man_e = {}
        for lbl, key, ph in [
            ("Estado inicial", "estado",    "ej. 1000000000"),
            ("Condición",      "condicion", "ej. 1111111111"),
            ("Alcance",        "alcance",   "ej. 1111111111"),
            ("Mecanismo",      "mecanismo", "ej. 1111111111"),
        ]:
            r = ctk.CTkFrame(P, fg_color=BG2, corner_radius=0)
            r.pack(fill="x", padx=16, pady=3)
            ctk.CTkLabel(r, text=lbl, font=SANS_SM, text_color=FG2,
                         width=104, anchor="w").pack(side="left", padx=(0, 10))
            e = self._entry(r, placeholder=ph)
            e.pack(side="left", fill="x", expand=True)
            self._man_e[key] = e

        ctk.CTkFrame(P, fg_color=BG2, height=14, corner_radius=0).pack()

    # ── Panel CSV ─────────────────────────────────────────────────────────────

    def _render_csv(self):
        self._clear_panel()
        P = self._panel

        fr = ctk.CTkFrame(P, fg_color=BG2, corner_radius=0)
        fr.pack(fill="x", padx=16, pady=(14, 8))
        self._csv_lbl = ctk.CTkLabel(fr, text="Sin archivo seleccionado",
                                     font=SANS_SM, text_color=FG2)
        self._csv_lbl.pack(side="left")
        self._btn(fr, "Seleccionar CSV", self._pick_csv,
                  pri=False, width=130, height=30).pack(side="right")

        self._csv_info = ctk.CTkLabel(P, text="", font=SANS_XS, text_color=FG2)
        self._csv_info.pack(anchor="w", padx=16, pady=(0, 4))

        r = ctk.CTkFrame(P, fg_color=BG2, corner_radius=0)
        r.pack(fill="x", padx=16, pady=3)
        ctk.CTkLabel(r, text="Estado inicial", font=SANS_SM, text_color=FG2,
                     width=104, anchor="w").pack(side="left", padx=(0, 10))
        self._csv_estado = self._entry(r, placeholder="auto-rellenado al cargar archivo")
        self._csv_estado.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(
            P,
            text="Condición, alcance y mecanismo se usan como sistema completo (todos los nodos).",
            font=SANS_XS, text_color=FG2,
        ).pack(anchor="w", padx=16, pady=(6, 14))

    # ── Panel Excel ───────────────────────────────────────────────────────────

    def _render_excel(self):
        self._clear_panel()
        P = self._panel

        # archivo
        fr = ctk.CTkFrame(P, fg_color=BG2, corner_radius=0)
        fr.pack(fill="x", padx=16, pady=(14, 8))
        self._xls_lbl = ctk.CTkLabel(fr, text="Sin archivo seleccionado",
                                     font=SANS_SM, text_color=FG2)
        self._xls_lbl.pack(side="left")
        self._btn(fr, "Seleccionar Excel", self._pick_excel,
                  pri=False, width=140, height=30).pack(side="right")

        # hoja
        r = ctk.CTkFrame(P, fg_color=BG2, corner_radius=0)
        r.pack(fill="x", padx=16, pady=3)
        ctk.CTkLabel(r, text="Hoja", font=SANS_SM, text_color=FG2,
                     width=104, anchor="w").pack(side="left", padx=(0, 10))
        self._sheet_var = ctk.StringVar(value="—")
        self._sheet_om = ctk.CTkOptionMenu(
            r, variable=self._sheet_var, values=["—"],
            font=SANS_SM, fg_color=BG, text_color=FG,
            button_color=BORDER, button_hover_color=FG2,
            dropdown_fg_color=BG, dropdown_text_color=FG,
            height=32, width=220,
            command=self._on_sheet_change,
        )
        self._sheet_om.pack(side="left")

        # rango de filas
        r2 = ctk.CTkFrame(P, fg_color=BG2, corner_radius=0)
        r2.pack(fill="x", padx=16, pady=3)
        ctk.CTkLabel(r2, text="Rango de filas", font=SANS_SM, text_color=FG2,
                     width=104, anchor="w").pack(side="left", padx=(0, 10))
        self._xls_desde = self._entry(r2, placeholder="inicio", width=72)
        self._xls_desde.pack(side="left")
        self._xls_desde.insert(0, "1")
        ctk.CTkLabel(r2, text="—", font=SANS_SM, text_color=FG2
                     ).pack(side="left", padx=8)
        self._xls_hasta = self._entry(r2, placeholder="fin", width=72)
        self._xls_hasta.pack(side="left")

        # N nodos
        r3 = ctk.CTkFrame(P, fg_color=BG2, corner_radius=0)
        r3.pack(fill="x", padx=16, pady=3)
        ctk.CTkLabel(r3, text="N  (nodos)", font=SANS_SM, text_color=FG2,
                     width=104, anchor="w").pack(side="left", padx=(0, 10))
        self._n_entry = ctk.CTkEntry(
            r3, textvariable=self._n_var, fg_color=BG, text_color=FG,
            border_color=BORDER, font=MONO, corner_radius=4,
            height=32, border_width=1, width=72,
        )
        self._n_entry.pack(side="left")
        ctk.CTkLabel(r3, text="Auto-detectado desde la hoja",
                     font=SANS_XS, text_color=FG2).pack(side="left", padx=(10, 0))

        # semilla
        r4 = ctk.CTkFrame(P, fg_color=BG2, corner_radius=0)
        r4.pack(fill="x", padx=16, pady=3)
        ctk.CTkLabel(r4, text="Semilla", font=SANS_SM, text_color=FG2,
                     width=104, anchor="w").pack(side="left", padx=(0, 10))
        self._seed_e = self._entry(r4, placeholder="73", width=72)
        self._seed_e.pack(side="left")
        self._seed_e.insert(0, "73")

        # condición
        r5 = ctk.CTkFrame(P, fg_color=BG2, corner_radius=0)
        r5.pack(fill="x", padx=16, pady=3)
        ctk.CTkLabel(r5, text="Condición", font=SANS_SM, text_color=FG2,
                     width=104, anchor="w").pack(side="left", padx=(0, 10))
        self._xls_condicion = self._entry(r5, placeholder="cadena de bits")
        self._xls_condicion.pack(side="left", fill="x", expand=True)

        # estado inicial
        r6 = ctk.CTkFrame(P, fg_color=BG2, corner_radius=0)
        r6.pack(fill="x", padx=16, pady=3)
        ctk.CTkLabel(r6, text="Estado inicial", font=SANS_SM, text_color=FG2,
                     width=104, anchor="w").pack(side="left", padx=(0, 10))
        self._xls_estado = self._entry(r6, placeholder="cadena de bits")
        self._xls_estado.pack(side="left", fill="x", expand=True)

        # pre-fill defaults con N actual
        self._fill_excel_defaults()

        # indicador de progreso
        self._xls_prog = ctk.CTkLabel(P, text="", font=SANS_XS, text_color=FG2)
        self._xls_prog.pack(anchor="w", padx=16, pady=(6, 14))

    def _fill_excel_defaults(self):
        """Rellena condición y estado inicial según N actual."""
        try:
            n = int(self._n_var.get())
            if not 2 <= n <= 25:
                return
        except ValueError:
            return
        if hasattr(self, "_xls_condicion"):
            cur = self._xls_condicion.get()
            if not cur or set(cur) <= {"0", "1"} and len(cur) != n:
                self._xls_condicion.delete(0, "end")
                self._xls_condicion.insert(0, "1" * n)
        if hasattr(self, "_xls_estado"):
            cur = self._xls_estado.get()
            if not cur or set(cur) <= {"0", "1"} and len(cur) != n:
                self._xls_estado.delete(0, "end")
                self._xls_estado.insert(0, "1" + "0" * (n - 1))

    def _on_sheet_change(self, sheet):
        m = re.match(r"(\d+)", sheet)
        if m:
            self._n_var.set(m.group(1))

    def _on_n_change(self, *_):
        if self.mode.get() == "excel":
            self._fill_excel_defaults()

    # ── sección CONFIG ────────────────────────────────────────────────────────

    def _section_config(self, p):
        outer = ctk.CTkFrame(p, fg_color=BG, corner_radius=0)
        outer.pack(fill="x", padx=PAD, pady=(0, 18))

        left = ctk.CTkFrame(outer, fg_color=BG, corner_radius=0)
        left.pack(side="left", anchor="n")
        ctk.CTkLabel(left, text="ALGORITMO", font=CAP, text_color=FG2
                     ).pack(anchor="w", pady=(0, 10))
        for txt, val in [("QNodes", "qnodes"), ("GeoMIP Recursivo", "geomip")]:
            ctk.CTkRadioButton(
                left, text=txt, variable=self.algo, value=val,
                font=SANS, text_color=FG,
                fg_color=PRI, hover_color=PRI_HOV, border_color=BORDER,
            ).pack(anchor="w", pady=4)

        right = ctk.CTkFrame(outer, fg_color=BG, corner_radius=0)
        right.pack(side="left", padx=(72, 0), anchor="n")
        ctk.CTkLabel(right, text="K-PARTICIONES", font=CAP, text_color=FG2
                     ).pack(anchor="w", pady=(0, 10))
        kr = ctk.CTkFrame(right, fg_color=BG, corner_radius=0)
        kr.pack(anchor="w")
        for k in [2, 3, 4, 5]:
            ctk.CTkCheckBox(
                kr, text=f"k = {k}", variable=self.k_sel[k],
                font=SANS, text_color=FG,
                fg_color=PRI, hover_color=PRI_HOV, border_color=BORDER,
                checkmark_color=BG,
            ).pack(side="left", padx=(0, 24))

    # ── sección ACCIONES ──────────────────────────────────────────────────────

    def _section_actions(self, p):
        row = ctk.CTkFrame(p, fg_color=BG, corner_radius=0)
        row.pack(fill="x", padx=PAD, pady=(0, 6))

        self._btn_run = self._btn(row, "EJECUTAR", self._on_run, pri=True)
        self._btn_run.pack(side="left", padx=(0, 12))

        self._btn_cancel = self._btn(row, "CANCELAR", self._on_cancel,
                                     pri=False, width=120, height=38)
        self._btn_cancel.pack(side="left", padx=(0, 12))
        self._btn_cancel.configure(state="disabled")

        self._btn_next = self._btn(row, "SIGUIENTE  ›", self._on_next,
                                   pri=False, width=148, height=38)
        self._btn_next.pack(side="left")
        self._btn_next.configure(state="disabled")

        self._timer_lbl = ctk.CTkLabel(row, text="", font=SANS_SM, text_color=FG2)
        self._timer_lbl.pack(side="right")

    # ── sección RESULTADOS ────────────────────────────────────────────────────

    def _section_results(self, p):
        self._cap(p, "RESULTADOS")

        box = ctk.CTkFrame(
            p, fg_color=BG2, corner_radius=6,
            border_width=1, border_color=BORDER,
        )
        box.pack(fill="both", expand=True, padx=PAD, pady=(0, 8))

        self._res = ctk.CTkTextbox(
            box, fg_color=BG2, text_color=FG, font=MONO,
            border_width=0, activate_scrollbars=True, height=280,
        )
        self._res.pack(fill="both", expand=True, padx=16, pady=16)
        self._set_result(
            "Los resultados aparecerán aquí una vez ejecutado el análisis.\n\n"
            "  Partición   —   distribución de nodos por parte\n"
            "  φ (phi)     —   pérdida de información integrada\n"
            "  Tiempo      —   duración de la ejecución por k"
        )

        self._btn_save = self._btn(p, "GUARDAR RESULTADOS", self._on_save,
                                   pri=False, width=190, height=34)
        self._btn_save.pack(anchor="e", padx=PAD, pady=(0, 40))
        self._btn_save.configure(state="disabled")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _set_result(self, text: str):
        self._res.configure(state="normal")
        self._res.delete("1.0", "end")
        self._res.insert("end", text)
        self._res.configure(state="disabled")

    def _sync_next(self):
        can = (
            self.mode.get() == "excel"
            and len(self._excel_rows) > 0
            and self._excel_idx < len(self._excel_rows) - 1
            and not self._runner.running
        )
        self._btn_next.configure(state="normal" if can else "disabled")

    def _set_running_state(self, running: bool):
        if running:
            self._btn_run.configure(state="disabled")
            self._btn_cancel.configure(state="normal")
            self._btn_next.configure(state="disabled")
            self._btn_save.configure(state="disabled")
        else:
            self._btn_run.configure(state="normal")
            self._btn_cancel.configure(state="disabled")
            self._sync_next()
            if self._all_results:
                self._btn_save.configure(state="normal")

    def _show_errors(self, errors: list):
        lines = ["ERRORES DE VALIDACIÓN\n"]
        for e in errors:
            lines.append(f"  {e.field}:  {e.message}")
        self._set_result("\n".join(lines))

    # ── file pickers ──────────────────────────────────────────────────────────

    def _pick_csv(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not p:
            return
        self._csv_path = p
        self._csv_lbl.configure(text=Path(p).name)
        try:
            import numpy as np
            arr = np.genfromtxt(p, delimiter=",")
            if arr.ndim == 2:
                rows, cols = arr.shape
                if hasattr(self, "_csv_info"):
                    self._csv_info.configure(
                        text=f"N = {cols}  ·  {rows} estados  ·  {cols} nodos")
                if hasattr(self, "_csv_estado"):
                    self._csv_estado.delete(0, "end")
                    self._csv_estado.insert(0, "1" + "0" * (cols - 1))
        except Exception:
            pass

    def _pick_excel(self):
        p = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx *.xls")])
        if p:
            self._excel_path = p
            self._xls_lbl.configure(text=Path(p).name)
            try:
                import pandas as pd
                sheets = pd.ExcelFile(p).sheet_names
                self._sheet_om.configure(values=sheets)
                self._sheet_var.set(sheets[0])
                self._on_sheet_change(sheets[0])
            except Exception:
                pass

    # ── recolección de parámetros ─────────────────────────────────────────────

    def _collect_manual(self, ks: list, algo: str):
        errors = []
        try:
            tpm = parse_tpm_text(self.tpm_box.get("1.0", "end"))
        except ValueError as exc:
            return None, [ValidationError("TPM", str(exc))]

        errors.extend(validate_tpm(tpm))
        if errors:
            return None, errors

        N = tpm.shape[1]
        fields = {k: self._man_e[k].get().strip()
                  for k in ["estado", "condicion", "alcance", "mecanismo"]}
        errors.extend(validate_bit_fields(fields, N))
        errors.extend(validate_ks(ks))
        if errors:
            return None, errors

        return ExecutionParams(
            algo=algo, tpm=tpm,
            estado_inicial=fields["estado"],
            condicion=fields["condicion"],
            alcance=fields["alcance"],
            mecanismo=fields["mecanismo"],
            ks=ks,
        ), []

    def _collect_csv(self, ks: list, algo: str):
        if not self._csv_path:
            return None, [ValidationError("CSV", "No hay ningún archivo seleccionado.")]

        try:
            tpm = load_tpm_csv(self._csv_path)
        except ValueError as exc:
            return None, [ValidationError("CSV", str(exc))]

        errors = list(validate_tpm(tpm))
        if errors:
            return None, errors

        N = tpm.shape[1]

        # estado inicial: del campo si fue editado, si no el default
        estado = getattr(self, "_csv_estado", None)
        estado = estado.get().strip() if estado else ""
        if not estado:
            estado = "1" + "0" * (N - 1)

        errors.extend(validate_bit_fields({"estado": estado}, N))
        errors.extend(validate_ks(ks))
        if errors:
            return None, errors

        all_ones = "1" * N
        return ExecutionParams(
            algo=algo, tpm=tpm,
            estado_inicial=estado,
            condicion=all_ones,
            alcance=all_ones,
            mecanismo=all_ones,
            ks=ks,
        ), []

    def _start_excel(self, ks: list, algo: str):
        """Carga las filas del Excel y prepara params para la primera."""
        errors = []
        if not self._excel_path:
            return None, [ValidationError("Excel", "No hay ningún archivo seleccionado.")]

        N, n_errs    = validate_n(self._n_var.get())
        seed, s_errs = validate_seed(self._seed_e.get().strip())
        errors.extend(n_errs)
        errors.extend(s_errs)
        errors.extend(validate_ks(ks))

        condicion = self._xls_condicion.get().strip()
        estado    = self._xls_estado.get().strip()
        errors.extend(validate_bit_fields({"condicion": condicion, "estado": estado}, N))

        if errors:
            return None, errors

        try:
            total = count_excel_rows(self._excel_path, self._sheet_var.get())
        except Exception as exc:
            return None, [ValidationError("Excel", f"No se pudo leer el archivo: {exc}")]

        d, h, r_errs = validate_excel_range(
            self._xls_desde.get().strip(),
            self._xls_hasta.get().strip(),
            total,
        )
        errors.extend(r_errs)
        if errors:
            return None, errors

        try:
            self._excel_rows = load_excel_rows(
                self._excel_path, self._sheet_var.get(), d, h)
        except Exception as exc:
            return None, [ValidationError("Excel", str(exc))]

        if not self._excel_rows:
            return None, [ValidationError("Rango de filas",
                "No se encontraron pruebas en el rango indicado.")]

        # guardar estado de la sesión Excel
        self._excel_idx  = 0
        self._excel_N    = N
        self._excel_seed = seed
        self._excel_tpm  = generate_tpm(N, seed)
        self._excel_cond = condicion
        self._excel_est  = estado
        self._excel_algo = algo
        self._excel_ks   = ks

        return self._make_excel_params(self._excel_rows[0]), []

    def _make_excel_params(self, row: dict) -> ExecutionParams:
        n = self._excel_N
        return ExecutionParams(
            algo=self._excel_algo,
            tpm=self._excel_tpm,
            estado_inicial=self._excel_est,
            condicion=self._excel_cond,
            alcance=letras_a_binario(row["alcance"], n),
            mecanismo=letras_a_binario(row["mecanismo"], n),
            ks=self._excel_ks,
            semilla=self._excel_seed,
            prueba_num=row["prueba"],
            alcance_label=row["alcance"],
            mecanismo_label=row["mecanismo"],
        )

    # ── ejecución ─────────────────────────────────────────────────────────────

    def _on_run(self):
        if self._runner.running:
            return
        self._cancelled = False

        mode = self.mode.get()
        ks   = [k for k, v in self.k_sel.items() if v.get()]
        algo = self.algo.get()

        if mode == "manual":
            params, errors = self._collect_manual(ks, algo)
        elif mode == "csv":
            params, errors = self._collect_csv(ks, algo)
        else:
            # Excel: también carga las filas y resetea la sesión
            self._excel_rows = []
            self._excel_idx  = -1
            params, errors = self._start_excel(ks, algo)

        if errors:
            self._show_errors(errors)
            return

        self._set_running_state(True)
        self._timer_lbl.configure(text="")
        self._set_result("Ejecutando...")

        if mode == "excel" and hasattr(self, "_xls_prog"):
            total = len(self._excel_rows)
            self._xls_prog.configure(text=f"Prueba {self._excel_idx + 1} de {total}")

        self._runner.execute(
            params,
            on_tick   = lambda t: self.after(0, self._cb_tick, t),
            on_result = lambda r: self.after(0, self._cb_result, r),
            on_error  = lambda m: self.after(0, self._cb_error, m),
        )

    def _on_next(self):
        if self._runner.running or self._excel_idx >= len(self._excel_rows) - 1:
            return
        self._cancelled = False
        self._excel_idx += 1
        params = self._make_excel_params(self._excel_rows[self._excel_idx])

        self._set_running_state(True)
        self._timer_lbl.configure(text="")
        self._set_result("Ejecutando...")

        if hasattr(self, "_xls_prog"):
            total = len(self._excel_rows)
            self._xls_prog.configure(
                text=f"Prueba {self._excel_idx + 1} de {total}")

        self._runner.execute(
            params,
            on_tick   = lambda t: self.after(0, self._cb_tick, t),
            on_result = lambda r: self.after(0, self._cb_result, r),
            on_error  = lambda m: self.after(0, self._cb_error, m),
        )

    def _on_cancel(self):
        self._cancelled = True
        self._runner.cancel()
        self._set_result("Ejecución cancelada por el usuario.")
        self._timer_lbl.configure(text="")
        self._set_running_state(False)

    # ── callbacks (siempre llamados via self.after para thread-safety) ─────────

    def _cb_tick(self, elapsed: float):
        self._timer_lbl.configure(text=f"{elapsed:.1f} s")

    def _cb_result(self, result: RunResult):
        if self._cancelled:
            return
        self._last_result = result
        self._all_results.append(result)
        self._set_result(self._format_result(result))
        self._timer_lbl.configure(text=f"{result.elapsed:.2f} s  ·  completado")
        self._set_running_state(False)
        if self.mode.get() == "excel" and hasattr(self, "_xls_prog"):
            remaining = len(self._excel_rows) - self._excel_idx - 1
            total     = len(self._excel_rows)
            sfx = (f"  ·  {remaining} pendiente{'s' if remaining != 1 else ''}"
                   if remaining > 0 else "  ·  completado")
            self._xls_prog.configure(
                text=f"Prueba {self._excel_idx + 1} de {total}{sfx}")

    def _cb_error(self, message: str):
        if self._cancelled:
            self._cancelled = False
            return
        self._set_result(f"ERROR\n\n  {message}")
        self._timer_lbl.configure(text="error")
        self._set_running_state(False)

    # ── formateo de resultados ────────────────────────────────────────────────

    def _format_result(self, result: RunResult) -> str:
        p     = result.params
        algo  = "GEOMIP RECURSIVO" if p.algo == "geomip" else "QNODES"
        lines = [
            f"PRUEBA #{p.prueba_num}  ·  alc: {p.alcance_label}  ·  mec: {p.mecanismo_label}",
            f"ALGORITMO: {algo}  ·  tiempo total: {result.elapsed:.2f} s",
            "─" * 60,
            "",
        ]
        for k_str in sorted(result.raw.keys(), key=int):
            data = result.raw[k_str]
            phi  = data.get("perdida", 0.0)
            t    = data.get("tiempo", 0.0)
            part = data.get("particion", "—")
            lines.append(f"k = {k_str}   φ = {phi:.6f}   ({t:.2f} s)")
            for sub in part.splitlines():
                lines.append(f"    {sub}")
            lines.append("")
        return "\n".join(lines)

    # ── exportar ─────────────────────────────────────────────────────────────

    def _on_save(self):
        if not self._all_results:
            return
        RESULTS_DIR.mkdir(exist_ok=True)
        default = f"gui_{int(time.time())}.xlsx"
        path = filedialog.asksaveasfilename(
            initialdir=str(RESULTS_DIR),
            initialfile=default,
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")],
        )
        if not path:
            return
        try:
            export_results(self._all_results, path)
            messagebox.showinfo("Guardado", f"Resultados guardados en:\n{path}")
        except Exception as exc:
            messagebox.showerror("Error al guardar", str(exc))


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
