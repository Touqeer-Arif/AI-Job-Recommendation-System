"""
main_app.py
-----------
Tkinter desktop GUI for the AI Job Recommendation System.

This is the DEPLOYMENT layer only. It contains no training logic and no
data-cleaning logic of its own -- it imports the reusable modules from src/
(preprocess, features, recommend, ats), keeping interface code separate
from program logic, as required by the course project policy.

Run with:
    python app/main_app.py

Requires that notebooks/job_recommender_training.ipynb has already been run
once in Google Colab and that the resulting models/ folder (job_role_classifier
.joblib, salary_regressor.joblib, skill_vectorizer.joblib, label_encoders.joblib)
has been placed in the project's models/ directory.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Make project root importable (so `from src import recommend` works
# regardless of the directory this script is launched from)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)  # config.json and models/ are resolved relative to project root

from src.recommend import JobRecommender


EDUCATION_LEVELS = ["high school", "bachelor", "master", "phd"]

# ----------------------------------------------------------------------
# Design tokens — one deliberate palette, used consistently everywhere.
# Navy + warm off-white canvas, single teal accent for primary actions,
# and a three-tier traffic-light scale reserved ONLY for ATS scores (so
# color always means the same thing wherever it appears in the app).
# ----------------------------------------------------------------------
COLOR_BG = "#f4f5f7"
COLOR_SIDEBAR = "#16213a"
COLOR_SIDEBAR_TEXT = "#e7eaf0"
COLOR_SIDEBAR_SUBTEXT = "#8d95ab"
COLOR_CARD = "#ffffff"
COLOR_BORDER = "#e3e5ea"
COLOR_TEXT = "#1c2130"
COLOR_TEXT_MUTED = "#6b7184"
COLOR_ACCENT = "#0d9488"
COLOR_ACCENT_DARK = "#0b7d73"
COLOR_SCORE_HIGH = "#0d9488"
COLOR_SCORE_MID = "#d97706"
COLOR_SCORE_LOW = "#dc2626"
COLOR_CHIP_BG = "#eef2f5"

FONT_FAMILY = "Segoe UI"


def score_color(score: float) -> str:
    if score >= 65:
        return COLOR_SCORE_HIGH
    elif score >= 35:
        return COLOR_SCORE_MID
    return COLOR_SCORE_LOW


class JobRecommenderApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Job Recommendation System")
        self.geometry("1080x720")
        self.minsize(900, 600)
        self.configure(bg=COLOR_BG)

        self.cv_path = None
        self.recommender = None
        self._card_widgets = []  # track dynamically built result cards for cleanup

        self._build_styles()
        self._build_layout()
        self._load_model_async()

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    def _build_styles(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Sidebar.TFrame", background=COLOR_SIDEBAR)
        style.configure("Canvas.TFrame", background=COLOR_BG)
        style.configure("Card.TFrame", background=COLOR_CARD)

        style.configure("Brand.TLabel", background=COLOR_SIDEBAR, foreground=COLOR_SIDEBAR_TEXT,
                         font=(FONT_FAMILY, 16, "bold"))
        style.configure("BrandSub.TLabel", background=COLOR_SIDEBAR, foreground=COLOR_SIDEBAR_SUBTEXT,
                         font=(FONT_FAMILY, 9), wraplength=210, justify="left")
        style.configure("SidebarSection.TLabel", background=COLOR_SIDEBAR, foreground=COLOR_SIDEBAR_TEXT,
                         font=(FONT_FAMILY, 10, "bold"))
        style.configure("SidebarText.TLabel", background=COLOR_SIDEBAR, foreground=COLOR_SIDEBAR_SUBTEXT,
                         font=(FONT_FAMILY, 9))
        style.configure("SidebarStatus.TLabel", background=COLOR_SIDEBAR, foreground="#f0c36d",
                         font=(FONT_FAMILY, 9), wraplength=210, justify="left")

        style.configure("PageTitle.TLabel", background=COLOR_BG, foreground=COLOR_TEXT,
                         font=(FONT_FAMILY, 20, "bold"))
        style.configure("PageSub.TLabel", background=COLOR_BG, foreground=COLOR_TEXT_MUTED,
                         font=(FONT_FAMILY, 10))

        style.configure("CardTitle.TLabel", background=COLOR_CARD, foreground=COLOR_TEXT,
                         font=(FONT_FAMILY, 11, "bold"))
        style.configure("CardText.TLabel", background=COLOR_CARD, foreground=COLOR_TEXT,
                         font=(FONT_FAMILY, 10))
        style.configure("CardMuted.TLabel", background=COLOR_CARD, foreground=COLOR_TEXT_MUTED,
                         font=(FONT_FAMILY, 9))

        style.configure("TEntry", padding=6)
        style.configure("TCombobox", padding=4)

        style.configure("Accent.TButton", font=(FONT_FAMILY, 11, "bold"), padding=10,
                         background=COLOR_ACCENT, foreground="#ffffff", borderwidth=0)
        style.map("Accent.TButton",
                  background=[("active", COLOR_ACCENT_DARK), ("disabled", "#a9c9c5")],
                  foreground=[("disabled", "#f0f0f0")])

        style.configure("Ghost.TButton", font=(FONT_FAMILY, 9, "bold"), padding=6,
                         background=COLOR_CHIP_BG, foreground=COLOR_TEXT, borderwidth=0)
        style.map("Ghost.TButton", background=[("active", "#e1e5ea")])

        style.configure("Horizontal.TProgressbar", background=COLOR_ACCENT,
                         troughcolor=COLOR_CHIP_BG, borderwidth=0)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self):
        # === Sidebar (left, fixed width) ===
        sidebar = ttk.Frame(self, style="Sidebar.TFrame", width=240)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        brand_frame = ttk.Frame(sidebar, style="Sidebar.TFrame", padding=(22, 28, 22, 16))
        brand_frame.pack(fill="x")
        ttk.Label(brand_frame, text="JobMatch AI", style="Brand.TLabel").pack(anchor="w")
        ttk.Label(brand_frame, text="CV-based job role & salary recommender",
                  style="BrandSub.TLabel").pack(anchor="w", pady=(6, 0))

        ttk.Frame(sidebar, style="Sidebar.TFrame", height=1).pack(fill="x", pady=10)

        howto_frame = ttk.Frame(sidebar, style="Sidebar.TFrame", padding=(22, 4, 22, 4))
        howto_frame.pack(fill="x")
        ttk.Label(howto_frame, text="HOW IT WORKS", style="SidebarSection.TLabel").pack(anchor="w")
        for i, step in enumerate([
            "Upload your CV (PDF, DOCX or TXT)",
            "Set education & project count",
            "Click Analyze",
            "Review ranked roles, ATS scores\n& estimated salary",
        ], start=1):
            step_frame = ttk.Frame(sidebar, style="Sidebar.TFrame", padding=(22, 6, 22, 0))
            step_frame.pack(fill="x")
            ttk.Label(step_frame, text=f"{i}.", style="SidebarText.TLabel",
                      foreground=COLOR_ACCENT_DARK if False else "#3fc1b0").pack(side="left", anchor="n")
            ttk.Label(step_frame, text=step, style="SidebarText.TLabel",
                      wraplength=190, justify="left").pack(side="left", padx=(6, 0))

        ttk.Frame(sidebar, style="Sidebar.TFrame").pack(fill="both", expand=True)

        status_frame = ttk.Frame(sidebar, style="Sidebar.TFrame", padding=(22, 10, 22, 24))
        status_frame.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="Loading trained models...")
        ttk.Label(status_frame, textvariable=self.status_var, style="SidebarStatus.TLabel").pack(anchor="w")

        # === Main canvas (right) ===
        main = ttk.Frame(self, style="Canvas.TFrame", padding=(28, 24, 28, 24))
        main.pack(side="left", fill="both", expand=True)

        ttk.Label(main, text="Analyze a CV", style="PageTitle.TLabel").pack(anchor="w")
        ttk.Label(main, text="Get ranked job recommendations, ATS-style match scores, "
                              "and a salary estimate.",
                  style="PageSub.TLabel").pack(anchor="w", pady=(2, 16))

        # --- Input card ---
        input_card = ttk.Frame(main, style="Card.TFrame", padding=18)
        input_card.pack(fill="x")
        self._add_card_border(input_card)

        ttk.Label(input_card, text="CV file", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", columnspan=3)

        self.file_label_var = tk.StringVar(value="No file selected")
        ttk.Label(input_card, textvariable=self.file_label_var, style="CardMuted.TLabel").grid(
            row=1, column=0, sticky="w", pady=(6, 14), columnspan=2)
        ttk.Button(input_card, text="Browse...", style="Ghost.TButton",
                   command=self.browse_file).grid(row=1, column=2, sticky="e", pady=(6, 14))

        ttk.Label(input_card, text="Education level", style="CardMuted.TLabel").grid(
            row=2, column=0, sticky="w")
        ttk.Label(input_card, text="Completed projects", style="CardMuted.TLabel").grid(
            row=2, column=1, sticky="w", padx=(16, 0))

        self.education_var = tk.StringVar(value="bachelor")
        ttk.Combobox(input_card, textvariable=self.education_var, values=EDUCATION_LEVELS,
                     state="readonly", width=18).grid(row=3, column=0, sticky="w", pady=(4, 0))

        self.projects_var = tk.StringVar(value="0")
        ttk.Entry(input_card, textvariable=self.projects_var, width=10).grid(
            row=3, column=1, sticky="w", padx=(16, 0), pady=(4, 0))

        self.analyze_btn = ttk.Button(input_card, text="Analyze CV", style="Accent.TButton",
                                       command=self.run_recommendation, state="disabled")
        self.analyze_btn.grid(row=3, column=2, sticky="e", pady=(4, 0))

        input_card.columnconfigure(0, weight=1)
        input_card.columnconfigure(1, weight=1)
        input_card.columnconfigure(2, weight=1)

        self.progress = ttk.Progressbar(main, mode="indeterminate", style="Horizontal.TProgressbar")
        # packed/unpacked dynamically during inference

        # --- Results area (scrollable) ---
        results_outer = ttk.Frame(main, style="Canvas.TFrame")
        results_outer.pack(fill="both", expand=True, pady=(16, 0))

        self.results_canvas = tk.Canvas(results_outer, bg=COLOR_BG, highlightthickness=0)
        results_scrollbar = ttk.Scrollbar(results_outer, orient="vertical", command=self.results_canvas.yview)
        self.results_inner = ttk.Frame(self.results_canvas, style="Canvas.TFrame")

        self.results_inner.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        )
        self._results_window = self.results_canvas.create_window((0, 0), window=self.results_inner, anchor="nw")
        self.results_canvas.bind(
            "<Configure>",
            lambda e: self.results_canvas.itemconfig(self._results_window, width=e.width)
        )
        self.results_canvas.configure(yscrollcommand=results_scrollbar.set)

        self.results_canvas.pack(side="left", fill="both", expand=True)
        results_scrollbar.pack(side="right", fill="y")

        # mousewheel scrolling
        self.results_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self._show_placeholder()

    def _add_card_border(self, frame):
        frame.configure(relief="solid", borderwidth=1)

    def _on_mousewheel(self, event):
        self.results_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _show_placeholder(self):
        self._clear_results()
        placeholder = ttk.Frame(self.results_inner, style="Canvas.TFrame", padding=(0, 40, 0, 0))
        placeholder.pack(fill="x")
        ttk.Label(placeholder, text="Upload a CV and click Analyze to see recommendations here.",
                  style="PageSub.TLabel").pack()
        self._card_widgets.append(placeholder)

    def _clear_results(self):
        for w in self._card_widgets:
            w.destroy()
        self._card_widgets = []

    # ------------------------------------------------------------------
    # Model loading (happens once, in a background thread, on startup)
    # ------------------------------------------------------------------
    def _load_model_async(self):
        def load():
            try:
                self.recommender = JobRecommender()
                self.status_var.set("● Models loaded — ready to analyze")
                self.analyze_btn.configure(state="normal")
            except FileNotFoundError:
                self.status_var.set(
                    "⚠ Trained model files not found in models/. "
                    "Run the Colab training notebook first and copy the "
                    "models/ folder here."
                )
            except Exception as e:
                self.status_var.set(f"⚠ Error loading models: {e}")

        threading.Thread(target=load, daemon=True).start()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Select your CV",
            filetypes=[("CV documents", "*.pdf *.docx *.txt"), ("All files", "*.*")],
        )
        if path:
            self.cv_path = path
            self.file_label_var.set(os.path.basename(path))

    def run_recommendation(self):
        if not self.cv_path:
            messagebox.showwarning("No CV selected", "Please upload a CV file first.")
            return
        if self.recommender is None:
            messagebox.showerror("Model not loaded", "Trained models are not loaded yet.")
            return

        try:
            projects_count = float(self.projects_var.get() or 0)
        except ValueError:
            messagebox.showerror("Invalid input", "Completed projects must be a number.")
            return

        self.analyze_btn.configure(state="disabled")
        self.progress.pack(fill="x", pady=(10, 0))
        self.progress.start(12)
        self._clear_results()
        loading = ttk.Frame(self.results_inner, style="Canvas.TFrame", padding=(0, 30, 0, 0))
        loading.pack(fill="x")
        ttk.Label(loading, text="Analyzing CV...", style="PageSub.TLabel").pack()
        self._card_widgets.append(loading)

        def work():
            try:
                result = self.recommender.recommend_from_file(
                    filepath=self.cv_path,
                    projects_count=projects_count,
                    education=self.education_var.get(),
                    top_n=3,
                )
                self.after(0, lambda: self._display_results(result))
            except Exception as e:
                self.after(0, lambda: self._display_error(str(e)))
            finally:
                self.after(0, self._finish_inference)

        threading.Thread(target=work, daemon=True).start()

    def _finish_inference(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.analyze_btn.configure(state="normal")

    def _display_error(self, message):
        self._clear_results()
        err_card = ttk.Frame(self.results_inner, style="Card.TFrame", padding=18)
        err_card.pack(fill="x", pady=(0, 4))
        self._add_card_border(err_card)
        ttk.Label(err_card, text="Could not generate recommendations", style="CardTitle.TLabel",
                  foreground=COLOR_SCORE_LOW).pack(anchor="w")
        ttk.Label(err_card, text=message, style="CardText.TLabel", wraplength=700,
                  justify="left").pack(anchor="w", pady=(6, 0))
        self._card_widgets.append(err_card)

    # ------------------------------------------------------------------
    # Results rendering
    # ------------------------------------------------------------------
    def _display_results(self, result: dict):
        self._clear_results()

        # --- Summary card ---
        summary = ttk.Frame(self.results_inner, style="Card.TFrame", padding=18)
        summary.pack(fill="x", pady=(0, 12))
        self._add_card_border(summary)

        ttk.Label(summary, text="CV Summary", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", columnspan=3)

        ttk.Label(summary, text="Detected experience", style="CardMuted.TLabel").grid(
            row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Label(summary, text=f"{result['experience_years_detected']:.1f} years",
                  style="CardText.TLabel", font=(FONT_FAMILY, 13, "bold")).grid(
            row=2, column=0, sticky="w")

        ttk.Label(summary, text="Estimated salary (top match)", style="CardMuted.TLabel").grid(
            row=1, column=1, sticky="w", pady=(10, 0), padx=(24, 0))
        ttk.Label(summary, text=f"${result['predicted_salary']:,.0f}",
                  style="CardText.TLabel", font=(FONT_FAMILY, 13, "bold"),
                  foreground=COLOR_ACCENT_DARK).grid(row=2, column=1, sticky="w", padx=(24, 0))

        ttk.Label(summary, text="Skills detected", style="CardMuted.TLabel").grid(
            row=3, column=0, sticky="w", pady=(14, 4), columnspan=3)
        chips_frame = ttk.Frame(summary, style="Card.TFrame")
        chips_frame.grid(row=4, column=0, columnspan=3, sticky="w")
        self._render_chips(chips_frame, result["matched_skills"])

        self._card_widgets.append(summary)

        # --- One match card per recommended role ---
        for i, role in enumerate(result["top_roles"], start=1):
            self._card_widgets.append(self._build_role_card(i, role))

    def _render_chips(self, parent, skills):
        row, col = 0, 0
        max_cols = 5
        for skill in skills:
            chip = tk.Label(parent, text=skill.title(), bg=COLOR_CHIP_BG, fg=COLOR_TEXT,
                             font=(FONT_FAMILY, 9), padx=10, pady=4)
            chip.grid(row=row, column=col, padx=(0, 6), pady=(0, 6), sticky="w")
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

    def _build_role_card(self, rank: int, role: dict) -> ttk.Frame:
        card = ttk.Frame(self.results_inner, style="Card.TFrame", padding=18)
        card.pack(fill="x", pady=(0, 12))
        self._add_card_border(card)

        header = ttk.Frame(card, style="Card.TFrame")
        header.pack(fill="x")

        title_text = f"#{rank}  {role['job_role'].title()}"
        ttk.Label(header, text=title_text, style="CardTitle.TLabel",
                  font=(FONT_FAMILY, 13, "bold")).pack(side="left")

        # --- ATS score badge (signature element) ---
        badge_color = score_color(role["ats_score"])
        badge = tk.Frame(header, bg=badge_color, padx=12, pady=4)
        badge.pack(side="right")
        tk.Label(badge, text=f"ATS {role['ats_score']:.0f}%", bg=badge_color, fg="#ffffff",
                  font=(FONT_FAMILY, 10, "bold")).pack()

        ttk.Label(header, text=f"Model confidence: {role['confidence']:.0f}%",
                  style="CardMuted.TLabel").pack(side="right", padx=(0, 12))

        # Description
        ttk.Label(card, text=role["description"], style="CardText.TLabel", wraplength=760,
                  justify="left").pack(anchor="w", pady=(12, 10), fill="x")

        # Matched / missing skill rows
        if role["matched_required_skills"]:
            matched_row = ttk.Frame(card, style="Card.TFrame")
            matched_row.pack(fill="x", pady=(2, 4))
            ttk.Label(matched_row, text="✓ Matched:", style="CardMuted.TLabel",
                      foreground=COLOR_ACCENT_DARK).pack(side="left")
            ttk.Label(matched_row, text=", ".join(s.title() for s in role["matched_required_skills"]),
                      style="CardText.TLabel", wraplength=640, justify="left").pack(
                side="left", padx=(6, 0))

        if role["missing_required_skills"]:
            missing_row = ttk.Frame(card, style="Card.TFrame")
            missing_row.pack(fill="x", pady=(2, 0))
            ttk.Label(missing_row, text="○ To improve:", style="CardMuted.TLabel").pack(side="left")
            ttk.Label(missing_row, text=", ".join(s.title() for s in role["missing_required_skills"]),
                      style="CardMuted.TLabel", wraplength=620, justify="left").pack(
                side="left", padx=(6, 0))

        return card


if __name__ == "__main__":
    app = JobRecommenderApp()
    app.mainloop()
