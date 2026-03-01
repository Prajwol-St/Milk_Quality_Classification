import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading, os, sys
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")


BG      = "#0D1117"
SURFACE = "#161B22"
CARD    = "#1C2128"
BORDER  = "#30363D"
ACCENT  = "#58A6FF"
SUCCESS = "#3FB950"
WARNING = "#D29922"
DANGER  = "#F85149"
TEXT    = "#E6EDF3"
DIM     = "#8B949E"
GRADE_COLORS = {"high": SUCCESS, "medium": WARNING, "low": DANGER}
GRADE_ICONS  = {"high": "🟢",    "medium": "🟡",    "low": "🔴"}

FH1 = ("Consolas", 20, "bold")
FH2 = ("Consolas", 13, "bold")
FH3 = ("Consolas", 11, "bold")
FSM = ("Consolas", 10)
FXS = ("Consolas", 9)


SCRIPT_DIR  = os.path.dirname(os.path.abspath(sys.argv[0]))
DEFAULT_PKL = os.path.join(SCRIPT_DIR, "milk_model.pkl")


class MilkApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🥛 Milk Quality Classifier")
        self.geometry("1100x780")
        self.configure(bg=BG)
        self.resizable(True, True)

        self.model       = None
        self.scaler      = None
        self.le          = None
        self.feat_cols   = ["pH","Temprature","Taste","Odor","Fat","Turbidity","Colour"]
        self.importances = None
        self.metrics     = {}

        self._build_ui()
        threading.Thread(target=self._load_model, daemon=True).start()

    # ─────────────────── MODEL LOAD / TRAIN ───────────────────────

    def _load_model(self):
        """Load pre-trained .pkl. If missing, prompt user."""
        if os.path.exists(DEFAULT_PKL):
            self._log("Loading pre-trained model…")
            try:
                bundle = joblib.load(DEFAULT_PKL)
                self._apply_bundle(bundle)
                m = self.metrics
                self._set_status(
                    f"● Model loaded  |  Test {m.get('test_acc', 0):.2%}  "
                    f"|  CV {m.get('cv_mean', 0):.2%}±{m.get('cv_std', 0):.2%}  "
                    f"|  Gap {m.get('gap', 0):.2%}", SUCCESS)
                self._log(f"Model ready — trained on {m.get('n_rows', 'unknown')} rows")
                self.after(0, self._refresh_ui)
            except Exception as e:
                self._set_status("● Failed to load pkl", DANGER)
                self._log(f"Error: {e}")
        else:
            self._set_status("● milk_model.pkl not found — load a CSV to train", WARNING)
            self._log(f"Expected: {DEFAULT_PKL}")

    def _apply_bundle(self, bundle):
        self.model       = bundle["model"]
        self.scaler      = bundle["scaler"]
        self.le          = bundle["label_encoder"]
        self.feat_cols   = bundle["feature_cols"]
        self.importances = bundle["importances"]
        self.metrics     = bundle["metrics"]

    def _load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return
        self._set_status("● Retraining…", DIM)
        threading.Thread(target=self._retrain, args=(path,), daemon=True).start()

    def _retrain(self, csv_path):
        try:
            from sklearn.ensemble import (RandomForestClassifier,
                GradientBoostingClassifier, VotingClassifier)
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            from sklearn.model_selection import (train_test_split,
                cross_val_score, StratifiedKFold)
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


            self._log("Parsing CSV…")
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            cmap = {"fat ":"Fat","temprature":"Temprature",
                    "temperature":"Temprature","colour":"Colour","color":"Colour"}
            df.columns = [cmap.get(c.lower(), c) for c in df.columns]

            target = next((c for c in df.columns if c.lower()=="grade"), None)
            if not target:
                raise ValueError("'Grade' column not found.")

            feat_cols = [c for c in self.feat_cols if c in df.columns]
            df = df.dropna(subset=feat_cols+[target]).drop_duplicates()

            le = LabelEncoder()
            sc = StandardScaler()
            X  = sc.fit_transform(df[feat_cols].values.astype(float))
            y  = le.fit_transform(df[target].str.strip().str.lower())

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

            self._log(f"Training on {len(df)} rows…")
            rf = RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_split=5,
                min_samples_leaf=3, max_features="sqrt",
                class_weight="balanced", random_state=42, n_jobs=-1)
            gb = GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.08, max_depth=4,
                subsample=0.8, min_samples_leaf=5, random_state=42)
            model = VotingClassifier([("rf",rf),("gb",gb)],
                                     voting="soft", weights=[2,1])
            model.fit(X_tr, y_tr)

            tr_acc = accuracy_score(y_tr, model.predict(X_tr))
            te_acc = accuracy_score(y_te, model.predict(X_te))
            cv     = cross_val_score(model, X, y,
                cv=StratifiedKFold(10, shuffle=True, random_state=42))
            y_pred_te = model.predict(X_te)
            report = classification_report(y_te, y_pred_te,
                target_names=le.classes_, zero_division=0)
            cm = confusion_matrix(y_te, y_pred_te)


            bundle = {
                "model": model, "scaler": sc, "label_encoder": le,
                "feature_cols": feat_cols,
                "importances": model.estimators_[0].feature_importances_,
                "metrics": {
                    "train_acc": tr_acc, "test_acc": te_acc,
                    "gap": tr_acc - te_acc, "cv_mean": cv.mean(),
                    "cv_std": cv.std(), "classes": le.classes_,
                    "n_rows": len(df), "report": report,
                    "confusion_matrix": cm.tolist()
                }
            }


            # Save new pkl so next launch is instant
            joblib.dump(bundle, DEFAULT_PKL, compress=3)
            self._apply_bundle(bundle)

            self._set_status(
                f"● Retrained & saved  |  Test {te_acc:.2%}  "
                f"|  CV {cv.mean():.2%}±{cv.std():.2%}  "
                f"|  Gap {tr_acc-te_acc:.2%}", SUCCESS)
            self._log(f"Saved new milk_model.pkl  ({len(df)} rows)")
            self.after(0, self._refresh_ui)

        except Exception as e:
            self._log(f"Error: {e}")
            self._set_status("● Retraining failed", DANGER)
            self.after(0, lambda: messagebox.showerror("Error", str(e)))

    # ─────────────────── UI BUILD ──────────────────────────────────

    def _build_ui(self):
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=20, pady=(16,0))
        tk.Label(hdr, text="🥛  MILK QUALITY CLASSIFIER",
                 font=FH1, bg=BG, fg=ACCENT).pack(side="left")
        self.status_lbl = tk.Label(hdr, text="● Loading…",
                                   font=FSM, bg=BG, fg=DIM)
        self.status_lbl.pack(side="right")
        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=20, pady=8)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=20)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_left(body)
        self._build_right(body)

        bar = tk.Frame(self, bg=SURFACE, height=26)
        bar.pack(fill="x", side="bottom")
        self.log_lbl = tk.Label(bar, text="", font=FXS, bg=SURFACE, fg=DIM)
        self.log_lbl.pack(side="left", padx=12, pady=3)
        tk.Label(bar, text=f"pkl: {DEFAULT_PKL}",
                 font=FXS, bg=SURFACE, fg=DIM).pack(side="right", padx=12)

    def _card(self, parent, title, **gkw):
        f = tk.Frame(parent, bg=CARD, highlightthickness=1,
                     highlightbackground=BORDER)
        f.grid(**gkw, padx=8, pady=6, sticky="nsew")
        tk.Label(f, text=title, font=FH2, bg=CARD, fg=TEXT
                 ).pack(anchor="w", padx=14, pady=(12,4))
        ttk.Separator(f, orient="horizontal").pack(fill="x", padx=14)
        return f

    def _build_left(self, parent):
        card = self._card(parent, "📋  Sample Parameters", row=0, column=0)
        fields = [
            ("pH",          "6.6",  "Ideal: 6.50 – 6.80"),
            ("Temperature", "35",   "°C  |  Fresh: 35–45"),
            ("Taste",       "1",    "1 = Good   0 = Bad"),
            ("Odor",        "0",    "1 = Good   0 = Bad"),
            ("Fat",         "1",    "1 = Present   0 = Absent"),
            ("Turbidity",   "0",    "1 = High   0 = Low"),
            ("Colour",      "254",  "Range: 240 – 255"),
        ]
        self.entries = {}
        grid = tk.Frame(card, bg=CARD)
        grid.pack(fill="x", padx=14, pady=10)
        for i, (lbl, default, hint) in enumerate(fields):
            r, c = divmod(i, 2)
            cell = tk.Frame(grid, bg=CARD)
            cell.grid(row=r, column=c, padx=8, pady=5, sticky="ew")
            grid.columnconfigure(c, weight=1)
            tk.Label(cell, text=lbl,  font=FH3, bg=CARD, fg=TEXT).pack(anchor="w")
            tk.Label(cell, text=hint, font=FXS,  bg=CARD, fg=DIM ).pack(anchor="w")
            e = tk.Entry(cell, font=FSM, bg=SURFACE, fg=TEXT,
                         insertbackground=ACCENT, relief="flat",
                         highlightthickness=1, highlightbackground=BORDER,
                         highlightcolor=ACCENT, width=16)
            e.insert(0, default)
            e.pack(fill="x", pady=(2,0))
            self.entries[lbl.lower().replace(" ","")] = e

        btns = tk.Frame(card, bg=CARD)
        btns.pack(fill="x", padx=14, pady=(4,8))
        tk.Button(btns, text="⚡  CLASSIFY MILK QUALITY",
                  font=("Consolas",12,"bold"), bg=ACCENT, fg=BG,
                  activebackground="#79B8FF", relief="flat",
                  cursor="hand2", pady=10, command=self._classify
                  ).pack(fill="x")
        tk.Button(btns, text="💾  Load CSV & Retrain  (auto-saves pkl)",
                  font=FXS, bg=SURFACE, fg=DIM,
                  activebackground=CARD, relief="flat",
                  cursor="hand2", pady=6, command=self._load_csv
                  ).pack(fill="x", pady=(5,0))

        res = tk.Frame(card, bg=CARD)
        res.pack(fill="x", padx=14, pady=(0,14))
        self.result_lbl = tk.Label(res, text="", font=("Consolas",16,"bold"), bg=CARD)
        self.result_lbl.pack()
        self.prob_lbl = tk.Label(res, text="", font=FXS, bg=CARD, fg=DIM)
        self.prob_lbl.pack()
        self.conf_lbl = tk.Label(res, text="", font=FXS, bg=CARD, fg=DIM)
        self.conf_lbl.pack()

    def _build_right(self, parent):
        right = tk.Frame(parent, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=2)
        right.rowconfigure(1, weight=3)
        right.columnconfigure(0, weight=1)

        mc = self._card(right, "📊  Model Performance", row=0, column=0)
        self.metrics_inner = tk.Frame(mc, bg=CARD)
        self.metrics_inner.pack(fill="both", expand=True, padx=14, pady=8)
        tk.Label(self.metrics_inner, text="Loading model…",
                 font=FSM, bg=CARD, fg=DIM).pack()

        fc = self._card(right, "🔬  Feature Importances", row=1, column=0)
        self.fi_canvas = tk.Canvas(fc, bg=CARD, highlightthickness=0, height=200)
        self.fi_canvas.pack(fill="both", expand=True, padx=14, pady=(8,12))
        self.fi_canvas.bind("<Configure>", lambda e: self._draw_importances())

    # ─────────────────── CLASSIFY ─────────────────────────────────

    def _classify(self):
        if self.model is None:
            messagebox.showwarning("Not ready", "Model not loaded yet.")
            return
        try:
            key_order = ["ph","temperature","taste","odor","fat","turbidity","colour"]
            vals = [float(self.entries[k].get().strip())
                    for k in key_order if k in self.entries]
            if len(vals) != len(self.feat_cols):
                raise ValueError(f"Need {len(self.feat_cols)} features, got {len(vals)}.")

            X_s   = self.scaler.transform(np.array(vals).reshape(1,-1))
            idx   = self.model.predict(X_s)[0]
            proba = self.model.predict_proba(X_s)[0]
            grade = self.le.inverse_transform([idx])[0]
            conf  = proba.max()

            self.result_lbl.config(
                text=f"{GRADE_ICONS[grade]}  Grade: {grade.upper()}",
                fg=GRADE_COLORS[grade])
            self.prob_lbl.config(
                text="    ".join(f"{c}: {p:.0%}"
                                 for c,p in zip(self.le.classes_, proba)))
            self.conf_lbl.config(
                text=f"Confidence: {conf:.1%}",
                fg=SUCCESS if conf > 0.85 else WARNING)
            self._log(f"→ {grade.upper()}  (confidence {conf:.1%})")

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

    # ─────────────────── METRICS / IMPORTANCES ────────────────────

    def _refresh_ui(self):
        self._draw_metrics()
        self._draw_importances()

    def _draw_metrics(self):
        for w in self.metrics_inner.winfo_children():
            w.destroy()
        m = self.metrics

        row = tk.Frame(self.metrics_inner, bg=CARD)
        row.pack(fill="x", pady=(0,6))
        for lbl, val, clr in [
            ("Test Acc",  f"{m.get('test_acc', 0):.2%}",  SUCCESS),
            ("Train Acc", f"{m.get('train_acc', 0):.2%}", ACCENT),
            ("CV Mean",   f"{m.get('cv_mean', 0):.2%}",   ACCENT),
            ("Gap",       f"{m.get('gap', 0):.2%}",
             WARNING if m.get('gap', 0) < 0.05 else DANGER),
        ]:
            b = tk.Frame(row, bg=SURFACE, padx=8, pady=6)
            b.pack(side="left", expand=True, fill="both", padx=3)
            tk.Label(b, text=val, font=("Consolas",12,"bold"),
                     bg=SURFACE, fg=clr).pack()
            tk.Label(b, text=lbl, font=FXS, bg=SURFACE, fg=DIM).pack()

        gap = m.get("gap", 0)
        msg, clr = (("✅ No overfitting", SUCCESS) if gap < 0.02 else
                    ("⚠️  Slight overfit (acceptable)", WARNING) if gap < 0.05 else
                    ("❌ Overfitting detected", DANGER))
        tk.Label(self.metrics_inner, text=msg, font=FSM,
                 bg=CARD, fg=clr).pack(pady=(0,6))

        if "report" in m:
            tk.Label(self.metrics_inner, text="Per-class breakdown:",
                     font=FXS, bg=CARD, fg=DIM).pack(anchor="w")
            hdr = tk.Frame(self.metrics_inner, bg=CARD)
            hdr.pack(fill="x")
            for h, w in [("Class",9),("Precision",10),("Recall",9),("F1",9)]:
                tk.Label(hdr, text=h, font=FXS, bg=CARD, fg=DIM,
                         width=w, anchor="w").pack(side="left")
            for line in m["report"].strip().split("\n")[2:]:
                parts = line.split()
                if len(parts) >= 4 and parts[0] in m["classes"]:
                    r2 = tk.Frame(self.metrics_inner, bg=CARD)
                    r2.pack(fill="x", pady=1)
                    tk.Label(r2, text=parts[0], font=FXS,
                             fg=GRADE_COLORS.get(parts[0], TEXT),
                             bg=CARD, width=9, anchor="w").pack(side="left")
                    for v in parts[1:4]:
                        tk.Label(r2, text=v, font=FXS, bg=CARD,
                                 fg=TEXT, width=10, anchor="w").pack(side="left")

        if "confusion_matrix" in m:
            tk.Label(self.metrics_inner, text="Confusion Matrix:",
                     font=FXS, bg=CARD, fg=DIM).pack(anchor="w", pady=(6,2))
            cm_frame = tk.Frame(self.metrics_inner, bg=CARD)
            cm_frame.pack(fill="x")
            
            tk.Label(cm_frame, text="True \ Pred", font=FXS, bg=CARD, fg=DIM, width=12).grid(row=0, column=0)
            for j, c in enumerate(m["classes"]):
                tk.Label(cm_frame, text=c, font=FXS, bg=CARD, fg=DIM, width=8).grid(row=0, column=j+1)
            
            cm = m["confusion_matrix"]
            for i, c_true in enumerate(m["classes"]):
                tk.Label(cm_frame, text=c_true, font=FXS, bg=CARD, fg=GRADE_COLORS.get(c_true, TEXT), width=12, anchor="e").grid(row=i+1, column=0, padx=(0,6))
                for j, val in enumerate(cm[i]):
                    tk.Label(cm_frame, text=str(val), font=FXS, bg=CARD, fg=TEXT, width=8).grid(row=i+1, column=j+1)
        else:
            tk.Label(self.metrics_inner, text="Confusion Matrix: Retrain to view",
                     font=FXS, bg=CARD, fg=DIM).pack(anchor="w", pady=(6,0))

        tk.Label(self.metrics_inner,
                 text=f"Trained on {m.get('n_rows', 'unknown')} rows",
                 font=FXS, bg=CARD, fg=DIM).pack(pady=(6,0))


    def _draw_importances(self):
        c = self.fi_canvas
        c.delete("all")
        c.update_idletasks()
        if self.importances is None:
            return
        W = c.winfo_width() or 420
        H = c.winfo_height() or 200
        vals  = self.importances
        order = np.argsort(vals)[::-1]
        n     = len(self.feat_cols)
        bar_h = min(24, (H - 20) // n)
        maxv  = vals.max() or 1
        lw    = 95
        for i, idx in enumerate(order):
            y   = 10 + i * (bar_h + 5)
            bw  = int((vals[idx]/maxv) * (W - lw - 55))
            clr = ACCENT if i == 0 else "#3D6B8F"
            c.create_text(lw-6, y+bar_h//2, text=self.feat_cols[idx],
                          anchor="e", font=FXS, fill=TEXT)
            c.create_rectangle(lw, y, lw+bw, y+bar_h, fill=clr, outline="")
            c.create_text(lw+bw+5, y+bar_h//2,
                          text=f"{vals[idx]:.3f}", anchor="w",
                          font=FXS, fill=DIM)

    # ─────────────────── HELPERS ──────────────────────────────────

    def _set_status(self, msg, color=DIM):
        self.after(0, lambda: self.status_lbl.config(text=msg, fg=color))

    def _log(self, msg):
        self.after(0, lambda: self.log_lbl.config(text=msg))


if __name__ == "__main__":
    app = MilkApp()
    app.mainloop()