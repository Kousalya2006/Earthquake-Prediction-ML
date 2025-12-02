import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# ------------------------------------------
# LOAD MODEL BUNDLE
# ------------------------------------------
BUNDLE = 'eq_model_bundle_new.pkl'
bundle = joblib.load(BUNDLE)

features = bundle['features']
reg = bundle['reg_pipeline']
cls = bundle['cls_pipeline']


# ---------------------------------------------------
# EARTHQUAKE SUMMARY BASED ON MAGNITUDE
# ---------------------------------------------------
def get_summary(mag):
    if mag < 3:
        return ("Minor", "Very Low", "None")
    elif mag < 5:
        return ("Light", "Low", "Minimal")
    elif mag < 6.5:
        return ("Moderate", "Medium", "Noticeable")
    elif mag < 8:
        return ("Strong", "High", "Serious")
    else:
        return ("Very Strong", "Very High", "Catastrophic")


# ---------------------------------------------------
# GAUGE (SPEEDOMETER) GRAPH
# ---------------------------------------------------
def show_gauge(mag):
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': 'polar'})

    # Speedometer range (0 to 10)
    lower = 0
    upper = 10
    theta = np.linspace(0.75 * pi, 0.25 * pi, 100)

    # Color zones
    colors = []
    for v in np.linspace(lower, upper, 100):
        if v < 3:
            colors.append("green")
        elif v < 5:
            colors.append("yellow")
        elif v < 6.5:
            colors.append("orange")
        else:
            colors.append("red")

    # Draw speedometer
    ax.bar(theta, np.ones_like(theta) * 1, width=0.02, color=colors, alpha=0.8)

    # Pointer for magnitude
    angle = 0.75 * pi - (mag / 10) * (0.5 * pi)
    ax.arrow(angle, 0, 0, 0.7, width=0.03, color="black")

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title(f"Predicted Magnitude: {mag:.2f}", fontsize=16)

    plt.show()


# ----------------------------------------------------------
# TKINTER GUI WINDOW
# ----------------------------------------------------------
root = tk.Tk()
root.title("Earthquake Predictor")
root.geometry("550x750")

tk.Label(root, text="Earthquake Predictor", font=("Arial", 18, "bold")).pack(pady=10)
tk.Label(root, text="Enter the input features:", font=("Arial", 11)).pack()

frame = tk.Frame(root)
frame.pack(pady=10)

entries = {}

# Input fields for all model features
for f in features:
    row = tk.Frame(frame)
    row.pack(fill="x", pady=5)
    tk.Label(row, text=f, width=15, anchor="w").pack(side="left")
    ent = tk.Entry(row, width=15)
    ent.pack(side="left")
    entries[f] = ent


# ----------------------------------------------------------
# OUTPUT LABELS
# ----------------------------------------------------------
out_mag = tk.Label(root, text="Predicted Magnitude: —", font=("Arial", 14, "bold"))
out_mag.pack(pady=10)

out_cls = tk.Label(root, text="Strength: —", font=("Arial", 14, "bold"))
out_cls.pack(pady=5)

out_prob = tk.Label(root, text="Strong Probability: —", font=("Arial", 12))
out_prob.pack(pady=5)


# ----------------------------------------------------------
# SCIENTIFIC SUMMARY PANEL
# ----------------------------------------------------------
summary_title = tk.Label(root, text="Earthquake Summary:", font=("Arial", 14, "bold"))
summary_title.pack(pady=10)

summary_intensity = tk.Label(root, text="Intensity Level: —", font=("Arial", 12))
summary_intensity.pack()

summary_damage = tk.Label(root, text="Expected Damage: —", font=("Arial", 12))
summary_damage.pack()

summary_impact = tk.Label(root, text="Human Impact: —", font=("Arial", 12))
summary_impact.pack()


# ----------------------------------------------------------
# PREDICT FUNCTION
# ----------------------------------------------------------
def predict():
    try:
        vals = {}
        for f, ent in entries.items():
            txt = ent.get().strip()
            vals[f] = None if txt == "" else float(txt)

        X = pd.DataFrame([vals], columns=features)

        # Make predictions
        mag = float(reg.predict(X)[0])
        proba = float(cls.predict_proba(X)[0, 1])
        label = "STRONG" if proba >= 0.5 else "WEAK"

        # Update GUI text
        out_mag.config(text=f"Predicted Magnitude: {mag:.2f}")
        out_cls.config(text=f"Strength: {label}")
        out_prob.config(text=f"Strong Probability: {proba:.3f}")

        # Update scientific summary
        inten, dmg, imp = get_summary(mag)
        summary_intensity.config(text=f"Intensity Level: {inten}")
        summary_damage.config(text=f"Expected Damage: {dmg}")
        summary_impact.config(text=f"Human Impact: {imp}")

        # Show gauge graph
        show_gauge(mag)

    except Exception as e:
        messagebox.showerror("Error", str(e))


# ----------------------------------------------------------
# BUTTONS
# ----------------------------------------------------------
btn_frame = tk.Frame(root)
btn_frame.pack(pady=20)

tk.Button(btn_frame, text="Predict", width=10, command=predict).grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="Clear", width=10,
          command=lambda: [e.delete(0, tk.END) for e in entries.values()]).grid(row=0, column=1, padx=5)

root.mainloop()
