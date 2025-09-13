import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from .ris_localization import RISLocalization

class RISLocalizationViz(RISLocalization):
    def plot_error_bounds(self, theta_range, aeb_vals, crlb_vals, mcrlb_vals):
        plt.figure(figsize=(10, 6))
        plt.plot(np.degrees(theta_range), aeb_vals, label="AEB", marker="o", markersize=4)
        plt.plot(np.degrees(theta_range), crlb_vals, label="CRLB", marker="x", markersize=4)
        plt.plot(np.degrees(theta_range), mcrlb_vals, label="MCRLB", marker="s", markersize=4)
        plt.xlabel("Angle of Arrival (degrees)")
        plt.ylabel("Error Bound (radians)")
        plt.yscale("log")
        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.legend()
        plt.title("Localization Error Bounds vs. Angle of Arrival")
        plt.tight_layout()
        plt.show()

    def plot_heatmap(self, theta_range, power_range, error_bounds):
        safe_vals = np.maximum(error_bounds, 1e-30)
        log_vals = np.log10(safe_vals)

        plt.figure(figsize=(10, 7))
        sns.heatmap(
            log_vals,
            xticklabels=np.round(np.degrees(theta_range[::max(1, len(theta_range)//10)]), 2),
            yticklabels=np.round(power_range[::max(1, len(power_range)//10)], 1),
            cmap="viridis",
            cbar_kws={"label": "log10(Error Bound)"},
        )
        plt.xlabel("Angle of Arrival (degrees)")
        plt.ylabel("Transmit Power (dBm)")
        plt.title("Error Bound Heatmap (log10 scale)")
        plt.tight_layout()
        plt.show()

    def plot_fim_heatmap(self, fim):
        plt.figure(figsize=(6, 5))
        sns.heatmap(np.abs(fim), annot=True, fmt=".2e", cmap="viridis", cbar_kws={"label": "abs(FIM)"})
        plt.title("Fisher Information Matrix")
        plt.xlabel("Parameter Index")
        plt.ylabel("Parameter Index")
        plt.tight_layout()
        plt.show()

    def plot_interactive_error_bounds(self, theta_range, aeb_vals, crlb_vals, mcrlb_vals):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.degrees(theta_range), y=aeb_vals, name="AEB", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=np.degrees(theta_range), y=crlb_vals, name="CRLB", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=np.degrees(theta_range), y=mcrlb_vals, name="MCRLB", mode="lines+markers"))

        fig.update_layout(
            title="Interactive Localization Error Bounds",
            xaxis_title="Angle of Arrival (degrees)",
            yaxis_title="Error Bound (radians)",
            yaxis_type="log",
            hovermode="x unified",
            width=900,
            height=500,
        )
        fig.show()

    def plot_interactive_heatmap(self, theta_range, power_range, error_bounds):
        safe_vals = np.maximum(error_bounds, 1e-30)
        log_vals = np.log10(safe_vals)

        fig = go.Figure(
            data=go.Heatmap(
                z=log_vals,
                x=np.degrees(theta_range),
                y=power_range,
                colorscale="Viridis",
                hovertemplate="Angle: %{x:.1f}°<br>Power: %{y:.1f} dBm<br>log10(Error): %{z:.3f}<extra></extra>",
            )
        )
        fig.update_layout(title="Interactive Error Bound Heatmap", xaxis_title="Angle (degrees)", yaxis_title="Power (dBm)")
        fig.show()
