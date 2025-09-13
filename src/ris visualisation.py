import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from .ris_localization import RISLocalization


class RISLocalizationViz(RISLocalization):
    def plot_results(self, theta_range, aeb_values, crlb_values, mcrlb_values):
        plt.figure(figsize=(12, 6))
        plt.plot(np.degrees(theta_range), aeb_values, label="AEB")
        plt.plot(np.degrees(theta_range), crlb_values, label="CRLB")
        plt.plot(np.degrees(theta_range), mcrlb_values, label="MCRLB")
        plt.xlabel("Angle of Arrival (degrees)")
        plt.ylabel("Error Bound (radians)")
        plt.yscale("log")
        plt.grid(True)
        plt.legend()
        plt.title("Localization Error Bounds vs. Angle of Arrival")
        plt.show()

    def plot_heatmap(self, theta_range, power_range, error_bounds):
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            np.log10(error_bounds),
            xticklabels=np.round(np.degrees(theta_range[::5]), 1),
            yticklabels=np.round(power_range[::5], 1),
            cmap="viridis",
        )
        plt.xlabel("Angle of Arrival (degrees)")
        plt.ylabel("Transmission Power (dBm)")
        plt.title("Error Bound Heatmap (log10 scale)")
        plt.show()

    def plot_interactive_error_bounds(self, theta_range, aeb_values, crlb_values, mcrlb_values):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.degrees(theta_range), y=aeb_values, name="AEB", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=np.degrees(theta_range), y=crlb_values, name="CRLB", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=np.degrees(theta_range), y=mcrlb_values, name="MCRLB", mode="lines+markers"))

        fig.update_layout(
            title="Localization Error Bounds vs. Angle of Arrival (Interactive)",
            xaxis_title="Angle of Arrival (degrees)",
            yaxis_title="Error Bound (radians)",
            yaxis_type="log",
            hovermode="x unified",
        )
        fig.show()

    def plot_interactive_heatmap(self, theta_range, power_range, error_bounds):
        fig = go.Figure(data=go.Heatmap(
            z=np.log10(error_bounds),
            x=np.degrees(theta_range),
            y=power_range,
            colorscale="Viridis",
            hovertemplate="Angle: %{x:.1f}°<br>Power: %{y:.1f} dBm<br>Error (log10): %{z:.2f}<extra></extra>",
        ))
        fig.update_layout(title="Error Bound Heatmap (Interactive)")
        fig.show()

    def plot_fim_heatmap(self, fim):
        plt.figure(figsize=(8, 6))
        sns.heatmap(np.abs(fim), annot=True, fmt=".2e", cmap="viridis")
        plt.title("Fisher Information Matrix (Magnitude)")
        plt.xlabel("Parameter Index")
        plt.ylabel("Parameter Index")
        plt.show()
