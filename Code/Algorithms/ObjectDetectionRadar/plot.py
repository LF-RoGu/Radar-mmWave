import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

def create_interactive_plot(x_data, y_data, x_limits, y_limits):
    """
    Create an interactive plot with two subplots, a slider, and radio buttons.
    
    Parameters:
        x_data (array-like): The x data for the plot.
        y_data (array-like): The y data for the plot.
        x_limits (tuple): The x-axis limits as (xmin, xmax).
        y_limits (tuple): The y-axis limits as (ymin, ymax).
    """
    # Create the figure and subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    ax1, ax2 = axes
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Create lines for ax1
    (line1,) = ax1.plot([], [], 'o-', label="Data - Ax1")
    ax1.set_xlim(*x_limits)
    ax1.set_ylim(*y_limits)
    ax1.legend()

    # ax2 settings
    ax2.set_xlim(*x_limits)
    ax2.set_ylim(*y_limits)
    ax2.legend(["Current Value - Ax2"], loc="upper left")

    # Add sliders and radio buttons
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])  # [left, bottom, width, height]
    slider = Slider(ax_slider, "Frame", 0, len(x_data) - 1, valinit=0, valstep=1)

    ax_radio = plt.axes([0.025, 0.5, 0.15, 0.15])  # [left, bottom, width, height]
    radio = RadioButtons(ax_radio, ("continuous", "new"))

    # Update function
    def update(val):
        frame = int(slider.val)  # Get the slider value
        mode = radio.value_selected  # Get the selected mode
        
        # Update ax1
        if mode == "continuous":
            line1.set_data(x_data[: frame + 1], y_data[: frame + 1])
        elif mode == "new":
            line1.set_data([x_data[frame]], [y_data[frame]])
        
        # Update ax2 (clears the plot and shows only the current value)
        ax2.cla()  # Clear ax2
        ax2.set_xlim(*x_limits)  # Reset x-axis limits
        ax2.set_ylim(*y_limits)  # Reset y-axis limits
        ax2.plot([x_data[frame]], [y_data[frame]], 'ro')  # Plot only the current point
        ax2.set_title("Current Value - Ax2")
        ax2.legend(["Current Value"], loc="upper left")

        fig.canvas.draw_idle()

    # Connect the slider and radio buttons to the update function
    slider.on_changed(update)
    radio.on_clicked(update)

    plt.show()


# Example usage:
x = np.linspace(0, 10, 100)
y = 2 * x + 1  # Example: y = 2x + 1
create_interactive_plot(x, y, x_limits=(0, 10), y_limits=(0, 21))
