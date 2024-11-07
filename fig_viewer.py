# %%
import matplotlib.pyplot as plt
import sys
import pickle
import os
import io
from PIL import Image
import numpy as np

class FigureNavigator:
    def __init__(self, figures):
        self.figures = figures
        self.index = 0

        # Create a main figure and axes for displaying images
        self.fig, self.ax = plt.subplots()
        self.image = None

        # Display the first figure
        self.show_figure()

        # Connect the key press event handler
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Show the window
        plt.show()

    def show_figure(self):
        # Render the current figure to an image buffer
        buf = io.BytesIO()
        self.figures[self.index].savefig(buf, format='png')
        buf.seek(0)
        im = Image.open(buf)
        im_array = np.asarray(im)
        buf.close()

        # Clear the axes and display the image
        self.ax.clear()
        self.ax.imshow(im_array)
        self.ax.axis('off')  # Hide the axes for a cleaner look
        self.fig.canvas.manager.set_window_title(f"Figure {self.index + 1} of {len(self.figures)}")
        self.fig.canvas.draw()

    def on_key_press(self, event):
        if event.key == 'right':
            # Move to the next figure
            self.index = (self.index + 1) % len(self.figures)
            self.show_figure()
        elif event.key == 'left':
            # Move to the previous figure
            self.index = (self.index - 1) % len(self.figures)
            self.show_figure()


# %%
# Create some example figures
if len(sys.argv) != 2:
    print("Usage: python fig_viewer.py /path/to/file.pickle")
    sys.exit(1)

file = sys.argv[1]

if not os.path.isfile(file):
    print(f"The specified filename does not exist: {file}")
    sys.exit(1)

figures = []
with open(file, 'rb') as f:
    while True:
        try:
            fig = pickle.load(f)
            figures.append(fig)
        except EOFError:
            break

# Start the figure navigator
navigator = FigureNavigator(figures)

