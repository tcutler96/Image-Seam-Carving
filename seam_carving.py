import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


# loads image from path
def load_image(path):
    image = cv2.imread(path)
    return image


# detects edges using Sobel operators and calculates their magnitude as energy
def get_energy_array(image, plot=False):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    if plot:
        plt.subplot(1, 5, 1), plt.imshow(image[:, :, [2, 1, 0]])
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        plt.subplot(1, 5, 2), plt.imshow(grey, cmap='gray')
        plt.title('Grey Scale'), plt.xticks([]), plt.yticks([])
        plt.subplot(1, 5, 3), plt.imshow(sobel_x, cmap='gray')
        plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
        plt.subplot(1, 5, 4), plt.imshow(sobel_y, cmap='gray')
        plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
        plt.subplot(1, 5, 5), plt.imshow(energy, cmap='gray')
        plt.title('Edge Energy'), plt.xticks([]), plt.yticks([])
        plt.show()
    return energy


# calculates the minimum energy to bottom for all pixel in the image with associated directions
def get_min_energy_array(energy):
    min_energy = np.zeros_like(energy)
    directions = np.zeros_like(energy).astype('int')
    # the minimum energy on the last row is the row itself
    min_energy[-1] = energy[-1]
    # loops through all other pixels
    for height in range(energy.shape[0] - 2, -1, -1):
        for width in range(energy.shape[1]):
            # current pixel
            pixel = energy[height, width]
            # three pixels below
            below = min_energy[height + 1, max(width - 1, 0):min(width + 2, energy.shape[1] - 1)]
            min_energy[height, width] = pixel + min(below)
            directions[height, width] = np.argmin(below) + int(width == 0) - 1
    return min_energy, directions


# gets minimum energy path from the top of the image to the bottom
def get_min_energy_path(start, directions):
    heights = np.arange(0, directions.shape[0], 1)
    width = start
    widths = []
    for height in heights:
        widths.append(width)
        width += directions[height, width]
    return [heights, widths]


# removes minimum energy path from image
def remove_min_energy_path(array, x):
    output = array.tolist()
    for h, w in enumerate(x):
        del output[h][w]
    output = np.array(output)
    return output


# applies seam carving process to image in order to shrink while retaining important parts
def apply_seam_carving(image, n=50, save=None, plot=True):
    seam = image.copy()
    energy = get_energy_array(seam)
    for _ in range(n):
        energy_min, directions = get_min_energy_array(energy)
        _, x = get_min_energy_path(np.argmin(energy_min[0]), directions)
        # removes minimum energy path from image and energy array
        seam = remove_min_energy_path(seam, x)
        energy = remove_min_energy_path(energy, x)
    if save:
        cv2.imwrite(save, seam)
    if plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax1.imshow(image[:, :, [2, 1, 0]])
        ax1.title.set_text('Original Image'), ax1.set_xticks([]), ax1.set_yticks([]), ax1.axis('off')
        ax2.imshow(seam[:, :, [2, 1, 0]])
        ax2.title.set_text(f'Seam Carving (n={n})'), ax2.set_xlim(ax1.get_xlim()), ax2.set_xticks([])
        ax2.set_yticks([]), ax2.axis('off')
        fig.tight_layout()
        plt.show()
    return seam


# shows interactive plot of path of minimum energy depending on starting position
def show_min_energy_path(image):
    # draws window with image and current path
    def draw_window(im, path):
        ax1.imshow(im[:, :, [2, 1, 0]])
        ax1.scatter(path[1], path[0], s=0.1, c='tab:pink')
        ax1.title.set_text('Original Image'), ax1.set_xticks([]), ax1.set_yticks([])
        ax2.imshow(energy, cmap='viridis')
        ax2.scatter(path[1], path[0], s=0.1, c='tab:pink')
        ax2.title.set_text('Energy Array'), ax2.set_xticks([]), ax2.set_yticks([])

    # clears windows and then redraws it with updated path
    def update_window(val):
        ax1.cla()
        ax2.cla()
        draw_window(image, get_min_energy_path(int(val), directions))
        fig.canvas.draw_idle()

    energy, directions = get_min_energy_array(get_energy_array(image))
    # initialise figure
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    fig.tight_layout()
    # initial plot
    draw_window(image, get_min_energy_path(0, directions))
    # draw slider
    seam_slider_ax = fig.add_axes([0.15, 0.85, 0.7, 0.02], facecolor='lightgoldenrodyellow')
    seam_slider = Slider(seam_slider_ax, '', 0, image.shape[1] - 1, valinit=0, valstep=1)
    seam_slider.valtext.set_visible(False)
    # detects slider change
    seam_slider.on_changed(update_window)
    plt.title('Minimum Energy Path to Bottom')
    # maximise window
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()


# showcases the effect of seam carving with greater amounts removed
def showcase_seam_carving(paths_titles):
    fig, axes = plt.subplots(1, len(paths_titles))
    c = 0
    x_lim = 0
    for index, path_title in enumerate(paths_titles):
        c += 1
        image = load_image(path_title[0])
        # gets width of first image
        if x_lim == 0:
            x_lim = image.shape[1]
        axes[index].imshow(image[:, :, [2, 1, 0]])
        axes[index].title.set_text(path_title[1]), axes[index].set_xlim(0, x_lim)
        axes[index].set_xticks([]), axes[index].set_yticks([]), axes[index].axis('off')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    img = load_image(path='Images/clocks.jpg')
    show_min_energy_path(image=img)
    apply_seam_carving(image=img, n=50)
    showcase_seam_carving([['Images/clocks.jpg', 'n=0'],
                           ['Images/clocks_seam_50.jpg', 'n=50'],
                           ['Images/clocks_seam_100.jpg', 'n=100'],
                           ['Images/clocks_seam_150.jpg', 'n=150'],
                           ['Images/clocks_seam_200.jpg', 'n=200']])
