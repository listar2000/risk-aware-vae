from PIL import Image, ImageDraw, ImageFont
import numpy as np

# specify the image size and font size
image_size = (28, 28)
font_size = 24

# specify the font file to use
font_path = "arial.ttf"

# generate the list of letters to create
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# create an empty numpy array to store the images
images = np.zeros((len(letters), image_size[0], image_size[1]), dtype=np.uint8)

# loop over the letters and generate the corresponding images
for i, letter in enumerate(letters):
    print("creating letter {}".format(letter))
    # create a new image
    img = Image.new('L', image_size, color=255)

    # get a drawing context
    draw = ImageDraw.Draw(img)

    # set the font
    font = ImageFont.truetype(font_path, font_size)

    # draw the letter in black
    text_bbox = draw.textbbox((0, 0), letter, font=font)
    text_position = ((image_size[0] - text_bbox[2]) / 2, (image_size[1] - text_bbox[3]) / 2)
    draw.text(text_position, letter, fill=0, font=font)

    # convert the image to a numpy array and store it
    images[i] = np.array(img)

# save the images to disk
for i, letter in enumerate(letters):
    filename = f"{letter}.png"
    Image.fromarray(images[i]).save(filename)
