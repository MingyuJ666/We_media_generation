from PIL import Image, ImageDraw, ImageFont
import os
import cv2


def traverse_folder(folder_path):
    file_name = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_name.append(file_path)
    return file_name




def remove_ds_store(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == '.DS_Store':
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")







def combine_image(target_size, dataset, answer, images=[]):
    i = 0
    for batch in dataset:
       images.append(batch)
       i += 1

    if(i==9):
      # Create a new blank image to hold the nine-square grid
      grid_size = (target_size[0] * 3, target_size[1] * 3)
      grid_image = Image.new("RGB", grid_size)

      for i in range(3):
        for j in range(3):
          index = i * 3 + j
          grid_image.paste(images[index], (j * target_size[0], i * target_size[1]))


      grid_image.save("Output_image/grid_image.jpg")  # Save the grid image to a file

    if(i==4):

        # Create a new blank image to hold the four-square grid
        grid_size = (target_size[0] * 2, target_size[1] * 2)
        grid_image = Image.new("RGB", grid_size)

        # Paste the images onto the grid
        grid_image.paste(images[0], (0, 0))
        grid_image.paste(images[1], (target_size[0], 0))
        grid_image.paste(images[2], (0, target_size[1]))
        grid_image.paste(images[3], (target_size[0], target_size[1]))

        grid_image.save("Output_image/grid_image.jpg")  # Save the grid image to a file


















