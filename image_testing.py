# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # from PIL import Image

# # # # loading the segmented image
# # # segmented_image = Image.open("20200203_094523_forniceal_palpebral.png")

# # # # converting to NumPy array
# # # image_array = np.array(segmented_image)

# # # # checkign the shape
# # # print("Original shape:", image_array.shape)

# # # # separating the RGBA channels
# # # r, g, b, a = (
# # #     image_array[..., 0],
# # #     image_array[..., 1],
# # #     image_array[..., 2],
# # #     image_array[..., 3],
# # # )

# # # # creating a mask where the alpha channel is greater than 0
# # # mask = a > 0

# # # # creating an RGB image using the mask
# # # masked_image_rgb = np.zeros(
# # #     (image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8
# # # )

# # # # applyging the mask
# # # masked_image_rgb[mask] = image_array[mask][
# # #     :, :3
# # # ]  # keeping only RGB channels where the mask is True

# # # print(masked_image_rgb.shape)
# # # plt.imshow(masked_image_rgb)
# # # plt.axis("off")
# # # plt.title("Masked Image (Only Banana Shape)")
# # # plt.show()

# # # # Optionally savign the masked image
# # # # Image.fromarray(masked_image_rgb).save('masked_image.png')


# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

# segmented_image = Image.open("20200203_094523_forniceal_palpebral.png")

# image_array = np.array(segmented_image)

# print("Original shape:", image_array.shape)

# # seperating the RGBA channels
# a = image_array[..., 3]  # Get the alpha channel

# # creating a binary mask where the alpha channel is greater than 0
# binary_mask = (a == 255).astype(np.uint8)  # coverts to binary mask with 255 for the regions of interest

# print("Binary mask shape:", binary_mask.shape)

# # Display the binary mask
# plt.imshow(binary_mask, cmap='gray')
# plt.axis("off")
# plt.title("Binary Mask")
# plt.show()

