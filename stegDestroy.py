from PIL import Image
import numpy as np
from scipy.fftpack import dct, idct
from os.path import splitext

""" YCbCr to RGB """
def ycbcr_to_rgb(image_array):
    image = Image.fromarray(np.uint8(image_array), 'YCbCr')
    return image.convert('RGB')

def extract_mcus(image_array, block_size=8):
    """Extract 8x8 MCU blocks from the image"""
    height, width, _ = image_array.shape
    mcus = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image_array[y:y+block_size, x:x+block_size, :]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                mcus.append((y, x, block))
    return mcus

def apply_dct_to_mcu(mcu):
    """Apply 2D DCT to each channel of an MCU."""
    return np.stack([dct(dct(channel.T, norm='ortho').T, norm='ortho') for channel in np.rollaxis(mcu, 2)], axis=0)

def apply_idct_to_mcu(mcu):
    """Apply inverse 2D DCT to each channel of an MCU with corrected axis order."""
    return np.moveaxis(
        np.stack([idct(idct(channel.T, norm='ortho').T, norm='ortho') for channel in mcu], axis=0),
        0, -1
    )

def custom_operation(mcu_dct):
    # Example: Zeroing out high-frequency coefficients
    #mcu_dct[:, 2:, :] = 0
    #mcu_dct[:, :, 2:] = 0
    
    mcu_dct[1:, 6:, :] = 0
    mcu_dct[1:, :, 6:] = 0

    return mcu_dct

def process_and_recompress(image_path, output_path):
    """Process JPEG image, modify MCU DCT coefficients, and recompress it."""
    image = Image.open(image_path)
    ycbcr_image = image.convert('YCbCr')
    image_array = np.array(ycbcr_image, dtype=np.uint8)

    mcus = extract_mcus(image_array)
    
    for y, x, mcu in mcus:
        dct_mcu = apply_dct_to_mcu(mcu)
        modified_dct_mcu = custom_operation(dct_mcu)
        reconstructed_mcu = apply_idct_to_mcu(modified_dct_mcu)
        image_array[y:y+8, x:x+8, :] = reconstructed_mcu

    recompressed_image = ycbcr_to_rgb(image_array)
    recompressed_image.save(output_path, 'JPEG')
    print(f"Image saved to {output_path}.")

def main():
    image_path = input("Enter the path to the JPEG image: ")
    
    # Split the path and create output path
    name, ext = splitext(image_path)
    output_path = f"{name}_mod{ext}"
    
    process_and_recompress(image_path, output_path)



if __name__ == "__main__":
    main()
