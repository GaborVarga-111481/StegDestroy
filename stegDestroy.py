from PIL import Image
import numpy as np
from scipy.fftpack import dct, idct
from os.path import splitext

def process_and_save(image_path, output_path):

    image = Image.open(image_path).convert('YCbCr')
    image_array = np.array(image, dtype=np.float64)
    height, width, _ = image_array.shape
    
    for c in range(1,3): # Only process channels (Cb, Cr)
        
        for y in range(0, height - 7, 8):
            for x in range(0, width - 7, 8):

                block = image_array[y:y+8, x:x+8, c].copy()
                    
                # Shift values to be in range {-127, 128}
                block -= 128.0
                    
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                    
                # Zero out lower right triangle
                for i in range(8):
                    for j in range(8):
                        if i + j >= 1:
                            dct_block[i, j] = 0
                    
                # Apply inverse DCT
                idct_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
                    
                # Shift back to original range
                idct_block += 128.0
                    
                # Store back in the image array
                image_array[y:y+8, x:x+8, c] = idct_block
    
    # Convert back to uint8 and save
    output_image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8), mode='YCbCr')
    output_image = output_image.convert('RGB')
    output_image.save(output_path)
    print(f"Image saved to {output_path}.")

def main():
    image_path = input("Enter the path to the JPEG image: ")

    name, ext = splitext(image_path)
    output_path = f"{name}_mod{ext}"

    process_and_save(image_path, output_path)

if __name__ == "__main__":
    main()