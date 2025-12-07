from PIL import Image
import os

img_path = "./figures/software_architecture.png"
try:
    with Image.open(img_path) as img:
        img.load()  # Force load image data
        # Save as a new file first to verify
        new_path = "./figures/software_architecture_fixed.png"
        img.save(new_path)
        print(f"Successfully re-saved image to {new_path}")
        
        # If successful, overwrite the original
        os.replace(new_path, img_path)
        print("Overwrote original image with fixed version.")
except Exception as e:
    print(f"Error processing image: {e}")
