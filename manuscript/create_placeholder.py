from PIL import Image, ImageDraw

img = Image.new('RGB', (800, 600), color = (73, 109, 137))
d = ImageDraw.Draw(img)
d.text((10,10), "Software Architecture Placeholder", fill=(255,255,0))

img.save('./figures/software_architecture.png')
print("Created placeholder image.")
