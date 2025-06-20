import numpy as np
from noise import pnoise2
from PIL import Image

def save_as_png(data, filename):
    # Normalize to [0, 255] for grayscale image
    data = (data - data.min()) / (data.max() - data.min())
    img = Image.fromarray(np.uint8(data * 255), mode='L')
    img.save(filename)

def generate_perlin_png(filename="perlin.png", size=256, scale=10.0):
    data = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            data[i, j] = pnoise2(i / scale, j / scale, octaves=4)
    save_as_png(data, filename)

def generate_gaussian_png(filename="gaussian.png", size=256, mean=0.0, std=0.05):
    data = np.random.normal(mean, std, (size, size))
    save_as_png(data, filename)

def generate_fractal_png(filename="fractal.png", size=256, scale=10.0, octaves=6, persistence=0.5, lacunarity=2.0):
    data = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            freq = scale
            amp = 1.0
            value = 0.0
            for _ in range(octaves):
                value += amp * pnoise2(i / freq, j / freq)
                freq *= lacunarity
                amp *= persistence
            data[i, j] = value
    save_as_png(data, filename)

# 生成 PNG 图片
generate_perlin_png("perlin.png")
generate_gaussian_png("gaussian.png")
generate_fractal_png("fractal.png")
