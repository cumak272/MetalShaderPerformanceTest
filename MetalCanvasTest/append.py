import os
import sys

base_dir = '/Users/cumaalikesici/IOS Projects/MetalCanvasTest/MetalCanvasTest/'
with open(os.path.join(base_dir, 'Shaders.metal'), 'a') as target:
    for filename in ['P2.metal', 'P3.metal', 'P4.metal']:
        filepath = os.path.join(base_dir, filename)
        with open(filepath, 'r') as f:
            target.write(f.read() + '\n')
            
for filename in ['P2.metal', 'P3.metal', 'P4.metal']:
    os.remove(os.path.join(base_dir, filename))
