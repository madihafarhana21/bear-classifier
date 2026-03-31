__all__ = ['learn', 'classify_bear', 'categories', 'image', 'label', 'examples', 'intf']

# from fastai.vision.all import *
# import gradio as gr


# learn = load_learner('model.pkl')


# categories = ('Black', 'Grizzly', 'Teddy')

# def classify_bear(img):
#     pred, idx, probs = learn.predict(img) 
#     return dict(zip(categories, map(float,probs)))


# image = gr.Image(height=196, width=196)
# label = gr.Label()
# examples = ['teddy.jpg', 'black1.jpg', 'grizzly1.jpg', 'black.jpg', 'teddy1.jpg']

# intf = gr.Interface(fn=classify_bear, inputs=image, outputs=label, examples=examples)
# intf.launch(inline=False)


from fastai.vision.all import *
import gradio as gr
import pathlib
import platform
import numpy as np
import tempfile, os

if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

learn = load_learner('model.pkl')
categories = ('Black', 'Grizzly', 'Teddy')

def classify_bear(img):
    # Print exactly what Gradio is sending us
    print(f"TYPE: {type(img)}")
    print(f"VALUE: {img}")
    
    # Convert to numpy array first, then save as temp file
    # and pass the filepath to fastai — bypasses all transform issues
    if isinstance(img, dict):
        img = img.get("composite") or img.get("image") or list(img.values())[0]
    
    # Convert PIL/numpy to a temp file and predict from path
    # fastai handles file paths natively with no dict confusion
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        tmp_path = f.name
    
    if isinstance(img, np.ndarray):
        from PIL import Image as _PIL
        _PIL.fromarray(img).save(tmp_path)
    else:
        img.save(tmp_path)  # PIL Image
    
    try:
        pred, idx, probs = learn.predict(tmp_path)
    finally:
        os.unlink(tmp_path)
    
    return dict(zip(categories, map(float, probs)))

image = gr.Image(height=196, width=196, type="pil")
label = gr.Label()
examples = ['teddy.jpg', 'black1.jpg', 'grizzly1.jpg', 'black.jpg', 'teddy1.jpg']

intf = gr.Interface(
    fn=classify_bear,
    inputs=image,
    outputs=label,
    examples=examples,
    cache_examples=False
)
intf.launch(inline=False, share=True)