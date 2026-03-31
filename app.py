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

# Fix 1: WindowsPath on Linux
if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

learn = load_learner('model.pkl')
categories = ('Black', 'Grizzly', 'Teddy')

def classify_bear(img):
    img = PILImage.create(img)
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

image = gr.Image(height=196, width=196, type="pil")
label = gr.Label()
examples = ['teddy.jpg', 'black1.jpg', 'grizzly1.jpg', 'black.jpg', 'teddy1.jpg']

intf = gr.Interface(
    fn=classify_bear,
    inputs=image,
    outputs=label,
    examples=examples,
    cache_examples=False   # Fix 2: disable caching that triggers the bug at startup
)
intf.launch(inline=False, share=True)
