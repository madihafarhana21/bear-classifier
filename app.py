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
from PIL import Image as PilImg  # different alias to avoid fastai conflict

if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

learn = load_learner('model.pkl')
categories = ('Black', 'Grizzly', 'Teddy')

def classify_bear(img):
    if isinstance(img, dict):
        img = img.get("image") or img.get("composite") or list(img.values())[0]
    if not isinstance(img, PilImg.Image):  # use the new alias
        img = PilImg.fromarray(img)
    
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
    cache_examples=False
)
intf.launch(inline=False, share=True)