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
import torch
from torchvision import transforms

# Fix WindowsPath issue
if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

learn = load_learner('model.pkl', cpu=True)
learn.model.eval()

categories = ('Black', 'Grizzly', 'Teddy')

tfm = transforms.Compose([
    transforms.Resize((196, 196)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def classify_bear(img):
    img = tfm(img).unsqueeze(0)  # shape: (1, 3, 196, 196)

    with torch.no_grad():
        preds = learn.model(img)
        probs = torch.softmax(preds, dim=1)[0]

    return dict(zip(categories, map(float, probs)))

image = gr.Image(height=196, width=196, type="pil")
label = gr.Label()

intf = gr.Interface(
    fn=classify_bear,
    inputs=image,
    outputs=label,
    examples=['teddy.jpg', 'black1.jpg', 'grizzly1.jpg', 'black.jpg', 'teddy1.jpg'],
    cache_examples=False
)

intf.launch(inline=False, share=True)