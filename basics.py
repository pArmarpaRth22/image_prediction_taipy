from taipy import Gui
from tensorflow.keras import  models
from PIL import Image
import numpy as np

class_names = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}

model=models.load_model("model.keras")

def predict_image(model,path_to_image):
    img = Image.open(path_to_image)
    img = img.convert("RGB")
    img = img.resize((32, 32))
    data = np.asarray(img)
    data = data / 255
    probs = model.predict(np.array([data])[:1])

    top_prob = probs.max()
    top_pred = class_names[np.argmax(probs)]
    
    return top_prob, top_pred



path="placeholder_image.png"
content=""
val=""
prob=0
pred=""

page="""
<|text-center|
<|{"logo.png"}|image|>

<|{content}|file_selector|extensions=.png|width=25vw|>
select an image

<|{pred}|>

<|{path}|image|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>




>

"""
def on_change(state,var_name,var_value):
    if var_name=="content":
        state.path=var_value
        top_prob, top_pred = predict_image(model, var_value)
        state.prob = round(top_prob * 100)
        state.pred = "this is a " + top_pred
        state.img_path = var_value
    

if __name__=="__main__":
    app=Gui(page)
    app.run(use_reloader=True)