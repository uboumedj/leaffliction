from fastapi import FastAPI, Form, Path, Request
from fastapi.responses import HTMLResponse
from uvicorn import run
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from web_predict import generate_transformed_images, load_image
from cli_predict import soft_vote, hard_vote
from shutil import rmtree
import os
import locale
import glob
import random
import joblib


app = FastAPI()
locale.setlocale(locale.LC_TIME, "fr_FR.utf8")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
dataset = './images'


@app.get("/", response_class=HTMLResponse)
async def search(request: Request):

    image_folder = "static/resources/images/"
    image_amount = 50
    all_images = glob.glob(image_folder + '/**/*.JPG', recursive=True)
    images = random.sample(all_images, min(image_amount, len(all_images)))

    return templates.TemplateResponse(
        "index.html", {"request": request,
                       "images": images,
                       "image_folder": image_folder}
    )


@app.get("/overview", response_class=HTMLResponse)
def overview(request: Request):
    return templates.TemplateResponse("overview.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, image_path: str = Form(...)):
    model = joblib.load(filename="leaffliction.joblib")
    if os.path.exists("static/tmp"):
        rmtree("static/tmp")

    generate_transformed_images(image_path, destination="static/tmp")

    predictions = []
    transformations = sorted(os.listdir("static/tmp"))
    for i in range(len(transformations)):
        prediction = model.predict(load_image("static/tmp/" + transformations[i]))
        predictions.append(prediction[0])

    s_vote = soft_vote(predictions)
    h_vote = hard_vote(predictions)
    classes = sorted(os.listdir(os.path.dirname(os.path.dirname(image_path))))

    return templates.TemplateResponse("predict.html", {"request": request,
                                                       "class": classes[s_vote],
                                                       "transformations": transformations})


def main():
    run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()