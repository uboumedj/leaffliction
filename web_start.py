from fastapi import FastAPI, Form, Path, Request
from fastapi.responses import HTMLResponse
from uvicorn import run
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import locale
import re


app = FastAPI()
locale.setlocale(locale.LC_TIME, "fr_FR.utf8")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
dataset = './images'


@app.get("/", response_class=HTMLResponse)
async def search(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request}
    )


@app.get("/overview", response_class=HTMLResponse)
def overview(request: Request):
    return templates.TemplateResponse("overview.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
def overview(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


def main():
    run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()