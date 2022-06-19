import argparse
import easydict
import pathlib
import pprint
import random
import requests
import time
import tqdm

from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from typing import Dict


YELLOW_COLOR = "\033[33m"
RESET_COLOR = "\033[0m"

WATING_PAGE_SEC = 5
MAX_SCROLL_TIME = 60


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data", 
        type=str,
        default="data",
        help=" ".join([
            "The path you save the crawled images."
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--max-iter", 
        type=int,
        default=1_000,
        help=" ".join([
            "The number of images you will crawl for each category."
            "Default=%(default)s",
        ]),
    )

    config = p.parse_args()
    return config


def get_target_items_and_urls() -> Dict[str, str]:
    ## Number of total categories: 10.
    return easydict.EasyDict({
        "apple":        "https://browse.gmarket.co.kr/list?category=300004557&k=24&p=1", ## 1
        "tomato":       "https://browse.gmarket.co.kr/list?category=300004555&k=24&p=1", ## 2
        # "grape":        "https://browse.gmarket.co.kr/list?category=300007874&k=24&p=1", ## 3
        "kiwi":         "https://browse.gmarket.co.kr/list?category=300004560&k=24&p=1", ## 4
        # "melon":        "https://browse.gmarket.co.kr/list?category=300024519&k=24&p=1", ## 5
        "plum":         "https://browse.gmarket.co.kr/list?category=300004559&k=24&p=1", ## 6
        "watermelon":   "https://browse.gmarket.co.kr/list?category=300007911&k=24&p=1", ## 7
        "Koreanmelon":  "https://browse.gmarket.co.kr/list?category=300007873&k=24&p=1", ## 8
        "peach":        "https://browse.gmarket.co.kr/list?category=300007908&k=24&p=1", ## 9
        "mango":        "https://browse.gmarket.co.kr/list?category=300028958&k=24&p=1", ## 10
        # "blueberry":    "https://browse.gmarket.co.kr/list?category=300028960&k=24&p=1", ## 11
        "orange":       "https://browse.gmarket.co.kr/list?category=300028963&k=24&p=1", ## 12
        # "grapefruit":   "https://browse.gmarket.co.kr/list?category=300028964&k=24&p=1", ## 13
        # "cherry":       "https://browse.gmarket.co.kr/list?category=300028965&k=24&p=1", ## 14
    })


def crawl(item_name: str, url: str, save_dir: pathlib.PosixPath, max_iter: int = 1_000) -> None:
    ## Set options.
    options = Options()

    # options.add_argument("headless")
    options.add_argument("lang=ko_KR")
    options.add_argument("--no-sandbox")
    options.add_argument("window-size=1280x800")
    options.add_argument("--disable-dev-shm-usage") ## save shm memory
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    n = 0 ## num of saved images
    p = 1 ## current page

    with tqdm.tqdm(total=max_iter, desc=f"Getting Items: {item_name}") as pbar:
        while n < max_iter:
            ## First try...
            if p == 1:
                ## Get a web page.
                driver.get(url)
            else:
                try:
                    ## Start from the botton of the page.
                    driver.find_element(by=By.CLASS_NAME, value="link__page-next").click()
                except:
                    ## If the left pages are none, then excute.
                    return
            
            ## Sleep.
            time.sleep(WATING_PAGE_SEC)

            ## Scroll down for wating lazy loading images.
            last_y = driver.execute_script("return document.body.scrollHeight")
            for i in range(MAX_SCROLL_TIME):
                y = 800 * (i+1) ## height of window size == 800

                driver.execute_script(f"window.scrollTo(0, {y});")
                time.sleep(1)

                ## If it scroll to down of page, then break.
                if y > last_y:
                    break                

            ## Search images.
            image_tags = driver.find_elements(by=By.TAG_NAME, value="img")
            random.shuffle(image_tags) ## shuffle for keep alike human

            for image_tag in image_tags:
                if n >= max_iter:
                    break

                ## Unique tags: '', 'image', 'image__grade-icon', 'image__item', 'image__item  ', 'image__logo', 'image__title'
                if image_tag.get_attribute("class") == "image__item  ":
                    ## Fetch a url.
                    image_url = image_tag.get_attribute("src")

                    ## Request the image file.
                    reponse = requests.get(image_url)
                    time.sleep(1)

                    if reponse.status_code == requests.codes["ok"]:
                        ## Write image.
                        save_path = save_dir / Path(f"{item_name}_{n}.jpeg")
                        with open(save_path,"wb") as file:
                            file.write(reponse.content)

                        ## Add a number.
                        pbar.update(1)
                        n += 1

            ## Search the next page.
            p += 1

    ## Quit the window.
    driver.quit()


def main(config: argparse.Namespace) -> None:
    def print_config(config: argparse.Namespace) -> None:
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(vars(config))
    print_config(config)

    ## Do crawling.
    for item_name, url in get_target_items_and_urls().items():
        margin = 1.2
        
        ## Make the directory to save data.
        save_dir = Path(config.data) / Path(item_name)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        ## Crawl it.
        crawl(
            item_name=item_name, 
            url=url, 
            save_dir=save_dir, 
            max_iter=int(config.max_iter * margin), ## with some margin
        )

    ## Print the result.
    print(YELLOW_COLOR + "Results:" + RESET_COLOR)
    for item_name in get_target_items_and_urls().keys():
        n = len(list(Path(config.data, item_name).glob("*.jpeg")))
        print(YELLOW_COLOR + f"  - {item_name}: {n}" + RESET_COLOR)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
