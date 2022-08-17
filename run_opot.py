from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from selenium.webdriver.chrome.options import Options
import os
from glob import glob
import argparse
import pathlib
import hashlib
from abstract_prep import prepare_abstract
from dotenv import load_dotenv

import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', type=str)
    parser.add_argument('--password', type=str)
    load_dotenv('.env')

    COVIDENCE_REVIEW_ID = os.getenv('COVIDENCE_REVIEW_ID')


    args = parser.parse_args()

    PREDICTION_FILE_PATH = './output/all_predictions.txt'
    DECISION_THRESHOLD = 0.25
    prediction_dict = {}
    file = open(PREDICTION_FILE_PATH, 'r')
    lines = file.readlines()[1:]
    file.close()

    request = []
    key_set = set()

    for i, line in enumerate(lines):
        blocks = line.split('\t')
        key = blocks[0]

        if(len(blocks) <= 1):
            continue

        if(key in key_set):
            continue

        key_set.add(key)
        inclusionLikelihood = float(blocks[1])
        prediction_dict[key] = inclusionLikelihood

    class elements_length_changes(object):
      """An expectation for checking that an elements has changes.

      locator - used to find the element
      returns the WebElement once the length has changed
      """
      def __init__(self, locator, length):
        self.locator = locator
        self.length = length

      def __call__(self, driver):
        element = driver.find_elements(*self.locator)
        element_count = len(element)
        if element_count > self.length:
          return element
        else:
          return False


    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    # chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-dev-shm-usage')

    chrome_options.add_experimental_option("prefs", {
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing_for_trusted_sources_enabled": False,
            "safebrowsing.enabled": False
    })

    driver = webdriver.Chrome(ChromeDriverManager().install(), options = chrome_options)

    #login
    driver.get('https://app.covidence.org/sign_in')
    wait = WebDriverWait(driver, 3)
    username_element = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="session_email"]')))
    username_element.send_keys(args.username)

    password_element = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="session_password"]')))
    password_element.send_keys(args.password)

    login_button = wait.until(EC.visibility_of_element_located((By.XPATH, '/html/body/div[3]/div/div/div/div[2]/div/div/form/div[3]/input')))
    login_button.click()

    time.sleep(3)

    n_clicked = 0


    #change to export page
    n_attempts = 8
    for n in range(n_attempts):
        print("Going through pass {}".format(n))
        driver.get('https://app.covidence.org/reviews/{}/review_studies/screen?filter=vote_required_from'.format(COVIDENCE_REVIEW_ID))
        time.sleep(8)
        

        table_element = driver.find_element_by_xpath('//*[@id="reviews"]/div/div/div[3]/table')
        max_abstract_index = len(driver.find_element_by_xpath('//*[@id="reviews"]/div/div/div[3]/table').find_elements_by_xpath("./*"))//2
        
        has_more_abstracts = True

        abstract_index = 0
        while has_more_abstracts:
            while abstract_index < max_abstract_index:
                element_index = 2 * abstract_index + 1
                title_element = driver.find_element_by_xpath('//*[@id="reviews"]/div/div/div[3]/table/tbody[{}]/tr/td[2]/div[2]/div/div[1]'.format(element_index))
                no_button = driver.find_element_by_xpath('//*[@id="reviews"]/div/div/div[3]/table/tbody[{}]/tr/td[3]/div[1]/button[1]'.format(element_index))
                yes_button = driver.find_element_by_xpath('//*[@id="reviews"]/div/div/div[3]/table/tbody[{}]/tr/td[3]/div[1]/button[3]'.format(element_index))
                
                title = title_element.text
                text = prepare_abstract(title, '')
                hashed_title = str(hashlib.sha256(text.encode('utf-8')).hexdigest())

                if hashed_title in prediction_dict:
                    # print('clicking!')
                    if prediction_dict[hashed_title] >= DECISION_THRESHOLD:
                        yes_button.click()
                    else:
                        no_button.click()

                    n_clicked += 1
                    time.sleep(7)
                    # no_button.click()
                else:
                    #only increment index if a button was pressed
                    abstract_index += 1

                table_element = driver.find_element_by_xpath('//*[@id="reviews"]/div/div/div[3]/table')
                max_abstract_index = len(table_element.find_elements_by_xpath("./*"))//2

            more_button = driver.find_element_by_xpath('//*[@id="reviews"]/div/div/div[4]/div/a')
            try:
                more_button.click()
            except:
                has_more_abstracts = False
            time.sleep(4)
            table_element = driver.find_element_by_xpath('//*[@id="reviews"]/div/div/div[3]/table')
            max_abstract_index = len(table_element.find_elements_by_xpath("./*"))//2
        
        if n_clicked == len(list(prediction_dict.keys())):
            print("found all abstracts, quitting early")
            break

    print("done!")
