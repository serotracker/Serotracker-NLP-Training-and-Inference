from run_opot import COVIDENCE_REVIEW_ID
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
import shutil
from dotenv import load_dotenv


import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', type=str)
    parser.add_argument('--password', type=str)
    parser.add_argument('--mode', type=str)
    load_dotenv('.env')

    COVIDENCE_REVIEW_ID = os.getenv('COVIDENCE_REVIEW_ID')

    args = parser.parse_args()

    if args.mode == 'inference':
        args.download_folder = str(pathlib.Path().resolve()) + r"\csvs_for_inference"
    elif args.mode == 'training':
        args.download_folder = str(pathlib.Path().resolve()) + r"\csvs_for_training"

    def enable_download_headless(browser,download_dir):
        browser.command_executor._commands["send_command"] = ("POST", '/session/$sessionId/chromium/send_command')
        params = {'cmd':'Page.setDownloadBehavior', 'params': {'behavior': 'allow', 'downloadPath': download_dir}}
        browser.execute("send_command", params)

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

    for filename in os.listdir(args.download_folder):
        file_path = os.path.join(args.download_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-dev-shm-usage')

    chrome_options.add_experimental_option("prefs", {
            "download.default_directory": args.download_folder,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing_for_trusted_sources_enabled": False,
            "safebrowsing.enabled": False
    })

    driver = webdriver.Chrome(ChromeDriverManager().install(), options = chrome_options)

    enable_download_headless(driver, args.download_folder)

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


    #change to export page
    driver.get('https://app.covidence.org/reviews/{}/exports/new'.format(COVIDENCE_REVIEW_ID))

    format_selector = Select(wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="citation_export_format"]'))))
    format_selector.select_by_visible_text("CSV")

    if args.mode == 'inference':
        stages = ['Title and abstract screening']
    elif args.mode == 'training':
        stages = ['Title and abstract screening',
            'Full text review',
            'Included',
            'Excluded',
            'Irrelevant']


    #download each stage in csv
    for stage in stages:
        #get current number of export buttons
        number_of_export_buttons = len(driver.find_elements_by_xpath('//*[@id="export-history"]/div/table/tbody/tr'))

        #select stage and click
        stage_selector = Select(wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="citation_export_category"]'))))
        stage_selector.select_by_visible_text(stage)

        export_button = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="export-references"]/div[3]/button')))
        export_button.click()
        time.sleep(1)

        #wait until number of export buttons increases
        condition = elements_length_changes((By.XPATH, '//*[@id="export-history"]/div/table/tbody/tr'), number_of_export_buttons)
        WebDriverWait(driver, 5).until(condition)
        
        #wait until download button shows
        long_wait = WebDriverWait(driver, 120) #up to 2 minutes
        download_button = long_wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="export-history"]/div/table/tbody/tr[1]/td[4]/button')))
        download_button.click()

        time.sleep(30) #give it 30s to download
        
        #rename file
        f = glob(os.path.join(args.download_folder,"review_*"))[0]
        os.rename(f, os.path.join(args.download_folder, stage.lower().replace(' ', '_') + '.csv'))
