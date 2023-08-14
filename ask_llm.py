from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time
import json
import re
from selenium.webdriver.common.by import By
import os



def post_login(username, password, driver):

    print('# Chat8 mainpage')
    url = 'https://cdnai.lexiao66.com/'
    driver.get(url)
    time.sleep(3)
    print('username, password')
    input_account = driver.find_element(By.XPATH, '//*[@id="van-field-1-input"]')
    input_psw = driver.find_element(By.XPATH, '//*[@id="van-field-2-input"]')

    input_account.send_keys(username)
    input_psw.send_keys(password)

    print('# login button')
    bt_logoin = driver.find_element(By.XPATH, '//*[@id="app"]/div[1]/div/div/div[2]/form/div[3]/button')
    bt_logoin.click()

def post_questions(questions, driver):
    print('Find target')

    que = ''
    num = 0
    for i in questions:
       que = que + i
       que = que + ','
       num+=1

    input_account = driver.find_element(By.CLASS_NAME, 'van-field__control')
    input_account.send_keys("Every 3 keyword belong to one picture, each picture belong to the same person, please write a sentence to describe all this " + str(num) + " keywords.")
    time.sleep(3)
    input_account.send_keys('These are the keywords: '+que)
    print('# Send!')
    bt_push = driver.find_element(By.CLASS_NAME, 'van-field__button')
    bt_push.click()
    return 1


def save_answers(driver):
    div_elements = driver.find_elements(By.CLASS_NAME, 'vuepress-markdown-body')
    last_div_element = div_elements[-1]  # 提取最后一个元素
    answer = last_div_element.text
    return answer




def ask_gpt(path,username,password,questions):

    ser = Service(path)
    chrome_options = webdriver.ChromeOptions()
    prefs = {"profile.default_content_setting_values.notifications": 2}
    chrome_options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(service=ser, options=chrome_options)
    driver.maximize_window()  # max window
    post_login(username, password, driver)
    time.sleep(8)
    post_questions(questions, driver)
    time.sleep(8)
    final_answer = save_answers(driver)
    return final_answer



if __name__ == '__main__':
    path_driver = '/Users/jin666/Desktop/jmy_generate/gpt-test/chromedriver_mac_arm64/chromedriver'  # diver
    username = "18201768019"
    password = "chat8app"
    question = ['tea','drink','student']
    o = ask_gpt(path_driver,username,password,question)
    print(o)

