from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import meteo_scraping_utils as msu
import time 
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os 
import glob
import shutil


def compute_coordinates(reference_coord, km_on_x, km_on_y, points_on_x, points_on_y):
    return msu.compute_coordinates(reference_coord, km_on_x, km_on_y, points_on_x, points_on_y)


def data_downloader(start_date, end_date, coordinates, move_path):
    
    path="chromedriver.exe"
    driver = webdriver.Chrome(path) 
    driver.get("https://open-meteo.com/en/docs/historical-weather-api")

    msu.input_dates(driver,  start_date, end_date) 
    msu.check_checkboxes(driver, 'hourly')
    
    max_attempts=15 #almost 15 seconds max to download the file, otherwise problems!! -> eventually increase
    index=0
    
    downloaded_coordinates=[]
    failed_download=[]
    
    while index < len(coordinates):
        print(f'downloading {index+1} of {len(coordinates)}')
        attempts = 0
        coord = coordinates[index]
        latitude, longitude=coord
        
        #print(f"trying to download lat: {latitude}, long: {longitude}\n")
        
        #input coordinates and download
        latitude_input, longitude_input=msu.input_coordinates(driver, latitude, longitude)
        download=WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "//form[@id='api_form']/div[@class='col-12']/button[3]")))
        driver.execute_script('arguments[0].click()',download)
        
        
        def pick_latest(attempts):
            try:
                list_of_files = glob.glob(f'C:/Users/pavesema/Downloads/era*.csv') 
                latest_file = max(list_of_files, key=os.path.getctime)
                return latest_file
            except:
                if attempts==max_attempts:
                    #print(f"max attempts to find lat: {latitude}, long: {longitude} reached")
                    return None
                time.sleep(1)
                attempts+=1
                return pick_latest(attempts)
            
        latest_file=pick_latest(attempts)
        
        msu.clear_inputs([latitude_input, longitude_input])
        index+=1
        
        if latest_file is None:
            print(f"failed to download lat: {latitude}, long: {longitude}\n")
            failed_download.append((latitude, longitude))
            continue
        
        #print(f"successfully downloaded lat: {latitude}, long: {longitude}\n")
        shutil.move(latest_file, f"{move_path}\{latitude}_{longitude}.csv")
        downloaded_coordinates.append(coord)
    
    driver.quit()
    
    
    return downloaded_coordinates, failed_download
    
    
        
