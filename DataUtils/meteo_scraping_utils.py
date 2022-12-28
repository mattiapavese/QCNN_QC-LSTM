from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from haversine import inverse_haversine, Direction
import numpy as np



def input_coordinates(driver, latitude, longitude):
    
    latitude_input = driver.find_element(by='id',value="latitude")
    for _ in range(10):
        latitude_input.send_keys(Keys.DELETE)
    latitude_input.send_keys(latitude)
    

    longitude_input = driver.find_element(by="id", value="longitude")
    for _ in range(10):
        longitude_input.send_keys(Keys.DELETE)
    longitude_input.send_keys(longitude)
    
    
    return latitude_input, longitude_input

def clear_inputs(inputs):
    for input in inputs:
        input.clear()
        
        
def input_dates(driver, start_date, end_date):
    
    start_date_input = driver.find_element(by='id',value="start_date")
    start_date_input.send_keys(Keys.CONTROL, 'a')
    for _ in range(10):
        start_date_input.send_keys(Keys.DELETE)
    start_date_input.send_keys(start_date)
    
    end_date_input = driver.find_element(by='id',value="end_date")
    end_date_input.send_keys(Keys.CONTROL, 'a')
    for _ in range(10):
        end_date_input.send_keys(Keys.DELETE)
    end_date_input.send_keys(end_date)
    
def check_checkboxes(driver, name):
    checkboxes=driver.find_elements(By.NAME, name )
    for checkbox in checkboxes:
        if not checkbox.is_selected():
            driver.execute_script('arguments[0].click()', checkbox)

def compute_coordinates(lat_long_start, tot_dist_x, tot_dist_y, n_points_x, n_points_y):
    
    d_x = tot_dist_x / (n_points_x - 1)
    d_y = tot_dist_y / (n_points_y - 1)
    
    coordinates = []
    row_start = lat_long_start
    
    for _ in range(n_points_y):
        for x in range(n_points_x):
            coord = inverse_haversine(row_start,x*d_x,Direction.EAST)
            coord = tuple([round(c,2) for c in coord])
            coordinates.append(coord)
        row_start = inverse_haversine(row_start, d_y, Direction.SOUTH)
    
    return coordinates
        
            
            
            
    
    
    
    
    
    
    
    