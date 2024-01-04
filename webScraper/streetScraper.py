from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import event_firing_webdriver as ef
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support import select
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.firefox.webdriver import WebDriver
from selenium.webdriver.common.keys import Keys
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta

opt = webdriver.FirefoxOptions()
driver = webdriver.Firefox(options=opt)  # Replace with the path to your WebDriver
lon, lat = 27.1720578,78.0419803
h_angle, t_angle = 240, 65
# TODO: check if lon lat or lat lon
t_angle = 65.39
h_angle = 240.39
# TODO wtf is 2a, 75y?
driver.get(f'https://www.google.com/maps/@{lon},{lat},2a,75y,{h_angle}h,{t_angle}t/data=!3m6!1e1!3m4!1s26Tu70cYpZecUZyQmKWWTg!2e0!7i13312!8i6656?hl=sv&entry=ttu')

# Get rid of banners etc and take a screenshot of canvas / export


# Save the image
driver.save_screenshot('screenshot.png')
#I want to only get the canvas element
canvases = driver.find_elements(By.TAG_NAME, 'canvas')
# Look for all canvases, pick out the correct one with the image based on some heuristic
for canvas in canvases:
    # Find the correct canvas
    pass

# I want to get the canvas as a png
canvas_png = canvas.screenshot_as_png
# I want to save the canvas as a png file
with open('canvas.png', 'wb') as f:
    f.write(canvas_png)
#I want to get the canvas as a base64 encoded string
canvas_base64 = canvas_png.encode('base64')
driver.quit()