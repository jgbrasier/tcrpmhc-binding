from urllib.request import urlopen
import os
import re
from bs4 import BeautifulSoup
import csv

# URL of the page containing the table
url = 'https://sysimm.org/immune-scape/vdjdb-models'

# Directory to save the files
directory = '/Users/jgbrasier/Downloads/vdjdb_models'

# Make the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Send a request to the URL and get the response
html = urlopen(url)
# Check if the request was successful
if html.status == 200:
    # Parse the HTML content of the page
    soup = BeautifulSoup(html, 'html.parser')
    
    # Find the table containing the files
    table = soup.find_all('table')[0]
    rows = table.find_all('tr')

    File = open(os.path.join(directory, 'data.csv'), 'wt+')
    Data = csv.writer(File)
    # Loop through all the rows in the table
    for row in rows:
        filtered_row = []
        # Get the cells in the row
        for cell in row.find_all(['td', 'th']):
            filtered_row.append(cell.get_text())
        Data.writerow(filtered_row)
    File.close()
        # # Get the URL of the file
        # file_url = 'https://sysimm.org' + filename
        
        # # Download the file
        # file_response = requests.get(file_url)
        
        # Save the file to disk
        # with open(os.path.join(directory, filename.split('/')[-1]), 'wb') as f:
        #     f.write(file_response.content)

else:
    print(f"Request failed with status code {html.status}")
