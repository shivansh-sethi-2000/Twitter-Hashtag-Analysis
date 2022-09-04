# Project name
Twitter Hashtag Trend Analysis Tool
## Team Members
- Shivansh Sethi
- Rohit

## Tool Description

The Main purpose of the tool is for users to analyse bot/propoganda based behaviour during a given timeline on a certain hashtag.
There are very few tools currently which provide an all in one analysis, so we tried to make one that not only helps the user to analyse/visualize the data but also does some analysis itself by maintaining score of the users found based on some parameters that can helps user to filter out the accounts for further analysis.

## Installation

### As this repo contains some git lfs objects and the git lfs bandwidth is limited you have to follow the below steps to doenload the repo.

1. Fork the Following Repository  https://github.com/shivansh-sethi-2000/BellingCat.git

2. Go to your repo Settings

3. Find "Include Git LFS objects in archives" under the Archives section and check it
        ![What is this](images/archive.png)

4. Return to the archived repository. 

5. Download as .zip

6. Download will pause for a minute or so before it starts downloading lfs objects. Wait and it should continue.

7. Move to the tool's directory and install the required Packages

        cd BellingCat
        pip install -r req.txt

8. add your API keys and sceret to my_tokens.py file

- you can also Download the cardiffnlp and universal-sentence-encoder_4 folders from the link : https://drive.google.com/drive/folders/1mOe2WVAit0AakFINY3k1iaVLoP4ExO8n?usp=sharing
- just place the cardiffnlp and universal-sentence-encoder_4 folders in the same directory.

## Usage
1. To run the Script use the following command

        streamlit run Hashtag_Trend_Script.py

2. It will Take you to your Web Page If not you can use the Url showing in your terminal

3. The HomePage will Look Like this
    ![What is this](images/main.png)

## Additional Information
- Next Steps would include adding network graphs of filtered Users for better analysis and image tweet analysis.
- there are some restrictions in using twitter API and same are applied here. Also, the text pre processing might take a liitle time for medium-large datasets.
- After seeing a lot bot/propoganda behaviour of accounts there should be a tool available to general public to filter those accounts and any information they might share.
