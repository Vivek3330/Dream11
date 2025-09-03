import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
import numpy as np
import warnings
import sys

warnings.filterwarnings("ignore")

# Creating the player mapping dictionary

player_map = {'Vansh Bedi': 1413379,
 'Andre Siddharth': 1440190,
 'Ramakrishna Ghosh': 1339053,
 'Shaik Rasheed': 1292497,
 'Gurjapneet Singh': 1269869,
 'Matheesha Pathirana': 1194795,
 'Noor Ahmad': 1182529,
 'Anshul Kamboj': 1175428,
 'Nathan Ellis': 826915,
 'Mukesh Choudhary': 1125688,
 'Ruturaj Gaikwad': 1060380,
 'Kamlesh Nagarkoti': 1070188,
 'Rachin Ravindra': 959767,
 'Khaleel Ahmed': 942645,
 'Shivam Dube': 714451,
 'Rahul Tripathi': 446763,
 'Sam Curran': 662973,
 'Shreyas Gopal': 344580,
 'Deepak Hooda': 497121,
 'Devon Conway': 379140,
 'Jamie Overton': 510530,
 'Vijay Shankar': 477021,
 'Ravichandran Ashwin': 26421,
 'Ravindra Jadeja': 234675,
 'MS Dhoni': 28081,
 'Madhav Tiwari': 1460385,
 'Manvanth Kumar L': 1392186,
 'Tripurana Vijay': 1292527,
 'Vipraj Nigam': 1449074,
 'Tristan Stubbs': 595978,
 'Abishek Porel': 1277545,
 'Ashutosh Sharma': 1131978,
 'Donovan Ferreira': 698315,
 'Sameer Rizvi': 1175489,
 'Jake Fraser-McGurk': 1168049,
 'Ajay Mandal': 1059570,
 'Darshan Nalkande': 1111917,
 'T Natarajan': 802575,
 'Mukesh Kumar': 926851,
 'Kuldeep Yadav': 559235,
 'Mohit Sharma': 537119,
 'Lokesh Rahul': 422108,
 'Axar Patel': 554691,
 'Dushmantha Chameera': 552152,
 'Karun Nair': 398439,
 'Faf du Plessis': 44828,
 'Mitchell Starc': 311592,
 'Gurnoor Brar Singh': 1287033,
 'Nishant Sindhu': 1292506,
 'Arshad Khan': 1244751,
 'Sai Sudharsan': 1151288,
 'Kumar Kushagra': 1207295,
 'Manav Suthar': 1175426,
 'Sherfane Rutherford': 914541,
 'Gerald Coetzee': 596010,
 'Anuj Rawat': 1123073,
 'Kulwant Khejroliya': 1083033,
 'Shubman Gill': 1070173,
 'Ravisrinivasan Sai Kishore': 1048739,
 'Mahipal Lomror': 853265,
 'Karim Janat': 793467,
 'Washington Sundar': 719715,
 'Mohammed Siraj': 940973,
 'Glenn Phillips': 823509,
 'Rashid-Khan': 793463,
 'Prasidh Krishna': 917159,
 'Jayant Yadav': 447587,
 'Rahul Tewatia': 423838,
 'Kagiso Rabada': 550215,
 'Jos Buttler': 308967,
 'Ishant Sharma': 236779,
 'Harshit Rana': 1312645,
 'Angkrish Raghuvanshi': 1292495,
 'Vaibhav Arora': 1209292,
 'Luvnith Sisodia': 1155253,
 'Mayank Markande': 1081442,
 'Chetan Sakariya': 1131754,
 'Rahmanullah Gurbaz': 974087,
 'Spencer Johnson': 1123718,
 'Varun Chakravarthy': 1108375,
 'Anrich Nortje': 481979,
 'Ramandeep Singh': 1079470,
 'Rovman Powell': 820351,
 'Rinku Singh': 723105,
 'Venkatesh Iyer': 851403,
 'Moeen Ali': 8917,
 'Quinton de Kock': 379143,
 'Andre Russell': 276298,
 'Sunil Narine': 230558,
 'Manish Pandey': 290630,
 'Ajinkya Rahane': 277916,
 'Umran Malik': 1246528,
 'Digvesh Singh': 1460529,
 'Prince Yadav': 1300836,
 'Shamar Joseph': 1356971,
 'Mayank Yadav': 1292563,
 'Arshin Kulkarni': 1403153,
 'Akash Deep': 1176959,
 'Akash Singh': 1175458,
 'Yuvraj Chaudhary': 1175463,
 'Ravi Bishnoi': 1175441,
 'Abdul Samad': 1175485,
 'Shahbaz Ahmed': 1159711,
 'Rajvardhan Hangargekar': 1175429,
 'Ayush Badoni': 1151270,
 'Aryan Juyal': 1130300,
 'Mohsin Khan': 1132005,
 'Matthew Breetzke': 595267,
 'Manimaran Siddharth': 1151286,
 'Rishabh Pant': 931581,
 'Himmat Singh': 805235,
 'Aiden Markram': 600498,
 'Avesh Khan': 694211,
 'Nicholas Pooran': 604302,
 'David Miller': 321777,
 'Mitchell Marsh': 272450,
 'Bevon Jacobs': 1410577,
 'Naman Dhir': 1287032,
 'Robin Minz': 1350762,
 'Raj Angad Bawa': 1292502,
 'Vignesh Puthur': 1460388,
 'Satyanarayana Raju': 1392201,
 'Ashwani Kumar': 1209126,
 'Tilak Varma': 1170265,
 'KL Shrijith': 778241,
 'Mujeeb-ur-Rahman': 974109,
 'Ryan Rickelton': 605661,
 'Arjun Tendulkar': 1148776,
 'Will Jacks': 897549,
 'Hardik Pandya': 625371,
 'Mitchell Santner': 502714,
 'Reece Topley': 461632,
 'Corbin Bosch': 594322,
 'Jasprit Bumrah': 625383,
 'Trent Boult': 277912,
 'Suryakumar Yadav': 446507,
 'Deepak Chahar': 447261,
 'Karn Sharma': 30288,
 'Rohit Sharma': 34102,
 'Musheer Khan': 1316430,
 'Harnoor Singh Pannu': 1292496,
 'Pyla Avinash': 1324449,
 'Suryansh Shedge': 1339698,
 'Harpreet Brar': 1168641,
 'Priyansh Arya': 1175456,
 'Kuldeep Sen': 1163695,
 'Marco Jansen': 696401,
 'Nehal Wadhera': 1151273,
 'Prabhsimran Singh': 1161024,
 'Aaron Hardie': 1124283,
 'Arshdeep Singh': 1125976,
 'Azmatullah Omarzai': 819429,
 'Vishnu Vinod': 732293,
 'Xavier Bartlett': 1050545,
 'Shashank Singh': 377534,
 'Lockie Ferguson': 493773,
 'Josh Inglis': 662235,
 'Vyshak Vijaykumar': 777815,
 'Pravin Dubey': 777515,
 'Yash Thakur': 1070196,
 'Shreyas Iyer': 642519,
 'Marcus Stoinis': 325012,
 'Yuzvendra Chahal': 430246,
 'Glenn Maxwell': 325026,
 'Abhinandan Singh': 1449085,
 'Swastik Chikara': 1403198,
 'Mohit Rathee': 1349361,
 'Suyash Sharma': 1350792,
 'Jacob Bethell': 1194959,
 'Rasikh Salam': 1161489,
 'Yash Dayal': 1159720,
 'Manoj Bhandage': 1057399,
 'Nuwan Thushara': 955235,
 'Romario Shepherd': 677077,
 'Tim David': 892749,
 'Devdutt Padikkal': 1119026,
 'Krunal Pandya': 471342,
 'Rajat Patidar': 823703,
 'Jitesh Sharma': 721867,
 'Lungi Ngidi': 542023,
 'Philip Salt': 669365,
 'Liam Livingstone': 403902,
 'Josh Hazlewood': 288284,
 'Bhuvneshwar Kumar': 326016,
 'Swapnil Singh': 232292,
 'Virat Kohli': 253802,
 'Vaibhav Suryavanshi': 1408688,
 'Ashok Sharma': 1299879,
 'Kwena Maphaka': 1294342,
 'Kunal Singh Rathore': 1339031,
 'Akash Madhwal': 1206039,
 'Shubham Dubey': 1252585,
 'Yudhvir Singh Charak': 1206052,
 'Maheesh Theekshana': 1138316,
 'Dhruv Jurel': 1175488,
 'Kumar Kartikeya': 1159843,
 'Yashasvi Jaiswal': 1151278,
 'FazalHaq Farooqi': 974175,
 'Riyan Parag': 1079434,
 'Tushar Deshpande': 822553,
 'Jofra Archer': 669855,
 'Wanindu Hasaranga': 784379,
 'Nitish Rana': 604527,
 'Shimron Hetmyer': 670025,
 'Sandeep Sharma': 438362,
 'Sanju Samson': 425943,
 'Aniket Verma': 1409976,
 'Eshan Malinga': 1306214,
 'K Nitish Reddy': 1175496,
 'Abhinav Manohar': 778963,
 'Atharva Taide': 1125958,
 'Simarjeet- Singh': 1159722,
 'Rahul Chahar': 1064812,
 'Abhishek Sharma': 1070183,
 'Kamindu Mendis': 784373,
 'Wiaan Mulder': 698189,
 'Zeeshan Ansari': 942371,
 'Ishan Kishan': 720471,
 'Heinrich Klaasen': 436757,
 'Sachin Baby': 432783,
 'Travis Head': 530011,
 'Adam Zampa': 379504,
 'Harshal Patel': 390481,
 'Pat Cummins': 489889,
 'Mohammed Shami': 481896,
 'Jaydev Unadkat': 390484,
 'Ayush Mhatre': 1452455,
 'Dewald Brevis': 1070665,
'Shardul Thakur': 475281}

# Function to extract Playing 11 from the Excel file

def get_playing_xi(file_path, match_number):
    xls = pd.ExcelFile(file_path)
    sheet_name = f"Match_{match_number}"

    if sheet_name not in xls.sheet_names:
        print(f"Sheet {sheet_name} not found!")
        return None

    df = pd.read_excel(xls, sheet_name=sheet_name)
    playing_xi = df[df["IsPlaying"]=="PLAYING"]
    playing_xi = playing_xi.sort_values(by="lineupOrder")

    return playing_xi

# Function to fetch the stats of each player

def fetch_player_stats(player_name, player_type, stats_dict):
    if player_name not in player_map:
        print(f" Player {player_name} not found in mapping!")
        return None

    player_id = player_map[player_name]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Initialize the player's stats if not already present
    if player_name not in stats_dict:
        stats_dict[player_name] = {"Batting": None, "Bowling": None, "Fielding": None}

    # Function to scrape data from the website
    def scrape_data(url, stat_type):
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        tables = soup.find_all("table", class_="engineTable")
        if len(tables) >= 4:
            data =  pd.read_html(str(tables[3]))[0]
            return data.head(16).iloc[::-1]
        return None

    # Fetch batting stats if applicable
    if player_type in ["BAT", "ALL","WK","BOWL"]:
        batting_url = f"https://stats.espncricinfo.com/ci/engine/player/{player_id}.html?class=6;host=6;orderby=start;orderbyad=reverse;template=results;type=batting;view=innings"
        batting_data = scrape_data(batting_url, "Batting")
        # columns_for_batting = ["4s", "6s", "SR", "Runs","BF"] 
        
        if batting_data is not None:
            stats_dict[player_name]["Batting"] = batting_data

    # Fetch bowling stats if applicable
    if player_type in ["BOWL", "ALL"]:
        bowling_url = f"https://stats.espncricinfo.com/ci/engine/player/{player_id}.html?class=6;host=6;orderby=start;orderbyad=reverse;template=results;type=bowling;view=innings"
        bowling_data = scrape_data(bowling_url, "Bowling")
        
        if bowling_data is not None:
            stats_dict[player_name]["Bowling"] = bowling_data
            
    if player_type in ["BAT", "ALL","WK","BOWL"]:
        fielding_url = f"https://stats.espncricinfo.com/ci/engine/player/{player_id}.html?class=6;host=6;orderby=start;orderbyad=reverse;template=results;type=fielding;view=innings"
        fielding_data = scrape_data(fielding_url, "Fielding")
        
        if fielding_data is not None:
            stats_dict[player_name]["Fielding"] = fielding_data


# Providing the excel path and match number

# file_path = "SquadPlayerNames_IndianT20League.xlsx"
file_path = "SquadPlayerNames_IndianT20League.xlsx"
match_number = 57 # Change as needed

# Getting player name and player type

main_df = get_playing_xi(file_path, match_number)
playing_22 = main_df[["Player Name", "Player Type"]].values.tolist()

# Dictionary to store stats for all players
player_stats = {}

if playing_22:
    for player_name, player_type in playing_22:
        fetch_player_stats(player_name, player_type, player_stats)

# Creating lag 1 series

def create_lagged_features(scores, lag=1):
    X, y = [], []
    for i in range(len(scores) - lag):
        X.append(scores[i:i+lag])
        y.append(scores[i+lag])
    return np.array(X).reshape(-1,1), np.array(y)

# Applying linear regression on the lag 1 series

def train_and_predict(scores):
    if len(scores) < 3:
        return "N/A"  # Return N/A if not enough data

    X, y = create_lagged_features(scores)

    model = LinearRegression()
    model.fit(X, y)
    
    last_5_scores = np.array(np.mean(scores[-5:])).reshape(1, -1)
    next_score = model.predict(last_5_scores)
    
    # Making sure that negative values or not predicted
    if next_score[0] < 0:
        return 0
    return round(next_score[0], 2)  # Rounding off to 2 decimals

columns_to_predict = ["4s", "6s", "SR", "Runs", "Wkts", "Econ", "Ct","St","BF"]

# Initiating a dictionary to store predictions for all players
all_predictions = {}

# Loop through each player in player_stats

for player_name, stats in player_stats.items():
    # Initializing predictions dictionary for this player
    
    player_type = stats.get("player_type", "UNKNOWN")  # Edge case if no player type given then we take it as unknown
    predictions = {col: "N/A" for col in columns_to_predict}  # filling n/a as default values for every column

    # Checking if batting data exists
    if "Batting" in stats and stats["Batting"] is not None:
        df_batting = stats["Batting"]  # DataFrame for batting stats

        for col in ["4s", "6s", "SR", "Runs","BF"]:
            if col in df_batting.columns:
                df_batting[col] = df_batting[col].astype(str).str.replace("*", "", regex=False)  # Removing * from score as if player is not out then the score contains *
                df_batting[col] = df_batting[col].replace(["-", "DNB", "TDNB", "sub"], np.nan)  # Replacing invalid values
    
                if player_type == "BOWL":
                    df_batting[col] = df_batting[col].fillna(0) # Filling na values in batting stats with 0 for bowlers
                    
                 # If player is not blower then we are filling na values in batting stats with mean
                else:
                    df_batting[col] = pd.to_numeric(df_batting[col], errors='coerce')
        
                    mean_value = df_batting[col].mean()
                    
                    if np.isnan(mean_value): # Just to handle corner cases
                        mean_value = 0
                    
                    df_batting[col] = df_batting[col].fillna(mean_value)  # Filling na values with mean
    
                scores = df_batting[col].dropna().astype(float).values
                # Storing all scores as avg of 3 matches so that scores wont deviate more, This is the main idea of our model
                window_size = 3
                scores = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
                predictions[col] = train_and_predict(scores) # Finding the predicted batting stats 

    # Checking if bowling data exists
    if "Bowling" in stats and stats["Bowling"] is not None:
        df_bowling = stats["Bowling"] # Dataframe for bowling

        for col in ["Wkts", "Econ"]:
            if col in df_bowling.columns:
                df_bowling[col] = df_bowling[col].replace(["TDNB"], np.nan)
                df_bowling[col] = pd.to_numeric(df_bowling[col], errors='coerce')
                mean_value = df_bowling[col].mean()
                    
                if np.isnan(mean_value): # Just to handle corner cases
                    mean_value = 0
                    
                df_bowling[col] = df_bowling[col].fillna(mean_value)

                df_bowling[col] = df_bowling[col].replace(["-","DNB", "sub","absent"], np.nan)  # Replace invalid values
                df_bowling[col] = df_bowling[col].astype(float)  # Convert to float

                df_bowling[col] = df_bowling[col].fillna(0)  # Fill with mean
                
                scores = df_bowling[col].dropna().astype(float).values
                # Storing all scores as avg of 3 matches so that scores wont deviate more, This is the main idea of our model
                window_size = 3
                scores = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
                predictions[col] = train_and_predict(scores) # Finding the predicted bowling stats

    # Checking if fielding data exists
    if "Fielding" in stats and stats["Fielding"] is not None:
        df_fielding = stats["Fielding"]  # DataFrame for fielding


        for col in ["Ct","St"]:
            if col in df_fielding.columns:
                df_fielding[col] = df_fielding[col].replace(["-", "DNB", "TDNB", "sub","absent"], np.nan)  # Replace invalid values
                df_fielding[col] = df_fielding[col].astype(float)

                df_fielding[col] = df_fielding[col].fillna(0)  # Filling na values with 0
                scores = df_fielding[col].dropna().values
                # Storing all scores as avg of 3 matches so that scores wont deviate more, This is the main idea of our model
                window_size = 3
                scores = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
                predictions[col] = train_and_predict(scores) # Finding the predicted fielding stats

    all_predictions[player_name] = predictions

# Converting the dictionary to DataFrame
predicted_df = pd.DataFrame.from_dict(all_predictions, orient="index")

# Printing the final DataFrame
print(predicted_df)

# Function to calculate dream11 score

def calculate_dream11_score(df):
    scores = {}

    for player, stats in df.iterrows():
        score = 0

        # Convert columns to numeric values safely
        stats["4s"] = pd.to_numeric(stats["4s"], errors='coerce')
        stats["6s"] = pd.to_numeric(stats["6s"], errors='coerce')
        stats["Runs"] = pd.to_numeric(stats["Runs"], errors='coerce')
        stats["Wkts"] = pd.to_numeric(stats["Wkts"], errors='coerce')
        stats["Econ"] = pd.to_numeric(stats["Econ"], errors='coerce')
        stats["SR"] = pd.to_numeric(stats["SR"], errors='coerce')
        stats["Ct"] = pd.to_numeric(stats["Ct"], errors='coerce')
        stats["St"] = pd.to_numeric(stats["St"], errors='coerce')
        stats["BF"] = pd.to_numeric(stats["BF"], errors='coerce')

        # Score based on runs
        if not pd.isna(stats["Runs"]):
            score += stats["Runs"]  # Runs add directly
            if stats["Runs"] >= 25: score += 4
            if stats["Runs"] >= 50: score += 4
            if stats["Runs"] >= 75: score += 4
            if stats["Runs"] >= 100: score += 4
            if stats["Runs"] == 0: score -= 2
        # Score for 4's
        if not pd.isna(stats["4s"]):
            score += stats["4s"] * 4

        # Score for 6's
        if not pd.isna(stats["6s"]):
            score += stats["6s"] * 6

        # Score for wicket
        if not pd.isna(stats["Wkts"]):
            score += stats["Wkts"] * 31

        # Score for Economy Rate
        if not pd.isna(stats["Econ"]):
            if stats["Econ"] < 5: score += 6
            elif 5 <= stats["Econ"] < 6: score += 4
            elif 6 <= stats["Econ"] < 7: score += 2
            elif 10 <= stats["Econ"] < 11: score -= 2
            elif 11 <= stats["Econ"] < 12: score -= 4
            elif stats["Econ"] >= 12: score -= 6

        # Score for strike Rate
        if not pd.isna(stats["SR"]):
            if stats["SR"] > 170 and stats["BF"] >= 10: score += 6
            elif 150 < stats["SR"] and stats["BF"] >= 10 <= 170: score += 4
            elif 130 < stats["SR"] and stats["BF"] >= 10 <= 150: score += 2
            elif 60 < stats["SR"] and stats["BF"] >= 10 <= 70: score -= 2
            elif 50 < stats["SR"] and stats["BF"] >= 10 <= 60: score -= 4
            elif stats["SR"] < 50 and stats["BF"] >= 10: score -= 6
        # Score for catches
        if not pd.isna(stats["Ct"]):
            score += stats["Ct"] * 8
        # Score for stumps
        if not pd.isna(stats["St"]):
            score += stats["St"] * 12
            
        scores[player] = score

    return pd.DataFrame.from_dict(scores, orient='index', columns=['Dream11 Score'])

# Finding Dream11 scores for the predicted stats
dream11_scores = calculate_dream11_score(predicted_df)
print(dream11_scores)

# Merging DataFrames
df = main_df.merge(dream11_scores, left_on="Player Name", right_index=True)
df = df.merge(predicted_df, left_on="Player Name", right_index=True)

# Sorting by Dream11 Score (Highest First)
df = df.sort_values(by="Dream11 Score", ascending=False)

# Initializing the team
final_team = []
selected_roles = {"BAT": 0, "BOWL": 0, "WK": 0, "ALL": 0}
selected_teams = set()

# Ensuring that there is one player per role
for role in ["BAT", "BOWL", "WK", "ALL"]:
    player = df[df["Player Type"] == role].iloc[0]  # Select the first player of that role
    final_team.append(player)
    selected_roles[role] += 1
    selected_teams.add(player["Team"])

# Ensuring that there is one player per team
for team in df["Team"].unique():
    if team not in selected_teams:
        player = df[df["Team"] == team].iloc[0]  # Select the first player from that team
        final_team.append(player)
        selected_roles[player["Player Type"]] += 1
        selected_teams.add(team)

# Filling remaining players based on higher dreamscore
remaining_players = df[~df["Player Name"].isin([p["Player Name"] for p in final_team])]

for _, player in remaining_players.iterrows():
    if len(final_team) >= 11:
        break
    final_team.append(player)
    selected_roles[player["Player Type"]] += 1
    selected_teams.add(player["Team"])

# Converting final team to dataframe
final_team_df = pd.DataFrame(final_team)

final_team_df["C/VC"] = "NA"

# Assigning the highest dream11 score player as captain
final_team_df.loc[final_team_df["Dream11 Score"].idxmax(), "C/VC"] = "C"

# Assigning the second-highest dream11 score player as vice-captain
second_highest_idx = final_team_df["Dream11 Score"].nlargest(2).index[-1]
final_team_df.loc[second_highest_idx, "C/VC"] = "VC"

final_team_df["C/VC"] = pd.Categorical(final_team_df["C/VC"], categories=["C", "VC", "NA"], ordered=True)

# Sorting DataFrame to Ensure C & VC Are at the Top
final_team_df = final_team_df.sort_values(by="C/VC")

# Selecting required columns which need to be shown in output
final_team_df = final_team_df[["Player Name", "Team", "C/VC"]]

# Print Final Team
print(final_team_df)

# Saving results in a csv
final_team_df.to_csv('CricTensors_Output.csv',index=False )

