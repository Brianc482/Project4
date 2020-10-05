import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

#Read the datasets from GitHub
data = pd.read_csv("https://raw.githubusercontent.com/fivethirtyeight/nfl-elo-game/master/data/nfl_games.csv")
initial = pd.read_csv("https://raw.githubusercontent.com/fivethirtyeight/nfl-elo-game/master/data/initial_elos.csv")
print(data.head(20))

#Variables to be used in data manipulation
years = []
months = []
days = []
team1 = []
team2 = []

#Using initial_elos.csv, adapt the current team names into a numerical expression
for index, row in data.iterrows():
    date = row.date.split('-')
    years.append(date[0])
    months.append(date[1])
    days.append(date[2])
    team1.append(initial.index[initial['team'] == row['team1']].values[0])
    team2.append(initial.index[initial['team'] == row['team2']].values[0])
    
#Add the newly columns to the dataset
data.insert(1, 'year', years)
data.insert(2, 'month', months)
data.insert(3, 'day', days)
data.insert(9, 'team1New', team1)
data.insert(10, 'team2New', team2)

# Remove unwanted columns
newData = data.drop(columns=['date', 'team1', 'team2'])

#Begin initiliazing testing and training samples
inputs = newData[['year', 'month', 'day', 'season', 'neutral', 'playoff', 'team1New', 
                  'team2New', 'elo1', 'elo2', 'elo_prob1', 'result1']].values
trainIn,   testIn   = train_test_split(inputs, test_size=0.2, random_state=0)
trainOut1, testOut1 = train_test_split(newData['score1'].values, test_size=0.2, random_state=0)
trainOut2, testOut2 = train_test_split(newData['score2'].values, test_size=0.2, random_state=0)

#Section utilizes the Random Forest Regression to determine the scores
random1 = RandomForestRegressor(n_estimators=100, random_state=0).fit(trainIn, trainOut1)
random2 = RandomForestRegressor(n_estimators=100, random_state=0).fit(trainIn, trainOut2)

RandomScore1 = round(random1.score(testIn, testOut1) * 100, 2)
RandomScore2 = round(random2.score(testIn, testOut2) * 100, 2)
RandomAVG = round((RandomScore1 + RandomScore2) / 2, 2)

#Display results of regression
print("\nUtilizing Random Forest Regression we find:\n")
print('Team 1 scores were correct {:.2f}%'.format(RandomScore1), "of the time")
print('Team 2 scores were correct {:.2f}%'.format(RandomScore2), "of the time")
print('Average of scores predicted correctly {:.2f}%'.format(RandomAVG))

#Section utilizes the Linear Regression to determine the scores
linear1 = LinearRegression().fit(trainIn, trainOut1)
linear2 = LinearRegression().fit(trainIn, trainOut2)

LinearScore1 = round(linear1.score(testIn, testOut1) * 100, 2)
LinearScore2 = round(linear2.score(testIn, testOut2) * 100, 2)
LinearAVG = round((LinearScore1 + LinearScore2) / 2, 2)

#Display results of regression
print("\nUtilizing Linear Regression we find:\n")
print('Team 1 scores were correct {:.2f}%'.format(LinearScore1), "of the time")
print('Team 2 scores were correct {:.2f}%'.format(LinearScore2), "of the time")
print('Average of scores predicted correctly {:.2f}%'.format(LinearAVG))

#Display the difference in scores based on each regression performed
Diff1 = RandomScore1 - LinearScore1
Diff2 = RandomScore2 - LinearScore2
print("\nRandom Forest Regression vs. Linear Regression:\n")
print("Difference in first score: {:.2f}%".format(Diff1))
print("Difference in second score: {:.2f}%".format(Diff2))
print("Overall difference in regression models: {:.2f}".format((Diff1+Diff2)/2))