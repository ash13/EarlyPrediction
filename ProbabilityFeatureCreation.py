import pandas as pd
import math
import pickle
import time

# Column Names
answerTextFeature = 'answer_text'
problemIDFeature = 'problem_id'
orderFeature = 'ordering'
studentIDFeature = 'user_id'
actionFeature = 'action_name'

#Action Names
actionAnswer = 'answer'

studentFirstAction = 2
threshold = 100

def analyzeAnswerProbabilities(problem):
    answerStatistics = {}
    onlyAnswers = problem[problem[actionFeature] == actionAnswer]
        
    firstAnswersCount = onlyAnswers[onlyAnswers[orderFeature] == studentFirstAction][problemIDFeature].count()
    restAnswersCount = onlyAnswers[onlyAnswers[orderFeature] != studentFirstAction][problemIDFeature].count()
    totalAnswersCount = onlyAnswers[problemIDFeature].count()
        
    if (firstAnswersCount + restAnswersCount !=  totalAnswersCount):
        raise Exception(problemID, "Something went wrong. Should not happen")
    
    for answerText, answer in onlyAnswers.groupby(answerTextFeature):
        
        firstAnswerProbability = np.nan
        restAnswerProbability = np.nan
            
        if (firstAnswersCount != 0): 
            firstAnswerProbability = float(answer[answer[orderFeature] == studentFirstAction][answerTextFeature].count()) / firstAnswersCount
        if (restAnswersCount != 0):   
            restAnswerProbability  = float(answer[answer[orderFeature] != studentFirstAction][answerTextFeature].count()) / restAnswersCount
                
        allAnswerProbability  = float(answer[answerTextFeature].count()) / totalAnswersCount
        
        answerStatistics[answerText] = [firstAnswerProbability,  restAnswerProbability, allAnswerProbability]
    return answerStatistics
    

def analyzeProblemProbabilities(studentRecords):
    answerFrequencies = {}
    actionFrequencies = {}
    
    firstCount = 0
    restCount = 0
    totalCount = 0
    
    belowThresholdColumns = [actionFeature, "first_action_count", "rest_action_count",  "all_action_count"]
    belowThresholdStatistics = pd.DataFrame(columns=belowThresholdColumns)
    
    
    for problemID, problem in studentRecords.groupby(problemIDFeature):

        answerStatistics = analyzeAnswerProbabilities(problem)
        if answerStatistics:   
            answerFrequencies[problemID] = answerStatistics
        
        actionStatistics = {}
        firstActionCount = problem[problem[orderFeature] == studentFirstAction][problemIDFeature].count()
        restActionCount = problem[problem[orderFeature] != studentFirstAction][problemIDFeature].count()
        totalActionCount = problem[problemIDFeature].count()

        numberStudents = len(problem[studentIDFeature].unique())
        for actionType, action in problem.groupby(actionFeature):
            thisFirstActionCount = float(action[action[orderFeature] == studentFirstAction][actionFeature].count())
            thisRestActionCount = float(action[action[orderFeature] != studentFirstAction][actionFeature].count())
            thisAllActionCount = float(action[actionFeature].count())
            
            if(numberStudents >= threshold):
                firstAction = np.nan
                restAction = np.nan
            
                if (firstActionCount != 0): 
                    firstAction = thisFirstActionCount / firstActionCount
                if (restActionCount != 0):  
                    restAction = thisRestActionCount / restActionCount
                
                allAction  = float(thisAllActionCount) / totalActionCount
                actionStatistics[actionType] = [firstAction,  restAction, allAction]

            else:
                firstCount += thisFirstActionCount
                restCount += thisRestActionCount
                totalCount += thisAllActionCount
                
                thisAction = pd.Series([actionType, thisFirstActionCount,  thisRestActionCount, thisAllActionCount], index=belowThresholdColumns)
                belowThresholdStatistics = belowThresholdStatistics.append(thisAction, ignore_index=True)
        if(numberStudents >= threshold):
            actionFrequencies[problemID] = actionStatistics
    
    actionStatistics = {}
    for actionType, action in belowThresholdStatistics.groupby(actionFeature):
        firstAction = float(action['first_action_count'].sum()) / firstCount
        restAction = float(action['rest_action_count'].sum()) / restCount
        allAction = float(action['all_action_count'].sum()) / totalCount
        
        actionStatistics[actionType] = [firstAction,  restAction, allAction] 
    actionFrequencies['all'] = actionStatistics
        
    return answerFrequencies, actionFrequencies

def analyzeStudentProblems(answerFrequencies, actionFrequencies, studentRecords):
    probColumns = ['user_id', 'problem_id', 'problem_log_id', 'ordering', 'probability_action', 'probability_action_action_count', 'probability_answer', 'probability_answer_action_count', 'log_likelihood_cumulative_answer']
    prob = pd.DataFrame(columns=probColumns)
    
    for problemID, problems in studentRecords.groupby(problemIDFeature): 
        print(problemID)
        
        problemAnswerStats = {}
        problemActionStats = {}
        
        if problemID in answerFrequencies:
            problemAnswerStats = answerFrequencies[problemID]
        if problemID in actionFrequencies:
            problemActionStats = actionFrequencies[problemID]
        else:
            problemActionStats = actionFrequencies['all']
        for studentID, student in problems.groupby(studentIDFeature): 
            cumulativeLL = 0
            student = student.sort_values(by=[orderFeature]) 
            
            for index, row in student.iterrows():
                answer = row[answerTextFeature]
                action = row[actionFeature]
                probAnswerCount = np.nan
                probActionCount = np.nan
                probAnswer = np.nan
                probAction = np.nan
                
                if (row[actionFeature] == actionAnswer):
                    if (row[orderFeature] == 1):
                        probAnswerCount = problemAnswerStats[answer][0]
                        probActionCount = problemActionStats[action][0]
                    else: 
                        probAnswerCount = problemAnswerStats[answer][1]
                        probActionCount = problemActionStats[action][1]
                    probAnswer = problemAnswerStats[answer][2]
                    probAction = problemActionStats[action][2]

                    cumulativeLL += math.log(probAnswer)
                else:
                    if (row[orderFeature] == 1):
                        probActionCount = problemActionStats[action][0]
                    else: 
                        probActionCount = problemActionStats[action][1]
                    probAction = problemActionStats[action][2]
                thisRecord = pd.Series([row[studentIDFeature], row[problemIDFeature], row['problem_log_id'], row[orderFeature], probAction, probActionCount, probAnswer, probAnswerCount, cumulativeLL], index=probColumns)
                prob = prob.append(thisRecord, ignore_index=True)
    return prob


studentRecords = pd.read_csv('skill_builders_16_17.csv')
sampleRecords = studentRecords[studentRecords[actionFeature].isin(["answer", "hint", "scaffold", "answerhint"])]

# Get action and answer probabilities (~10 minutes)
start = time.time()
answerFrequencies, actionFrequencies = analyzeProblemProbabilities(sampleRecords)
end = time.time()
print ("Getting probabilities took: ", end - start)

pickle.dump( answerFrequencies, open( "answerFrequencies.p", "wb" ) )
pickle.dump( actionFrequencies, open( "actionFrequencies.p", "wb" ) )

# For each row, calculate action and answer probabilities (~3.5 hours)          
start = time.time()
prob = analyzeStudentProblems(answerFrequencies, actionFrequencies, sampleRecords)
end = time.time()
print ("Calculating probabilities took: ", end - start)

pickle.dump( prob, open( "actionAnswerProbabilities.p", "wb" ) )