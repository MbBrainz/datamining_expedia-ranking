import numpy as np
from pandas import DataFrame
from sklearn.metrics import ndcg_score

def evaluate_score(df: DataFrame):
    """Evaluate the score of based on the predicted ranks and the computed rank. 
    Itll go over the srch_ids and evaluate the scores using sklearn.metrics.ndcg_score

    Args:
        df (_type_): The input dataframe with columns ["srch_id","prop_id", "scores", "predict"]

    Returns:
        float: resulting score
    """
    search_ids = df.srch_id.unique() # get unique id's
    score = 0
    
    
    for search_id in search_ids:
        y_scores =df.loc[df["srch_id"] == search_id]["scores"].to_numpy() # get the true rank for search query as numpy
        y_predict = df.loc[df["srch_id"] == search_id]["predict"].to_numpy()
        y_scores = np.expand_dims(y_scores, axis=0)# add dimension
        y_predict = np.expand_dims(y_predict, axis=0)
        score += ndcg_score(y_scores, y_predict) # calculate score
    score = score/len(search_ids)
    
    
    return score

