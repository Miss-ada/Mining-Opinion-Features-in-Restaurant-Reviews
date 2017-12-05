import json
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


filenameAFINN = 'AFINN/AFINN-111.txt'
afinn = dict(map(lambda (w, s): (w, int(s)), [
            ws.strip().split('\t') for ws in open(filenameAFINN) ]))


dict = ["the service", {"attentive": 3, "great": 2, "slow": 1, "real": 1, "communal": 1, "pretty": 1, "poor": 1, "impressed": 1, "pleasant": 1, "fine": 1, "solid": 1, "impeccable": 1, "questionable": 1, "fantastic": 1, "friendly": 1, "late": 1, "top": 1, "new": 1, "superb": 1, "personal": 1, "surprised": 1, "excellent": 1, "bright": 1}]
def sentiment(dict):
    """
    Returns a float for sentiment strength based on the input text.
    Positive values are positive valence, negative value are negative valence.
    """
    result = {}
    sentiment = 0
    sumFreq = 0
    for adj, freq in dict[1].iteritems():
        sumFreq += freq
        if adj in afinn.keys():
            sentiments = afinn[adj]
            sentiment += float(sentiments) * freq
        #put in 0-5 scale
    sumSentiment = sentiment/sumFreq/2 + 2.5
    result[dict[0]] = sumSentiment
    return result
    # if sentiments:
    #     # How should you weight the individual word sentiments?
    #     # You could do N, sqrt(N) or 1 for example. Here I use sqrt(N)
    #     sentiment = float(sum(sentiments)) / math.sqrt(len(sentiments))


if __name__ == '__main__':
    json_data = open('candidates_attributes.txt').read()
    data = json.loads(json_data)
    #output
    output = []
    for dict in data:
        output.append(sentiment(dict))

    with open('sentiment_score.txt', 'w') as outfile:
        json.dump(output, outfile)