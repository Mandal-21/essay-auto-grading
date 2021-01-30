from pymongo import MongoClient
import base64


class nlpDB:
    def __init__(self):
        try:
            self.client = MongoClient("mongodb+srv://amar:" + str(base64.b64decode("YW1hcg==").decode("utf-8")) + "@cluster0.w4axz.mongodb.net/<dbname>?retryWrites=true&w=majority")
            self.db = self.client.get_database('nlp_db')
            self.records = self.db['nlp_records']
        except Exception as e:
            print(e)


    # To add new row
    def updateDataBase(self, essay, essay_set, result):
        row = {}
        row['essay_token_pad'] = essay
        row['essay_set'] = essay_set
        # row['sent_count'] = sent_count
        # row['word_count'] = word_count
        row['result'] = result
        self.records.insert_one(row)