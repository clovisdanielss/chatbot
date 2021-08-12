import json
import random

from repository.default_repository import DefaultRepository


class FakeRepository(DefaultRepository):

    def __init__(self, path):
        self.path = path

    def save(self, user_id, data):
        super(FakeRepository, self).save(user_id, data)
        user_id = str(user_id)
        file_data = None
        with open(self.path) as file:
            file_data = json.load(file)
            has_found = False
            if user_id in file_data.keys():
                file_data[user_id].append(data.__dict__)
                has_found = True
            if not has_found:
                file_data[user_id] = []
                file_data[user_id].append(data.__dict__)
        with open(self.path, "w") as file:
            json.dump(file_data, file)
