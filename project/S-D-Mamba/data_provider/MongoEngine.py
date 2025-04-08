from pymongo import MongoClient
from pymongo.errors import ConnectionFailure


class DBEngine:
    __DBTABLES__ = {}

    def __init__(
        self,
        url="mongodb://root:root@127.0.0.1:27017",
        db="binance",
    ):
        try:
            self.client = MongoClient(url)
            self.client.admin.command("ping")
            self.db = self.client[db]
        except ConnectionFailure:
            raise Exception("Unable to connect to the server.")

    def getTop(self, collection="kline", keys=[("open_time", -1)], match={}):
        """_summary_

        Args:
            collection (str, optional): _description_. Defaults to "kline".
            keys (list, optional): _description_. Defaults to [("open_time", -1)]. -1: desc, 1: asc.
        Returns:
            _type_: _description_
        """
        # Example query function
        collection = self.db[collection]
        result = collection.find_one(sort=keys, **match)
        return result if result else None

    def getLength(self, collection="kline", match={}):
        """_summary_

        Args:
            collection (str, optional): _description_. Defaults to "kline".
            match (dict, optional): _description_. Defaults to {}.

        Returns:
            _type_: _description_
        """
        # Example query function
        collection = self.db[collection]
        return collection.count_documents(match)


if __name__ == "__main__":
    # from utils import convert_to_seconds

    # db = DBEngine(ip="10.68.34.200")
    # interval = convert_to_seconds("1s")
    # db = DBEngine(ip="192.168.101.14")
    # print(db.getMaxOpenTime("BTCUSDT", 1))
    # print(db.getLength("BTCUSDT", interval))
    # missing_intervals = db.check_continuous("BTCUSDT", 1)
    # for i in missing_intervals:
    #     continue
    pass
