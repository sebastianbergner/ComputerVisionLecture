import pickle

def save_data(data, name):
    try:
        with open(name, "wb") as f:
            pickle.dump(data, f)
    except Exception as ex:
        print("Error during saving ",name," error:", ex)

def load_data(name):
    try:
        with open(name, "rb") as f:
            data = pickle.load(f)
            return data
    except Exception as ex:
        print("Error during loading ",name," error:", ex)


