from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config

# loading the nlu training samples
training_data = load_data("C:/Users/yasmin/Desktop/PFE/Rasa/data/nlu_data.md")

# trainer to educate our pipeline
trainer = Trainer(config.load("C:/Users/yasmin/Desktop/PFE/Rasa/config.yml"))

# train the model
interpreter = trainer.train(training_data)

# store it for future use
model_directory = trainer.persist("C:/Users/yasmin/Desktop/PFE/Rasa/models/nlu", fixed_model_name="model")

