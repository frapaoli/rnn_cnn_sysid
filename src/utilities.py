# general purpose utilities
import pytz
from datetime import datetime


def get_model_purpose():

  print("*********************************************")
  print("*                                           *")
  print("*    AI-BASED SYSTEM IDENTIFICATION TOOL    *")
  print("*                                           *")
  print("*********************************************\n")

  print("Choose the purpose of models training:\n")
  print("- control: learn how the inputs affect the outputs in the short-term\n")
  print("- prediction: learn how the inputs affect the outputs in the long-term\n")

  while True:

    model_purpose = input("Insert the purpose of the trained model: ")

    model_purpose = model_purpose.lower()
    
    if model_purpose in ["control", "prediction"]:
      break
    else:
      print("Please enter a valid model purpose (i.e., \'control\' or \'prediction\').\n")

  print("\n\nThe model will be used for:\n")

  asterisks = '*' * len(model_purpose)

  print("***" + asterisks + "***")
  print("*  " + model_purpose + "  *")
  print("***" + asterisks + "***")

  return model_purpose


def mkdir_model_checkpoints(models_path):

  # get timestamp of current date-time (CET time zone)
  current_datetime = datetime.now(pytz.timezone('Europe/Berlin'))
  current_datetime_str = current_datetime.strftime("%Y%m%d_%H%M%S")

  # create a directory where to store the new model checkpoints during its training
  model_checkpoints_path = models_path + '/model_' + current_datetime_str

  return model_checkpoints_path
