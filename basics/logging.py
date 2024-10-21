from datetime import datetime
import time
class Logging:
    def __init__(self):
        self.logs=[]

    def log(self,message,level):
        time_stamp=datetime.now().strftime('%Y-%m-%d-%H-%M-%S ')
        log_entry=time_stamp+f'{level}: {message}'
        self.logs.append(log_entry)

    def info(self,message):
        self.log(message,"INFO")
    
    def warning(self,message):
        self.log(message,"WARNING")
    
    def error(self,message):
        self.log(message,"ERROR")

    def log_exception(self, excpetion):
        error_message=f'Exception occured: {str(excpetion)}'
        self.log(error_message, "ERROR")

    def get_logs(self):
        return self.logs

logger=Logging()

def example_function():
    logger.info("start the program")
    time.sleep(0.5)
    logger.warning("The software version is outdated")
    time.sleep(0.7)
    try:
        x=int("not a int")
    except ValueError as e:
        logger.error("An error occured")
        logger.log_exception(e)


if __name__=='__main__':
    example_function()
    all_logs=logger.get_logs()
    print("All logs:")
    for log in all_logs:
        print(log)
