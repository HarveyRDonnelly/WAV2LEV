
import logging

WIDTH = 80

class Logger():
    
    def __init__(
        self,
        script_name="Default",
        organisation="Computational Linguistics Group, University of Toronto",
        version="1.1.0",
        author="Harvey Donnelly"):
        
        self.script_name = script_name
        self.organisation = organisation
        self.version = version
        self.author = author
        
        self.red = "\x1b[38;5;196m"
        self.blue = "\x1b[38;5;44m"
        self.pink = "\x1b[38;5;175m"
        self.purple = "\033[1;95m"
        self.green = "\x1b[38;5;151m"
        self.grey = "\x1b[38;5;189m"
        self.white = "\x1b[1;97m"
        self.reset = "\x1b[0m"
        
    def welcome(self):
        
        welcome_banner = "\n"\
        + self.purple\
        + "~" * WIDTH\
        + self.blue \
        + """

██     ██  █████  ██    ██     ██████      ██      ███████ ██    ██ 
██     ██ ██   ██ ██    ██          ██     ██      ██      ██    ██ 
██  █  ██ ███████ ██    ██      █████      ██      █████   ██    ██ 
██ ███ ██ ██   ██  ██  ██      ██          ██      ██       ██  ██  
 ███ ███  ██   ██   ████       ███████     ███████ ███████   ████   
                                                                                                                                                                                                                                 
"""\
        + self.purple\
        + "~" * ((WIDTH - len(self.script_name) - 10)//2)\
        + " " * 5\
        + self.white\
        + self.script_name\
        + " " * 5\
        + self.purple\
        + "~" * ((WIDTH - len(self.script_name) - 10)//2)\
        + self.white\
        + "\n\n"\
        + "[ Authored by " + self.author + " ]\n"\
        + "[ Developed @ " + self.organisation + " ]\n"\
        + "[ Version " + str(self.version) + " ]\n\n"\
        + self.purple\
        + "~" * WIDTH\
        + "\n\n"\
        + self.reset
        
        print(welcome_banner)
        
    def complete(self):
        compl = "     COMPLETE     "
        complete_banner = "\n"\
            + self.purple\
            + "~" * ((WIDTH - len(compl))//2)\
            + self.white\
            + compl\
            + self.purple\
            + "~" * ((WIDTH - len(compl))//2)\
            + "\n\n"\
            + self.white\
            + "[ Developed @ " + self.organisation + " ]\n"\
            + "[ Version " + str(self.version) + " ]\n\n"\
            + self.white\
            + self.purple\
            + "~" * WIDTH\
            + "\n\n"\
            + self.reset
        
        print(complete_banner)
        
    def info(self, info_str):
        info_text = self.green\
            + "[INFO]"\
            + self.grey\
            + " => "\
            + self.white\
            + info_str\
            + "\n"\
            + self.white\
            + self.reset
        
        print(info_text)
        
    def debug(self, debug_str):
        debug_text = self.green\
            + "[DEBUG]"\
            + self.grey\
            + " => "\
            + self.white\
            + debug_str\
            + "\n"\
            + self.white\
            + self.reset
        
        print(debug_text)
        
    def error(self, error_str):
        error_text = self.red\
            + "[ERROR]"\
            + self.grey\
            + " => "\
            + self.white\
            + error_str\
            + "\n"\
            + self.white\
            + self.reset
        
        print(error_text)
        
if __name__ == "__main__":
    log = Logger()
    log.welcome()
    log.debug("Test")
    log.complete()