# message types
# dict object, single timepoint with multichannels
MESSAGE_TYPE_TIME_SAMPLE  = "TIME_SAMPLE"
# numpy matrix object: base64 encoded matrix of floats
MESSAGE_TYPE_MATRIX       = "MATRIX"
# message type used to issue control commands
MESSAGE_TYPE_COMMAND    = "COMMAND"

# data types
DATA_TYPE_CLASS_LABELS    = "CLASS_LABELS"
DATA_TYPE_RAW_DATA        = "RAW_DATA"
DATA_TYPE_RAW_COORDS      = "RAW_COORDS"
DATA_TYPE_LABELED_DATA    = "LABELED_DATA"
DATA_TYPE_LABELED_COORDS  = "LABELED_COORDS"
DATA_TYPE_STRING          = "STRING"

class colors:
    ENDC = '\033[0m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'

    BG_RASPBERRY = '\033[46m'

    DARK_RED = '\033[0;31m'
    RED = '\033[91m'
    ORANGE = '\033[1m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    DARK_GREEN = '\033[0;32m'
    CYAN = '\033[96m'
    MAGENTA = '\033[0;35m'
    RASPBERRY = '\033[0;36m'
    DARK_PURPLE = '\033[0;34m'

    GOLD = '\033[0;33m'
    SILVER = '\033[0;37m'
    GRAY = '\033[90m'

    BOLD_RED = '\033[1;31m'
    BOLD_ORANGE = '\033[1m'
    BOLD_YELLOW = '\033[1;33m'
    BOLD_GREEN = '\033[1;32m'
    BOLD_CYAN = '\033[1;36m'
    BOLD_BLUE  = '\033[1;34m'
    BOLD_PURPLE = '\033[1;35m'
    BOLD_GRAY  = '\033[1;30m'

    CYCLE = [YELLOW, GREEN, ORANGE,  RASPBERRY, DARK_RED, CYAN, RED, DARK_GREEN, MAGENTA, DARK_PURPLE]