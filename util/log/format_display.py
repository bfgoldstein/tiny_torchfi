class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class fg:
    BLACK   = '\033[30m'
    RED     = '\033[31m'
    GREEN   = '\033[32m'
    YELLOW  = '\033[33m'
    BLUE    = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN    = '\033[36m'
    WHITE   = '\033[37m'
    RESET   = '\033[39m'
    ENDC = '\033[0m'

class bg:
    BLACK   = '\033[40m'
    RED     = '\033[41m'
    GREEN   = '\033[42m'
    YELLOW  = '\033[43m'
    BLUE    = '\033[44m'
    MAGENTA = '\033[45m'
    CYAN    = '\033[46m'
    WHITE   = '\033[47m'
    RESET   = '\033[49m'
    ENDC = '\033[0m'

class style:
    BRIGHT    = '\033[1m'
    DIM       = '\033[2m'
    NORMAL    = '\033[22m'
    ENDC = '\033[0m'
    
    
# TODO: Add GNMT arguments
def displayConfig(args):
    # loging configs to screen
    from util.log import logConfig
    logConfig("model", "{}".format(args.arch))
    logConfig("quantization", "{}".format(args.quantize))
    if args.quantize:
        logConfig("mode", "{}".format(args.quant_mode))
        logConfig("# bits features", "{}".format(args.quant_bacts))
        logConfig("# bits weights", "{}".format(args.quant_bwts))
        logConfig("# bits accumulator", "{}".format(args.quant_baccum))
        logConfig("clip-acts", "{}".format(args.quant_cacts))
        logConfig("per-channel-weights", "{}".format(args.quant_channel))
        logConfig("model-activation-stats", "{}".format(args.quant_stats_file))
        logConfig("clip-n-stds", "{}".format(args.quant_cnstds))
        logConfig("scale-approx-mult-bits", "{}".format(args.quant_scalebits))
    logConfig("injection", "{}".format(args.injection))
    if args.injection:
        logConfig("layer", "{}".format(args.layer))
        logConfig("bit", "{}".format(args.bit))
        logConfig("location:", "  ")
        logConfig("\t features ", "{}".format(args.fiFeats))
        logConfig("\t weights ", "{}".format(args.fiWeights))
        if not(args.fiFeats ^ args.fiWeights): 
            logConfig(" ", "Setting random mode.")
    logConfig("pruned", "{}".format(args.pruned))
    logConfig("prune compensate", "{}".format(args.prune_compensate))
    if args.pruned:
        logConfig("checkpoint from ", "{}".format(args.pruned_file))
    logConfig("batch size", "{}".format(args.batch_size))