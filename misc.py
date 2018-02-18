import sys
import time

# Print iterations progress
# Originally from Stack Overflow, but with a couple of fixes to work in PyCharm:
# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='▋', start_time_seconds=0):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    progress = iteration / float(total)
    percent = ("{0:." + str(decimals) + "f}").format(100 * progress)
    fl = int(length * iteration // total)  # length of filled chars
    estimate = 0.0

    if start_time_seconds > 0 and progress > 0.0:
        current_time_seconds = int(round(time.time()))
        estimate = current_time_seconds - start_time_seconds
        estimate = int(round(estimate / progress * (1- progress)))

    bar = fill * fl + ' ' * (length - fl)
    sys.stdout.write('\r%s |%s| %s%% %s (remaining: ~%s s)'
                     % (prefix, bar, percent, suffix, estimate))
    # Print New Line on Complete
    if iteration == total:
        print()

    sys.stdout.flush()

