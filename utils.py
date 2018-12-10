import sys, os
from shutil import which

def is_valid_binary(binary):
    return which(binary) != None

