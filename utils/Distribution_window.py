import os

from Distribution import getNameValue


def analysis(leaf: str) ->int :

    if os.path.isdir(leaf) is False:
        return print("Argument {} is not a directory".format(leaf))
    leafName = os.path.join(leaf, '')
    getNameValue(leafName)
    
    return 0