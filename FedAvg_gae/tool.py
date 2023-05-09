import hashlib
import math
def hash(node):

    return hashlib.sha3_256(bytes(str(node), encoding='utf-8')).hexdigest()